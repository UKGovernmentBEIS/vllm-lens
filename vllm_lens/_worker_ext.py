"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers.types import Hook, HookContext, SteeringVector

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)

_DTYPE_LIST = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.float64,
]
_DTYPE_TO_IDX_MAP = {d: i for i, d in enumerate(_DTYPE_LIST)}


def _dtype_to_idx(dtype: torch.dtype) -> int:
    return _DTYPE_TO_IDX_MAP.get(dtype, 0)


_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _discover_layer_modules(worker: Any) -> dict[int, torch.nn.Module] | None:
    """Map global decoder-layer index → decoder-layer module on this rank.

    Uses vLLM's ``static_forward_context`` registry: every attention-like
    layer (``AttentionLayerBase`` — standard attention, MLA, Mamba-style
    mixers, …) registers itself there at construction time under its full
    module prefix (e.g. ``model.layers.7.self_attn.attn``).  The decoder
    layer is the prefix truncated at its first integer segment, which is a
    *global* layer index even under pipeline parallelism (non-owned layers
    never construct, so they never register).

    Returns ``None`` when the registry is unavailable or yields an
    incoherent mapping; callers fall back to ``_get_layers``.
    """
    try:
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
    except ImportError:
        return None

    registry = getattr(
        getattr(worker, "compilation_config", None), "static_forward_context", None
    )
    if not registry:
        return None

    model = worker.model_runner.model
    layer_map: dict[int, torch.nn.Module] = {}
    for prefix, module in registry.items():
        # FusedMoE and multimodal-encoder layers also register here.
        if not isinstance(module, AttentionLayerBase):
            continue
        # Encoder / cross-attention is not part of the decoder residual
        # stream.  Mamba-style mixers carry no attn_type and are kept so
        # hybrid models still get every decoder layer hooked.
        if getattr(module, "attn_type", "decoder") != "decoder":
            continue
        segments = prefix.split(".")
        idx_pos = next((i for i, seg in enumerate(segments) if seg.isdigit()), None)
        if idx_pos is None:
            logger.warning(
                "Registered layer %r has no integer layer index in its prefix; "
                "falling back to attribute-path layer discovery.",
                prefix,
            )
            return None
        layer_idx = int(segments[idx_pos])
        parent_path = ".".join(segments[: idx_pos + 1])
        try:
            layer = model.get_submodule(parent_path)
        except AttributeError:
            logger.warning(
                "Could not resolve decoder layer %r from registered layer %r; "
                "falling back to attribute-path layer discovery.",
                parent_path,
                prefix,
            )
            return None
        existing = layer_map.get(layer_idx)
        if existing is not None and existing is not layer:
            logger.warning(
                "Layer index %d resolves to two different modules (%r); "
                "falling back to attribute-path layer discovery.",
                layer_idx,
                prefix,
            )
            return None
        layer_map[layer_idx] = layer

    return layer_map or None


def _get_total_num_layers(worker: Any) -> int:
    """Total decoder layers across all PP ranks, for layer-index validation."""
    try:
        return int(worker.model_config.get_total_num_hidden_layers())
    except Exception:
        return len(_get_layers(worker.model_runner.model))


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the transformer decoder layers by attribute-path traversal.

    Fallback for when the ``static_forward_context`` registry is
    unavailable (see ``_discover_layer_modules``).
    """
    # Module.__getattr__ returns Tensor | Module, so pyright can't narrow
    # through chained attribute access.  Use Any for duck-typed traversal.
    m: Any = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if (
        hasattr(m, "model")
        and hasattr(m.model, "decoder")
        and hasattr(m.model.decoder, "layers")
    ):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        "Expected model.language_model.model.layers, "
        "model.model.decoder.layers, or model.model.layers"
    )


def _find_steering_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[SteeringVector]:
    """Find all steering configs that apply to an internal request ID.

    Matches by ``"{external_id}-"`` prefix (async path: vLLM appends
    ``"-{random_suffix}"`` to external IDs) and by ``_steering_id``
    sentinel in ``extra_args`` (offline path).
    """
    results: list[SteeringVector] = []
    for external_id, configs in extension._steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    # Offline path stores a lightweight string key in extra_args
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id and steering_id in extension._steering_data:
            results.extend(extension._steering_data[steering_id])
    return results


def _find_hook_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find all hook definitions that apply to an internal request ID.

    Checks three sources (in order):
    1. Per-request hooks keyed by external ID prefix (async path).
    2. Per-request hooks keyed by ``_hook_id`` sentinel (offline path).
    3. Persistent hooks (apply to every request).
    """
    results = _find_hook_configs_no_persistent(extension, internal_req_id, extra_args)
    results.extend(extension._persistent_hooks)
    return results


def _find_hook_configs_no_persistent(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find per-request hook definitions only (excludes persistent hooks)."""
    results: list[Hook] = []
    for external_id, hooks in extension._hook_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(hooks)
    if extra_args:
        hook_id = extra_args.get("_hook_id")
        if hook_id and hook_id in extension._hook_data:
            results.extend(extension._hook_data[hook_id])
    return results


def norm_match(
    residual: torch.Tensor,
    steering: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale a steering vector to match the L2 norm of the residual stream.

    Norm matching approach from the Activation Oracles paper
    (arXiv:2512.15674):

        h'_i = h_i + ‖h_i‖ · v_i / ‖v_i‖

    This rescales the steering vector so its magnitude matches the
    residual before addition, ensuring activations of varying provenance
    are automatically scaled to a consistent magnitude.
    """
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
    norm_ref: torch.Tensor,
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.

    ``norm_ref`` is the tensor whose per-token L2 norm ``norm_match`` should
    match.  For fused-residual models (e.g. Qwen3) the layer returns
    ``(hidden_states, residual)`` and ``target`` is only ``hidden_states``
    (the MLP-delta half); the *true* residual stream is
    ``hidden_states + residual``, so ``norm_ref`` must be that full stream,
    not the MLP-delta half, or the steering vector is scaled to a far smaller
    norm than HF uses.  For non-fused / plain-tensor layers the caller passes
    ``norm_ref = target``.  Required (no default) so a forgotten reference
    fails at the call instead of silently scaling to the wrong
    MLP-delta-half norm.
    """
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)  # (hidden,) or (n_pos, hidden)

        if vec.dim() == 1:
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(norm_ref[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:
            # 3D: position-specific
            pos_indices = (
                cfg.position_indices
                if cfg.position_indices is not None
                else list(range(vec.shape[0]))
            )
            abs_end = abs_start + n_tokens
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    v = norm_match(norm_ref[rel], v)
                target[rel] = target[rel] + v * cfg.scale


def _apply_hook_delta(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None,
    hook_hidden: torch.Tensor,
    start: int,
    end: int,
    result: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Write a post-hook's modification into the layer output.

    Applies ``result - hook_hidden[start:end]`` as a delta onto
    ``modified_output`` (cloning the original ``output`` lazily on first
    write) and updates ``hook_hidden`` in place so later hooks in the same
    forward pass observe the change.  Returns the (possibly newly created)
    ``modified_output``.
    """
    delta = result - hook_hidden[start:end]
    if modified_output is None:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
        else:
            modified_output = output.clone()
    if isinstance(modified_output, tuple):
        modified_output[0][start:end] = modified_output[0][start:end] + delta
    else:
        modified_output[start:end] = modified_output[start:end] + delta
    hook_hidden[start:end] = result
    return modified_output


def _hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """Core hook logic, separated so _make_hook can wrap it in try/except."""
    if not is_forward_context_available():
        return None

    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return None

    req_ids = runner.input_batch.req_ids

    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
        if attn_metadata is None:
            return None
    # Hybrid models (e.g. Qwen3-Next with GatedDeltaNet) have multiple
    # attention metadata entries — some (like GDNAttentionMetadata) lack
    # query_start_loc.  Find one that has it.
    query_start_loc: Int[torch.Tensor, "num_reqs_plus1"] | None = None  # type: ignore[reportUndefinedVariable]
    for _meta in attn_metadata.values():
        if hasattr(_meta, "query_start_loc"):
            query_start_loc = getattr(_meta, "query_start_loc")
            break
    if query_start_loc is None:
        logger.warning(
            "No attention metadata with query_start_loc found "
            "(keys: %s). Skipping hook for this step.",
            list(attn_metadata.keys()),
        )
        return None

    # --- Phase 1: detect steering requests --------------------------
    per_req_steering: list[list[SteeringVector]] = []
    needs_steering = False
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        configs = _find_steering_configs(extension, req_id, extra)
        per_req_steering.append(configs)
        if configs:
            needs_steering = True

    # --- Phase 2: apply steering ------------------------------------
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if needs_steering:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
            target = modified_output[0]
            # Fused-residual layers: true residual stream is output[0]+output[1].
            # norm_match must reference the full stream, not the MLP-delta half.
            norm_ref = output[0] + output[1] if output[1] is not None else output[0]
        else:
            modified_output = output.clone()
            target = modified_output
            norm_ref = target

        # Retrieve seq_lens for absolute position calculation.
        # seq_lens may be a tensor or a list depending on vLLM version.
        seq_lens: Any = getattr(attn_metadata, "seq_lens", None)

        for i in range(num_reqs):
            if not per_req_steering[i]:
                continue
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            n_query = end - start
            # Absolute position of the first token in this forward pass
            if seq_lens is not None:
                sl = seq_lens[i]
                sl_val = sl.item() if isinstance(sl, torch.Tensor) else int(sl)
                abs_start = int(sl_val - n_query)
            else:
                abs_start = 0  # fallback: treat as prefill from position 0
            _apply_steering(
                per_req_steering[i], layer_idx, target, start, end, abs_start, norm_ref
            )

    # --- Phase 2.5: run generic (post) hooks -------------------------
    # Per-request and persistent hooks are stored in separate context
    # dicts (per-request contexts are cleaned up after each request;
    # persistent ones accumulate).  Within each dict, contexts are keyed
    # by the hook's position in its category list — so a pre-hook and a
    # post-hook at different positions never share a HookContext, and the
    # returned result index ("0", "1", ...) is stable regardless of how
    # many pre/post hooks a request mixes.  Pre-hooks are handled in
    # _pre_hook_inner using the same position keys.
    per_req_hooks: list[list[Hook]] = []
    needs_hooks = False
    persistent_hooks = extension._persistent_hooks
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        hooks = _find_hook_configs_no_persistent(extension, req_id, extra)
        per_req_hooks.append(hooks)
        if hooks or persistent_hooks:
            needs_hooks = True

    if needs_hooks:
        # Compute hidden_states (summed if tuple) same as Phase 3 does.
        hook_src = modified_output if modified_output is not None else output
        if isinstance(hook_src, tuple):
            hook_hidden = (
                hook_src[0] + hook_src[1] if hook_src[1] is not None else hook_src[0]
            )
        else:
            hook_hidden = hook_src
        # Clone to avoid aliasing — hooks read/write this independently.
        hook_hidden = hook_hidden.clone()

        def _run_post_category(
            hooks: list[Hook],
            store: dict[str, dict[int, HookContext]],
            req_id: str,
            start: int,
            end: int,
        ) -> None:
            """Run the post-hooks in one category list at this layer.

            Contexts live in ``store[req_id][position]``, created lazily.
            """
            nonlocal modified_output
            for pos, hook in enumerate(hooks):
                if hook.pre or not hook.has_layer(layer_idx):
                    continue
                ctxs = store.setdefault(req_id, {})
                ctx = ctxs.get(pos)
                if ctx is None:
                    ctx = HookContext()
                    ctxs[pos] = ctx
                ctx.layer_idx = layer_idx
                ctx.seq_len = end - start
                ctx.model = runner.model
                ctx._prefetched = extension._prefetched_params

                result = hook.fn(ctx, hook_hidden[start:end])
                if result is not None:
                    modified_output = _apply_hook_delta(
                        output, modified_output, hook_hidden, start, end, result
                    )

        for i in range(num_reqs):
            if not (persistent_hooks or per_req_hooks[i]):
                continue
            req_id = req_ids[i]
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            # Persistent hooks fire first (base layer); per-request hooks
            # see the persistent-modified state.
            _run_post_category(
                persistent_hooks,
                extension._persistent_hook_contexts,
                req_id,
                start,
                end,
            )
            _run_post_category(
                per_req_hooks[i], extension._hook_contexts, req_id, start, end
            )

    # --- Phase 3: capture activations (rank 0 only) -----------------
    if getattr(extension, "_should_capture", True):
        capture_src = modified_output if modified_output is not None else output
        hidden_states: Float[torch.Tensor, "total_tokens hidden_dim"]  # type: ignore[reportUndefinedVariable]
        if isinstance(capture_src, tuple):
            if capture_src[1] is not None:
                hidden_states = capture_src[0] + capture_src[1]
            else:
                hidden_states = capture_src[0]
        else:
            hidden_states = capture_src

        for i in range(num_reqs):
            req_id = req_ids[i]
            req_state = runner.requests.get(req_id)
            if req_state is None or req_state.sampling_params is None:
                continue
            extra = req_state.sampling_params.extra_args
            if not extra:
                continue

            output_residual_stream = extra.get("output_residual_stream")
            if output_residual_stream is None:
                continue
            # vllm_xargs passes values as strings; parse JSON lists.
            if isinstance(output_residual_stream, str):
                try:
                    output_residual_stream = json.loads(output_residual_stream)
                except (json.JSONDecodeError, ValueError):
                    pass  # treat as truthy (capture all layers)
            if (
                isinstance(output_residual_stream, list)
                and layer_idx not in output_residual_stream
            ):
                continue

            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            activation: Float[torch.Tensor, "seq_len hidden_dim"] = hidden_states[  # type: ignore[reportUndefinedVariable]
                start:end
            ].cpu()

            if req_id not in extension._captured_states:
                extension._captured_states[req_id] = {}
            layer_states = extension._captured_states[req_id]
            if layer_idx not in layer_states:
                layer_states[layer_idx] = []
            layer_states[layer_idx].append(activation)

    return modified_output


def _pre_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Run pre-hooks (hook.pre=True) on the layer input.

    Only runs generic hooks — steering and activation capture are
    post-hook operations and are not affected.
    """
    if not is_forward_context_available():
        return None

    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return None

    req_ids = runner.input_batch.req_ids
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
        if attn_metadata is None:
            return None
    query_start_loc: torch.Tensor | None = None
    for _meta in attn_metadata.values():
        if hasattr(_meta, "query_start_loc"):
            query_start_loc = getattr(_meta, "query_start_loc")
            break
    if query_start_loc is None:
        return None

    # Pre-hooks share the same context stores as post-hooks, keyed by the
    # hook's position in its category list.  A hook at a given position is
    # either pre or post (never both), so pre and post never collide on the
    # same key — this is what lets a request mix pre- and post-hooks safely.
    persistent_hooks = extension._persistent_hooks
    modified = False
    working = input_tensor

    def _run_pre_category(
        hooks: list[Hook],
        store: dict[str, dict[int, HookContext]],
        req_id: str,
        start: int,
        end: int,
    ) -> None:
        """Run the pre-hooks in one category list at this layer."""
        nonlocal working, modified
        for pos, hook in enumerate(hooks):
            if not hook.pre or not hook.has_layer(layer_idx):
                continue
            ctxs = store.setdefault(req_id, {})
            hctx = ctxs.get(pos)
            if hctx is None:
                hctx = HookContext()
                ctxs[pos] = hctx
            hctx.layer_idx = layer_idx
            hctx.seq_len = end - start
            hctx.model = runner.model
            hctx._prefetched = extension._prefetched_params

            result = hook.fn(hctx, working[start:end])
            if result is not None:
                if not modified:
                    working = input_tensor.clone()
                    modified = True
                working[start:end] = result

    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        per_req = _find_hook_configs_no_persistent(extension, req_id, extra)
        if not any(h.pre for h in persistent_hooks) and not any(h.pre for h in per_req):
            continue

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        _run_pre_category(
            persistent_hooks, extension._persistent_hook_contexts, req_id, start, end
        )
        _run_pre_category(per_req, extension._hook_contexts, req_id, start, end)

    return working if modified else None


def _make_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward hook closure for a specific layer index."""

    def hook(
        _module: torch.nn.Module,
        _input: object,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Forward hook: apply steering vectors then capture activations.

        Returns the modified output if any steering was applied, ``None``
        otherwise (so PyTorch leaves the original output untouched).
        """
        try:
            return _hook_inner(extension, layer_idx, output)
        except Exception:
            logger.warning(
                "vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True
            )
            return None

    return hook


def _make_pre_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward pre-hook closure for a specific layer index.

    vLLM decoder layers have signature
    ``forward(positions, hidden_states, residual)`` — the hidden states
    are at ``args[1]``, not ``args[0]``.
    """

    def hook(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...] | None:
        """Forward pre-hook: run user pre-hooks on the layer input."""
        try:
            # hidden_states is args[1] (args[0] is positions).
            hidden = args[1]
            result = _pre_hook_inner(extension, layer_idx, hidden)
            if result is not None:
                return args[:1] + (result,) + args[2:]
            return None
        except Exception:
            logger.warning(
                "vllm-lens pre-hook error on layer %d, skipping",
                layer_idx,
                exc_info=True,
            )
            return None

    return hook


class HiddenStatesExtension:
    """Mixin injected into vLLM's GPU Worker at runtime.

    Configured via the ``worker_extension_cls`` engine arg. vLLM dynamically
    adds this class as a base of Worker
    (``Worker.__bases__ += (HiddenStatesExtension,)``), so ``self`` is the
    Worker instance and its methods are callable via
    ``collective_rpc("method_name")``.

    It doesn't extend Worker directly — vLLM handles that injection.
    """

    if TYPE_CHECKING:
        model_runner: Any  # Provided by Worker at runtime
        rank: int
        parallel_config: ParallelConfig

    # Per-request captured activations:
    # internal_req_id → { layer_idx → [tensor, ...] }
    _captured_states: dict[
        str,
        dict[int, list[Float[torch.Tensor, "seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}
    _hooks_installed: bool = False

    # Per-request steering configs:
    # key (external_req_id or _steering_id) → list of SteeringVector
    _steering_data: dict[str, list[SteeringVector]] = {}

    # Per-request hook definitions:
    # key (external_req_id or _hook_id) → list of Hook
    _hook_data: dict[str, list[Hook]] = {}

    # Persistent hooks (apply to every request, not auto-cleaned):
    _persistent_hooks: list[Hook] = []

    # Per-request hook contexts, keyed by internal request ID then by the
    # hook's position in the per-request hook list:
    # internal_req_id → { hook_position → HookContext }
    _hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Persistent hook contexts (separate from per-request to avoid cleanup
    # conflicts), keyed the same way by position in the persistent hook list:
    # internal_req_id → { hook_position → HookContext }
    _persistent_hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Whether this rank should capture activations (only TP rank 0).
    _should_capture: bool = True

    def install_hooks(self) -> None:
        """Register a forward hook on every decoder layer. Idempotent.

        Hooks are installed on **all** TP ranks because steering must
        modify hidden states everywhere.  Activation *capture* is gated
        to rank 0 only via ``_should_capture``.

        Requires ``enforce_eager=True`` in engine args — otherwise
        ``@support_torch_compile`` would compile the forward graph and
        hooks won't fire.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared).
        # Do NOT reset _persistent_hooks — they may have been set via
        # set_persistent_hooks() before the first generate call.
        self._captured_states = {}
        self._steering_data = {}
        self._hook_data = {}
        if not isinstance(self.__dict__.get("_persistent_hooks"), list):
            self._persistent_hooks = []
        self._hook_contexts = {}
        self._persistent_hook_contexts = {}

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layer_map = _discover_layer_modules(self)
        if layer_map is None:
            layers = _get_layers(self.model_runner.model)
            layer_map = {
                layer_idx: layer
                for layer_idx, layer in enumerate(layers)
                if not isinstance(layer, PPMissingLayer)
            }
        for layer_idx, layer in sorted(layer_map.items()):
            layer.register_forward_pre_hook(_make_pre_hook(self, layer_idx))
            layer.register_forward_hook(_make_hook(self, layer_idx))

    # ------------------------------------------------------------------
    # Steering data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store steering vectors for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``SteeringVector`` instances, validates layer indices
        against the model, moves activation tensors to GPU in the model's
        dtype, and stores them keyed by *key* (an external request ID or a
        synthetic ``_steering_id``).
        """
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)

        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype

        num_layers = _get_total_num_layers(self)
        vectors: list[SteeringVector] = []

        for sv in sv_list:
            for idx in sv.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )

            vectors.append(
                sv.model_copy(
                    update={
                        "activations": sv.activations.to(device=device, dtype=dtype)
                    }
                )
            )

        self._steering_data[key] = vectors

    def clear_steering_data(self, key: str) -> None:
        """Remove steering data for a completed request."""
        self._steering_data.pop(key, None)

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured activations without returning them.

        Called in the ``finally`` block of ``_patched_generate`` to clean
        up leaked state when a request is aborted or the client disconnects
        before ``get_captured_states`` is called.  On normal completion this
        is a no-op because ``get_captured_states`` already ``.pop()``-ed
        the entry.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]
                logger.debug("Cleared leaked activations for %s", req_id)

    def _build_payload(
        self, internal_req_id: str
    ) -> dict[str, dict[str, "Float[torch.Tensor, 'n_layers total_pos hidden_dim']"]]:  # type: ignore[reportUndefinedVariable]
        """Materialise the stacked-tensor payload for one internal request id.

        Pops the entry from ``_captured_states`` so successive calls do not
        re-emit the same data. Shared by :meth:`get_captured_states` and
        :meth:`get_captured_states_batch`.
        """
        layer_dict = self._captured_states.pop(internal_req_id)
        sorted_indices = sorted(layer_dict.keys())
        per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
            torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
        ]
        stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
            torch.stack(per_layer, dim=0)
        )
        if stacked.is_cuda:
            stacked = stacked.cpu()
        return {"activations": {"residual_stream": stacked}}

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle + zstd for safe ZMQ
        transport (the compression matters most when the response crosses
        the network in the OpenAI/Inspect HTTP path).

        Returns a dict when deserialized::

            {
                "activations": {
                    "residual_stream": Tensor,  # (n_layers, total_pos, d_model)
                }
            }

        Layers are stacked in ascending order along dim 0.
        Removes the request's data after retrieval.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                payload = self._build_payload(req_id)
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def get_captured_states_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Retrieve captured activations for many requests in one RPC.

        Equivalent to calling :meth:`get_captured_states` once per id, but
        emits a single payload covering every request that has data. At
        large batch sizes the per-request ``collective_rpc`` roundtrip is
        the dominant cost on the offline ``LLM.generate`` path; batching it
        cuts N round-trips to one.

        Returns ``pickle.dumps({external_req_id: payload, ...})`` where
        each ``payload`` is ``{"activations": {"residual_stream": Tensor}}``.
        Missing or unmatched ids are simply absent; ``None`` if nothing
        matched at all.

        We deliberately don't ``zstd``-compress here: this RPC only fires
        on the offline ``LLM.generate`` path, which is in-process IPC,
        not HTTP. ``get_captured_states`` keeps zstd for the OpenAI/Inspect
        path where the response crosses the network.
        """
        if not external_req_ids:
            return None
        out: dict[str, dict[str, Any]] = {}
        # Walk live state once and bucket by external id; matches the
        # ``"{external_req_id}-"`` prefix rule from get_captured_states.
        for req_id in list(self._captured_states):
            for external_req_id in external_req_ids:
                if req_id.startswith(f"{external_req_id}-"):
                    out[external_req_id] = self._build_payload(req_id)
                    break
        if not out:
            return None
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)

    def _debug_layer_discovery(self) -> list[int]:
        """Global indices of decoder layers found via the registry (testing)."""
        return sorted((_discover_layer_modules(self) or {}).keys())

    # ------------------------------------------------------------------
    # Hook data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_hook_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store hook definitions for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``Hook`` instances (using cloudpickle for the callable
        ``fn``), validates layer indices against the model, and stores them
        keyed by *key* (an external request ID or ``_hook_id`` sentinel).
        """
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        num_layers = _get_total_num_layers(self)
        for hook in hooks:
            for idx in hook.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
        self._hook_data[key] = hooks

    def get_hook_results(self, external_req_id: str) -> bytes | None:
        """Retrieve hook results (``ctx.saved`` dicts) for a request.

        Returns from ALL ranks (including PP ranks that own different
        layers).  The plugin merges results across ranks.
        Matches by ``"{external_req_id}-"`` prefix on ``_hook_contexts``.
        Returns ``{str(hook_position): ctx.saved}`` pickled, where
        ``hook_position`` indexes the per-request hook list.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                contexts = self._hook_contexts.pop(req_id)
                saved_dicts = {str(pos): ctx.saved for pos, ctx in contexts.items()}
                return pickle.dumps(saved_dicts)
        return None

    def clear_hook_data(self, key: str) -> None:
        """Remove hook definitions for a completed request."""
        self._hook_data.pop(key, None)

    def clear_hook_contexts(self, external_req_id: str) -> None:
        """Remove hook contexts for a completed or aborted request.

        Prefix-match cleanup, same pattern as ``clear_captured_states``.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                del self._hook_contexts[req_id]

    # ------------------------------------------------------------------
    # Persistent hook management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_persistent_hooks(self, pickled_data: bytes) -> None:
        """Append hooks that apply to every subsequent request.

        Accepts cloudpickle'd ``list[Hook]``.  Validates layer indices.
        Appends to existing persistent hooks (call ``clear_persistent_hooks``
        first for a clean slate).  Also ensures forward hooks are installed
        on the model layers.
        """
        self.install_hooks()
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        num_layers = _get_total_num_layers(self)
        for hook in hooks:
            for idx in hook.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
        self._persistent_hooks.extend(hooks)

    def get_all_hook_results(self) -> bytes | None:
        """Retrieve accumulated persistent hook contexts from all requests.

        Returns from ALL ranks (for PP support).  Does NOT clear — call
        ``clear_persistent_hooks`` explicitly.

        Returns pickled ``{internal_req_id: {hook_idx_str: ctx.saved}}``.
        """
        if not self._persistent_hook_contexts:
            return None
        results: dict[str, dict[str, dict[str, Any]]] = {}
        for req_id, contexts in self._persistent_hook_contexts.items():
            results[req_id] = {str(pos): ctx.saved for pos, ctx in contexts.items()}
        return pickle.dumps(results)

    def clear_persistent_hooks(self) -> None:
        """Remove persistent hooks and all accumulated contexts."""
        self._persistent_hooks = []
        self._persistent_hook_contexts = {}

    def clear_persistent_hook_results(self) -> None:
        """Drop accumulated persistent-hook contexts, keeping hooks registered.

        ``get_all_hook_results`` never drains, so results pile up across
        requests. This clears them without unregistering the hooks (so a
        fitted lens does not need re-uploading), letting a client bound
        accumulation by clearing between turns.
        """
        self._persistent_hook_contexts = {}

    # ------------------------------------------------------------------
    # Parameter prefetch (called via collective_rpc — all ranks in sync)
    # ------------------------------------------------------------------

    _prefetched_params: dict[str, torch.Tensor] = {}

    def prefetch_parameters(self, names: list[str]) -> None:
        """Pre-fetch and gather parameters across TP and PP ranks.

        Safe to call PP collectives here because ``collective_rpc``
        runs on all ranks simultaneously.  Results are stored in
        ``_prefetched_params`` for use by ``HookContext.get_parameter``.
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.model_executor.models.utils import PPMissingLayer

        model = self.model_runner.model
        tp_group = get_tp_group()
        pp_group = get_pp_group()

        for name in names:
            # Traverse to find the parameter.
            obj: Any = model
            parts = name.split(".")
            is_local = True
            for attr in parts:
                obj = getattr(obj, attr)
                if isinstance(obj, PPMissingLayer):
                    is_local = False
                    break

            param: torch.Tensor | None = None
            if is_local:
                local_t = torch.as_tensor(obj)

                # TP gather if sharded; otherwise reuse the existing tensor.
                module: Any = model
                for attr in parts[:-1]:
                    module = getattr(module, attr)
                tp_size = getattr(module, "tp_size", 1)
                if tp_size > 1:
                    gathered = [torch.empty_like(local_t) for _ in range(tp_size)]
                    dist.all_gather(gathered, local_t, group=tp_group.device_group)
                    gather_dim = getattr(module, "gather_dim", 0)
                    param = torch.cat(gathered, dim=gather_dim)
                else:
                    param = local_t  # no copy — reference to existing parameter

            # PP broadcast — safe here because all ranks are in this RPC.
            if pp_group.world_size > 1:
                has_it = torch.tensor(
                    [1 if is_local else 0], device="cuda", dtype=torch.int32
                )
                all_has = [torch.zeros_like(has_it) for _ in range(pp_group.world_size)]
                dist.all_gather(all_has, has_it, group=pp_group.device_group)
                source_pp = next(i for i, t in enumerate(all_has) if t.item() == 1)
                source_global = pp_group.ranks[source_pp]

                if param is None:
                    # Receive shape + dtype.
                    meta = torch.zeros(3, device="cuda", dtype=torch.int64)
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    ndim = int(meta[0].item())
                    dtype = _DTYPE_LIST[int(meta[1].item())]
                    shape_t = torch.zeros(ndim, device="cuda", dtype=torch.int64)
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )
                    shape = tuple(int(s) for s in shape_t.tolist())
                    param = torch.empty(shape, device="cuda", dtype=dtype)
                else:
                    meta = torch.tensor(
                        [param.ndim, _dtype_to_idx(param.dtype), 0],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    shape_t = torch.tensor(
                        list(param.shape),
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )

                dist.broadcast(param, src=source_global, group=pp_group.device_group)

            assert param is not None, f"Parameter {name!r} not found on any rank"
            self._prefetched_params[name] = param

    def clear_prefetched_params(self) -> None:
        """Remove all pre-fetched parameters."""
        self._prefetched_params = {}
