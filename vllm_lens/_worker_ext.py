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

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the transformer decoder layers regardless of model architecture."""
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
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.
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
                v = norm_match(target[start:end], v)
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
                    v = norm_match(target[rel], v)
                target[rel] = target[rel] + v * cfg.scale


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
        else:
            modified_output = output.clone()
            target = modified_output

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
                per_req_steering[i], layer_idx, target, start, end, abs_start
            )

    # --- Phase 2.5: run generic hooks --------------------------------
    # Collect per-request hooks and persistent hooks separately so their
    # contexts are stored in different dicts (per-request contexts get
    # cleaned up after each request; persistent ones accumulate).
    per_req_hooks: list[list[Hook]] = []
    per_req_persistent: list[list[Hook]] = []
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
        # _find_hook_configs returns per-request hooks only (persistent
        # hooks are handled separately below).
        hooks = _find_hook_configs_no_persistent(extension, req_id, extra)
        per_req_hooks.append(hooks)
        per_req_persistent.append(persistent_hooks)
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

        for i in range(num_reqs):
            # Persistent hooks fire first (base layer); per-request hooks
            # see the persistent-modified state.
            all_hooks = per_req_persistent[i] + per_req_hooks[i]
            if not all_hooks:
                continue
            req_id = req_ids[i]
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            seq_len = end - start

            n_persistent = len(per_req_persistent[i])
            n_per_req = len(per_req_hooks[i])

            # Get-or-create contexts: persistent first, then per-request
            # (matching all_hooks order above).
            if n_persistent > 0:
                if req_id not in extension._persistent_hook_contexts:
                    extension._persistent_hook_contexts[req_id] = [
                        HookContext() for _ in range(n_persistent)
                    ]
                ps_ctxs = extension._persistent_hook_contexts[req_id]
            else:
                ps_ctxs = []

            if n_per_req > 0:
                if req_id not in extension._hook_contexts:
                    extension._hook_contexts[req_id] = [
                        HookContext() for _ in range(n_per_req)
                    ]
                pr_ctxs = extension._hook_contexts[req_id]
            else:
                pr_ctxs = []

            contexts = ps_ctxs + pr_ctxs

            for hi, hook in enumerate(all_hooks):
                if hook.pre or layer_idx not in hook.layer_indices:
                    continue
                ctx = contexts[hi]
                ctx.layer_idx = layer_idx
                ctx.seq_len = seq_len
                ctx.model = runner.model

                result = hook.fn(ctx, hook_hidden[start:end])

                if result is not None:
                    # Apply delta so Phase 3 captures the modified state.
                    delta = result - hook_hidden[start:end]
                    if modified_output is None:
                        if isinstance(output, tuple):
                            modified_output = (output[0].clone(), output[1])
                        else:
                            modified_output = output.clone()
                    if isinstance(modified_output, tuple):
                        modified_output[0][start:end] = (
                            modified_output[0][start:end] + delta
                        )
                    else:
                        modified_output[start:end] = modified_output[start:end] + delta
                    # Update hook_hidden so subsequent hooks see the change.
                    hook_hidden[start:end] = result

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
            # Blocking .cpu() benchmarked faster than non_blocking + event sync
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

    # Collect pre-hooks from per-request and persistent sources.
    persistent_hooks = extension._persistent_hooks
    modified = False
    working = input_tensor

    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        all_hooks = [h for h in persistent_hooks if h.pre]
        per_req = _find_hook_configs_no_persistent(extension, req_id, extra)
        all_hooks.extend(h for h in per_req if h.pre)
        if not all_hooks:
            continue

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        seq_len = end - start

        # Get-or-create contexts under the same req_id as post-hooks
        # (no prefix) so get_hook_results / clear_hook_contexts can find them.
        n_persistent_pre = sum(1 for h in persistent_hooks if h.pre)
        n_per_req_pre = sum(1 for h in per_req if h.pre)

        if n_persistent_pre > 0:
            if req_id not in extension._persistent_hook_contexts:
                extension._persistent_hook_contexts[req_id] = [
                    HookContext() for _ in range(n_persistent_pre)
                ]
            ps_ctxs = extension._persistent_hook_contexts[req_id]
        else:
            ps_ctxs = []

        if n_per_req_pre > 0:
            if req_id not in extension._hook_contexts:
                extension._hook_contexts[req_id] = [
                    HookContext() for _ in range(n_per_req_pre)
                ]
            pr_ctxs = extension._hook_contexts[req_id]
        else:
            pr_ctxs = []

        contexts = ps_ctxs + pr_ctxs

        for hi, hook in enumerate(all_hooks):
            if layer_idx not in hook.layer_indices:
                continue
            hctx = contexts[hi]
            hctx.layer_idx = layer_idx
            hctx.seq_len = seq_len
            hctx.model = runner.model

            result = hook.fn(hctx, working[start:end])
            if result is not None:
                if not modified:
                    working = input_tensor.clone()
                    modified = True
                working[start:end] = result

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

    # Per-request hook contexts (one HookContext per hook per internal request):
    # internal_req_id → list[HookContext]
    _hook_contexts: dict[str, list[HookContext]] = {}

    # Persistent hook contexts (separate from per-request to avoid cleanup conflicts):
    # internal_req_id → list[HookContext]
    _persistent_hook_contexts: dict[str, list[HookContext]] = {}

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
        layers = _get_layers(self.model_runner.model)
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
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

        num_layers = len(_get_layers(self.model_runner.model))
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

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle for safe ZMQ
        transport.

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
                layer_dict = self._captured_states.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
                    torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
                ]
                stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
                    torch.stack(per_layer, dim=0)
                )
                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps(
                        {
                            "activations": {"residual_stream": stacked},
                        }
                    )
                )
        return None

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)

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
        num_layers = len(_get_layers(self.model_runner.model))
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
        Returns ``{str(hook_index): ctx.saved}`` pickled.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                contexts = self._hook_contexts.pop(req_id)
                saved_dicts = {str(i): ctx.saved for i, ctx in enumerate(contexts)}
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
        num_layers = len(_get_layers(self.model_runner.model))
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
            results[req_id] = {str(i): ctx.saved for i, ctx in enumerate(contexts)}
        return pickle.dumps(results)

    def clear_persistent_hooks(self) -> None:
        """Remove persistent hooks and all accumulated contexts."""
        self._persistent_hooks = []
        self._persistent_hook_contexts = {}
