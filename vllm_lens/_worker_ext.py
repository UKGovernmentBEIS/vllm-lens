"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.

Also supports capturing each attention head's individual write to the
residual stream (``extra_args["output_head_contributions"]``), for
component-level circuit analysis (e.g. ``_examples/induction_circuits.py``).
vLLM's attention runs through fused kernels with no hookable module
exposing the attention weight matrix itself, so this is derived instead
from a forward *pre*-hook on each layer's ``self_attn.o_proj``: since
``o_proj`` is linear, its input decomposes exactly per-head into each
head's additive contribution to the residual stream (the "OV circuit"
write), without needing attention weights at all.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd
from vllm.distributed.parallel_state import get_pp_group, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers.types import SteeringVector

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


def _get_attn_o_proj(layer: torch.nn.Module) -> tuple[torch.nn.Module, int, int]:
    """Find a decoder layer's attention output projection, plus its (TP-local)
    head count and head dim.

    Supports the ``self_attn.o_proj`` naming used across the Llama/Qwen/
    Mistral/Gemma family — the same architecture family ``_get_layers``
    targets.  ``o_proj`` is a ``RowParallelLinear`` with
    ``input_is_parallel=True``, so under tensor parallelism its input is
    already the local shard: shape ``(seq_len, num_heads * head_dim)``
    where ``num_heads`` is this rank's local head count, not the model's
    total head count.
    """
    attn: Any = getattr(layer, "self_attn", None)
    o_proj = getattr(attn, "o_proj", None) if attn is not None else None
    num_heads = getattr(attn, "num_heads", None) if attn is not None else None
    head_dim = getattr(attn, "head_dim", None) if attn is not None else None
    if o_proj is None or num_heads is None or head_dim is None:
        raise AttributeError(
            f"Cannot find self_attn.o_proj/num_heads/head_dim on "
            f"{type(layer).__name__}. Expected the Llama/Qwen/Mistral/Gemma "
            "style attention module layout."
        )
    return o_proj, num_heads, head_dim


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


def _get_batch_bookkeeping(
    extension: HiddenStatesExtension,
) -> tuple[Any, int, list[str], torch.Tensor, Any] | None:
    """Read per-forward-pass batch bookkeeping shared by all hooks.

    Returns ``(runner, num_reqs, req_ids, query_start_loc, attn_metadata)``,
    or ``None`` if no forward context / attention metadata is available this
    step (e.g. a dummy profiling pass).  Shared between the per-layer output
    hook and the per-head ``o_proj``-input hook since both need to slice the
    same dynamically-batched forward pass by request.
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

    return runner, num_reqs, req_ids, query_start_loc, attn_metadata


def _hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """Core hook logic, separated so _make_hook can wrap it in try/except."""
    bookkeeping = _get_batch_bookkeeping(extension)
    if bookkeeping is None:
        return None
    runner, num_reqs, req_ids, query_start_loc, attn_metadata = bookkeeping

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


def _head_contrib_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    o_proj_weight: torch.Tensor,
    z: torch.Tensor,
) -> None:
    """Decompose ``o_proj``'s input into each attention head's additive
    contribution to the residual stream, and capture per requesting request.

    ``o_proj`` computes ``attn_out = z @ W_O.T`` (plus bias, not attributed to
    any head).  Since this is linear, it decomposes exactly per-head:
    ``attn_out = sum_h  z[:, h_slice] @ W_O[:, h_slice].T``.  This gives each
    head's individual write to the residual stream — the "OV circuit" output
    — without needing the (unavailable, see module docstring) attention
    weight matrix itself.  ``z`` is already this rank's local TP shard of
    heads (``o_proj`` is a ``RowParallelLinear`` with
    ``input_is_parallel=True``), so no cross-rank communication is needed
    here; the caller tags results with this rank's TP index for merging.
    """
    bookkeeping = _get_batch_bookkeeping(extension)
    if bookkeeping is None:
        return
    runner, num_reqs, req_ids, query_start_loc, _attn_metadata = bookkeeping

    # (hidden_dim, num_heads * head_dim) -> (hidden_dim, num_heads, head_dim)
    w_heads = o_proj_weight.view(o_proj_weight.shape[0], num_heads, head_dim)

    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra:
            continue

        wanted_layers = extra.get("output_head_contributions")
        if wanted_layers is None:
            continue
        if isinstance(wanted_layers, list) and layer_idx not in wanted_layers:
            continue

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        seq_len = end - start
        # (seq_len, num_heads * head_dim) -> (seq_len, num_heads, head_dim)
        z_heads = z[start:end].view(seq_len, num_heads, head_dim)
        # contributions[h, t, :] = z_heads[t, h, :] @ w_heads[:, h, :].T
        # Blocking .cpu() benchmarked faster than non_blocking + event sync
        # (see the equivalent note on the residual-stream capture above).
        contributions: Float[torch.Tensor, "num_heads seq_len hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
            torch.einsum("thd,ohd->hto", z_heads, w_heads).cpu()
        )

        if req_id not in extension._captured_head_contribs:
            extension._captured_head_contribs[req_id] = {}
        layer_states = extension._captured_head_contribs[req_id]
        if layer_idx not in layer_states:
            layer_states[layer_idx] = []
        layer_states[layer_idx].append(contributions)


def _make_head_contrib_hook(
    extension: HiddenStatesExtension,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
) -> Callable:
    """Create a forward *pre*-hook closure capturing per-head OV contributions
    for a specific layer's ``o_proj``.

    A pre-hook (not a post-hook) is used because we need ``o_proj``'s
    *input* (the concatenated per-head values) rather than its output (the
    already-summed residual-stream write).
    """

    def hook(module: torch.nn.Module, args: tuple[object, ...]) -> None:
        """Forward pre-hook: capture per-head OV contributions, if wanted.

        Never returns a value (unlike the steering hook above) — this hook
        only observes ``o_proj``'s input, it never modifies it.
        """
        try:
            module_any: Any = module
            weight: torch.Tensor = module_any.weight
            z = args[0]
            assert isinstance(z, torch.Tensor)
            _head_contrib_hook_inner(
                extension, layer_idx, num_heads, head_dim, weight, z
            )
        except Exception:
            logger.warning(
                "vllm-lens head-contribution hook error on layer %d, skipping",
                layer_idx,
                exc_info=True,
            )

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

    # Per-request captured per-head OV contributions:
    # internal_req_id → { layer_idx → [tensor (num_heads seq_len hidden_dim), ...] }
    # Unlike _captured_states, every TP rank captures here (each rank holds a
    # different, non-replicated shard of heads — see get_captured_head_contributions).
    _captured_head_contribs: dict[
        str,
        dict[int, list[Float[torch.Tensor, "num_heads seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}

    # Per-request steering configs:
    # key (external_req_id or _steering_id) → list of SteeringVector
    _steering_data: dict[str, list[SteeringVector]] = {}

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
        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}
        self._captured_head_contribs = {}
        self._steering_data = {}

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
            layer.register_forward_hook(_make_hook(self, layer_idx))
            o_proj, num_heads, head_dim = _get_attn_o_proj(layer)
            o_proj.register_forward_pre_hook(
                _make_head_contrib_hook(self, layer_idx, num_heads, head_dim)
            )

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

    def clear_captured_head_contributions(self, external_req_id: str) -> None:
        """Remove captured per-head contributions without returning them.

        Same leaked-state cleanup role as ``clear_captured_states``, for the
        per-head capture channel.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_head_contribs):
            if req_id.startswith(prefix):
                del self._captured_head_contribs[req_id]
                logger.debug("Cleared leaked head contributions for %s", req_id)

    def get_captured_head_contributions(self, external_req_id: str) -> bytes | None:
        """Retrieve captured per-head OV contributions for a specific request.

        Same prefix-matching convention as ``get_captured_states``. Unlike
        residual-stream capture, this is **not** rank-0-only: under tensor
        parallelism, each rank holds a different (non-replicated) shard of
        attention heads, so every rank's data is needed to reconstruct the
        full head count.  The response is tagged with this rank's TP/PP
        indices so the caller (``_merge_captured_head_contributions`` in
        ``_activations_plugin.py``) can group and order ranks correctly
        without assuming a specific global-rank layout convention.

        Returns a dict when deserialized::

            {
                "head_contributions": Tensor,  # (n_layers_local, n_heads_local, total_pos, hidden_dim)
                "tp_rank": int,
                "pp_rank": int,
            }

        Layers are stacked in ascending (this rank's local) order along dim
        0; heads are stacked in ascending local-head order along dim 1.
        Removes the request's data after retrieval.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_head_contribs):
            if req_id.startswith(prefix):
                layer_dict = self._captured_head_contribs.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                per_layer: list[
                    Float[torch.Tensor, "num_heads total_pos hidden_dim"]  # type: ignore[reportUndefinedVariable]
                ] = [torch.cat(layer_dict[idx], dim=1) for idx in sorted_indices]
                stacked: Float[
                    torch.Tensor, "n_layers num_heads total_pos hidden_dim"  # type: ignore[reportUndefinedVariable]
                ] = torch.stack(per_layer, dim=0)
                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps(
                        {
                            "head_contributions": stacked,
                            "tp_rank": get_tensor_model_parallel_rank(),
                            "pp_rank": get_pp_group().rank_in_group,
                        }
                    )
                )
        return None

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)
