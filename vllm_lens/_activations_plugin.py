"""
vLLM general plugin that transparently captures residual-stream
activations via worker extension when ``output_residual_stream`` is
passed in ``extra_args``.

Installed automatically via the ``vllm.general_plugins`` entry point
(configured in pyproject.toml). Patches ``EngineArgs.create_engine_config``
to inject the worker extension and eager mode, and patches
``AsyncLLM.generate`` and ``LLM.generate`` to retrieve per-request
activations for both online (async) and offline (sync) usage.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch
import zstandard as zstd

from vllm_lens._helpers._serialize import (
    serialize_activations,
    serialize_hook_results,
)
from vllm_lens._helpers.types import Hook, SteeringVector

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

_WORKER_EXT = "vllm_lens._worker_ext.HiddenStatesExtension"

# Populated by register() with the original unpatched methods.
_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None
_original_chat_stream_generator: Callable | None = None
_original_register_routers: Callable | None = None


# ---------------------------------------------------------------------------
# PP merge helper
# ---------------------------------------------------------------------------


def _merge_hook_results(
    raw_list: list[bytes | None] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge hook results from multiple PP ranks.

    Each rank returns ``{hook_idx_str: ctx.saved}``.  For list values
    (e.g. accumulated activations per layer), concatenates rather than
    overwrites so data from all PP stages is preserved.
    """
    if not raw_list:
        return None
    merged: dict[str, dict[str, Any]] = {}
    for raw in raw_list:
        if raw is None:
            continue
        rank_results: dict[str, dict[str, Any]] = pickle.loads(raw)
        for hook_idx, saved in rank_results.items():
            if hook_idx not in merged:
                merged[hook_idx] = {}
            for key, val in saved.items():
                if (
                    key in merged[hook_idx]
                    and isinstance(merged[hook_idx][key], list)
                    and isinstance(val, list)
                ):
                    merged[hook_idx][key].extend(val)
                else:
                    merged[hook_idx][key] = val
    return merged or None


def _merge_captured_states(
    states: list[bytes | None] | None,
) -> dict[str, Any] | None:
    """Merge activation captures from multiple PP ranks.

    ``collective_rpc`` returns results in rank order (rank 0, 1, ...).
    With TP, only TP-rank-0 workers capture (others return ``None``).
    Each capturing rank's tensor is sorted by global layer index.
    Because lower PP ranks hold earlier layers, concatenating non-None
    results along dim 0 produces correct global layer ordering.
    """
    if not states:
        return None
    parts: list[dict[str, Any]] = [
        pickle.loads(_ZSTD_DECOMPRESSOR.decompress(s) if s[:4] == _ZSTD_MAGIC else s)
        for s in states
        if s is not None
    ]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]["activations"]
    merged = torch.cat([p["activations"]["residual_stream"] for p in parts], dim=0)
    return {"residual_stream": merged}


def _trim_activations(
    activations: dict[str, Any],
    expected_len: int,
) -> None:
    """Trim residual stream activations and input_ids to the expected length.

    The vLLM v1 scheduler may execute one extra forward pass after the EOS
    stop condition is hit, because ``schedule()`` commits the next step
    before ``update_from_output()`` checks stop conditions.  vLLM itself
    discards the extra output tokens
    (``vllm.v1.core.sched.scheduler.Scheduler.update_from_output`` skips
    already-finished requests), but our activation capture hooks still fire
    during that extra pass.  This trims the surplus positions so the
    residual stream shape is always deterministic.
    """
    rs = activations.get("residual_stream")
    if rs is not None and rs.shape[1] > expected_len:
        activations["residual_stream"] = rs[:, :expected_len, :]
    ids = activations.get("input_ids")
    if ids is not None and len(ids) > expected_len:
        activations["input_ids"] = ids[:expected_len]


# ---------------------------------------------------------------------------
# Engine config patch — inject worker extension + eager mode
# ---------------------------------------------------------------------------


def _patched_create_engine_config(self, *args, **kwargs):
    """Patch for ``EngineArgs.create_engine_config``.

    Injects our worker extension and forces eager mode *before* the
    ``VllmConfig`` is built, so the settings propagate through any
    engine creation path (``AsyncLLM.from_engine_args``,
    ``AsyncLLM.from_vllm_config``, ``vllm serve``, etc.) including
    across subprocess boundaries.
    """
    if not self.worker_extension_cls:
        self.worker_extension_cls = _WORKER_EXT
    self.enforce_eager = True

    assert _original_create_engine_config is not None
    return _original_create_engine_config(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Generate patch — install hooks and attach activations to output
# ---------------------------------------------------------------------------


async def _patched_generate(
    self: AsyncLLM,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    """Wrap generate to install hooks, apply steering, and attach activations.

    On the first call that requests activations or steering, sends a
    one-time RPC to install forward hooks on every decoder layer.

    If ``apply_steering_vectors`` is present in ``extra_args``, the
    steering data is sent to workers via RPC *before* generation starts
    (tensors can't survive msgspec serialization in extra_args).

    When generation finishes, retrieves the captured activations from
    the worker and attaches them as ``output.activations``.
    """
    # In vLLM v1, the chat completion endpoint creates an
    # EngineCoreRequest with a *cloned* SamplingParams before calling
    # generate(). add_request() uses the clone from the
    # EngineCoreRequest, ignoring the separately-passed sampling_params.
    # We must read/modify the clone so our changes take effect.
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest

        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:  # type: ignore[reportAttributeAccessIssue]
            effective_params = prompt.sampling_params  # type: ignore[reportAttributeAccessIssue]
    except ImportError:
        pass

    extra = effective_params.extra_args or {}
    wants_activations = extra.get("output_residual_stream") is not None
    # Extract steering data and remove from extra_args before vLLM
    # serialises the SamplingParams (tensors don't survive msgspec).
    steering_vectors = extra.pop("apply_steering_vectors", None)
    # When arriving via the OpenAI API (vllm_xargs), complex values
    # are JSON-encoded strings; decode and validate as SteeringVector.
    if isinstance(steering_vectors, str):
        steering_vectors = [
            SteeringVector.model_validate(d) for d in json.loads(steering_vectors)
        ]

    # Extract hooks (callables can't survive msgspec).
    hooks_list = extra.pop("apply_hooks", None)
    if isinstance(hooks_list, str):
        hooks_list = [Hook.model_validate(d) for d in json.loads(hooks_list)]

    # Allow explicit prefix-cache bypass via extra_args.
    skip_kv_cache = extra.pop("skip_reading_prefix_cache", None)

    has_persistent = getattr(self, "_has_persistent_hooks", False)
    needs_hooks = (
        wants_activations
        or steering_vectors is not None
        or hooks_list is not None
        or has_persistent
    )
    if needs_hooks or skip_kv_cache:
        # Hooks rely on forward passes firing; prefix-cached tokens skip
        # computation entirely, so force a fresh prefill for this request.
        effective_params.skip_reading_prefix_cache = True
    if needs_hooks and not getattr(self, "_hooks_installed", False):
        await self.collective_rpc("install_hooks")
        setattr(self, "_hooks_installed", True)

    # Send steering data to workers before the forward pass begins.
    if steering_vectors is not None:
        await self.collective_rpc(
            "set_steering_data",
            args=(request_id, pickle.dumps(steering_vectors)),
        )

    # Send hook data to workers before the forward pass begins.
    if hooks_list is not None:
        await self.collective_rpc(
            "set_hook_data",
            args=(request_id, cloudpickle.dumps(hooks_list)),
        )

    assert _original_generate is not None
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if output.finished:
                if wants_activations:
                    states = await self.collective_rpc(
                        "get_captured_states", args=(request_id,)
                    )
                    activations = _merge_captured_states(states)
                    if activations is not None:
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        _trim_activations(activations, n_prompt + n_gen - 1)
                        output.activations = activations
                if hooks_list is not None:
                    raw_results = await self.collective_rpc(
                        "get_hook_results", args=(request_id,)
                    )
                    merged = _merge_hook_results(raw_results)
                    if merged is not None:
                        output.hook_results = merged
            yield output
    finally:
        if steering_vectors is not None:
            await self.collective_rpc("clear_steering_data", args=(request_id,))
        if wants_activations:
            await self.collective_rpc("clear_captured_states", args=(request_id,))
        if hooks_list is not None:
            await self.collective_rpc("clear_hook_data", args=(request_id,))
            await self.collective_rpc("clear_hook_contexts", args=(request_id,))


# ---------------------------------------------------------------------------
# Offline (sync) LLM.generate patch
# ---------------------------------------------------------------------------


def _patched_llm_generate(
    self: LLM,
    prompts: Any,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    **kwargs,
) -> list:
    """Wrap ``LLM.generate`` to install hooks, apply steering, and attach activations.

    Same logic as the async variant but for the synchronous offline API.
    Because ``LLM.generate`` auto-assigns request IDs internally, steering
    data is keyed by a synthetic ``_steering_id`` stored in ``extra_args``
    (a lightweight string that survives msgspec serialization).
    """
    if isinstance(sampling_params, Sequence):
        params_list = list(sampling_params)
    elif sampling_params is not None:
        params_list = [sampling_params]
    else:
        params_list = []

    wants_activations = any(
        (sp.extra_args or {}).get("output_residual_stream") is not None
        for sp in params_list
    )

    # Extract steering vectors per-request.  We must pop them from
    # extra_args before vLLM serialises SamplingParams (tensors don't
    # survive msgspec), but keep them for the RPC call.
    steering_payloads: dict[str, bytes] = {}  # steering_id -> pickled vectors
    for idx, sp in enumerate(params_list):
        extra = sp.extra_args or {}
        vectors = extra.pop("apply_steering_vectors", None)
        if vectors is not None:
            steering_id = f"_steer_{idx}"
            steering_payloads[steering_id] = pickle.dumps(vectors)
            if sp.extra_args is None:
                sp.extra_args = {}
            sp.extra_args["_steering_id"] = steering_id

    # Extract hooks per-request (same pattern as steering).
    hook_payloads: dict[str, bytes] = {}  # hook_id -> cloudpickled hooks
    for idx, sp in enumerate(params_list):
        extra = sp.extra_args or {}
        hooks = extra.pop("apply_hooks", None)
        if hooks is not None:
            hook_id = f"_hook_{idx}"
            hook_payloads[hook_id] = cloudpickle.dumps(hooks)
            if sp.extra_args is None:
                sp.extra_args = {}
            sp.extra_args["_hook_id"] = hook_id

    # Pop skip_reading_prefix_cache from extra_args for each request.
    any_skip_kv_cache = False
    for sp in params_list:
        if (sp.extra_args or {}).pop("skip_reading_prefix_cache", None):
            any_skip_kv_cache = True

    has_steering = len(steering_payloads) > 0
    has_hooks = len(hook_payloads) > 0
    has_persistent = getattr(self, "_has_persistent_hooks", False)
    needs_hooks = wants_activations or has_steering or has_hooks or has_persistent
    if needs_hooks or any_skip_kv_cache:
        for sp in params_list:
            sp.skip_reading_prefix_cache = True

    if needs_hooks and not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]

    # Send steering data to workers before generation.
    for sid, payload in steering_payloads.items():
        self.collective_rpc("set_steering_data", args=(sid, payload))

    # Send hook data to workers before generation.
    for hid, payload in hook_payloads.items():
        self.collective_rpc("set_hook_data", args=(hid, payload))

    assert _original_llm_generate is not None
    outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)

    if wants_activations:
        for output in outputs:
            req_id = output.request_id
            states = self.collective_rpc("get_captured_states", args=(req_id,))
            activations = _merge_captured_states(states)
            if activations is not None:
                n_prompt = len(output.prompt_token_ids)
                n_gen = len(output.outputs[0].token_ids)
                _trim_activations(activations, n_prompt + n_gen - 1)
                output.activations = activations

    if has_hooks:
        for output in outputs:
            req_id = output.request_id
            raw_results = self.collective_rpc("get_hook_results", args=(req_id,))
            merged = _merge_hook_results(raw_results)
            if merged is not None:
                output.hook_results = merged

    # Clean up steering data.
    for sid in steering_payloads:
        self.collective_rpc("clear_steering_data", args=(sid,))

    # Clean up hook data and contexts.
    for hid in hook_payloads:
        self.collective_rpc("clear_hook_data", args=(hid,))
    if has_hooks:
        for output in outputs:
            self.collective_rpc("clear_hook_contexts", args=(output.request_id,))

    return outputs


# ---------------------------------------------------------------------------
# Response builder patches for vllm serve (OpenAI-compatible API)
# ---------------------------------------------------------------------------


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    """Wrap the completion response builder to inject serialized activations and hook results."""
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        activations = getattr(res, "activations", None)
        if activations is not None:
            response.activations = serialize_activations(activations)
        hook_results = getattr(res, "hook_results", None)
        if hook_results is not None:
            response.hook_results = serialize_hook_results(hook_results)
        if activations is not None or hook_results is not None:
            break
    return response


async def _patched_chat_full_generator(
    self, request, result_generator, *args, **kwargs
):
    """Wrap the chat completion full generator to inject serialized activations.

    The original method iterates ``result_generator`` internally, so we
    wrap it with a capturing async generator to grab the final
    ``RequestOutput`` (which has ``.activations`` attached by
    ``_patched_generate``).
    """
    assert _original_chat_full_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    response = await _original_chat_full_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    )

    # Only inject for successful responses (not ErrorResponse).
    if last_output is not None and hasattr(response, "model_dump"):
        activations = getattr(last_output, "activations", None)
        if activations is not None:
            response.activations = serialize_activations(activations)
        hook_results = getattr(last_output, "hook_results", None)
        if hook_results is not None:
            response.hook_results = serialize_hook_results(hook_results)

    return response


async def _patched_chat_stream_generator(
    self, request, result_generator, *args, **kwargs
):
    """Wrap the streaming chat generator to append hook results as a final SSE chunk.

    Captures the final ``RequestOutput`` (which has ``.hook_results``
    attached by ``_patched_generate``), then yields an extra
    ``data: {...}`` chunk with serialized hook results before ``[DONE]``.
    """
    assert _original_chat_stream_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    # Buffer the last chunk so we can inject hook results before [DONE].
    async for chunk in _original_chat_stream_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    ):
        if chunk.strip() == "data: [DONE]" and last_output is not None:
            import json as _json

            extra: dict[str, Any] = {}
            activations = getattr(last_output, "activations", None)
            if activations is not None:
                extra["activations"] = serialize_activations(activations)
            hook_results = getattr(last_output, "hook_results", None)
            if hook_results is not None:
                extra["hook_results"] = serialize_hook_results(hook_results)
            if extra:
                yield f"data: {_json.dumps(extra)}\n\n"
        yield chunk


# ---------------------------------------------------------------------------
# Persistent hooks — HTTP router
# ---------------------------------------------------------------------------


def _patched_register_routers(app):
    """Inject vllm-lens hook router after vLLM registers its own routes."""
    assert _original_register_routers is not None
    _original_register_routers(app)
    from vllm_lens._hooks_router import router as hooks_router

    app.include_router(hooks_router)


# ---------------------------------------------------------------------------
# Persistent hooks — offline LLM methods
# ---------------------------------------------------------------------------


def _llm_register_hooks(
    self: LLM,
    hooks: list,
    prefetch_params: list[str] | None = None,
) -> None:
    """Register persistent hooks that apply to every subsequent request."""
    if not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    self.collective_rpc("set_persistent_hooks", args=(cloudpickle.dumps(hooks),))
    if prefetch_params:
        self.collective_rpc("prefetch_parameters", args=(prefetch_params,))
    self._has_persistent_hooks = True  # type: ignore[reportAttributeAccessIssue]


def _llm_collect_hook_results(self: LLM) -> dict:
    """Collect all accumulated hook results from the worker."""
    raw_list = self.collective_rpc("get_all_hook_results")
    for raw in raw_list or ():
        if raw is not None:
            return pickle.loads(raw)
    return {}


def _llm_clear_hooks(self: LLM) -> None:
    """Remove persistent hooks and all accumulated contexts."""
    self.collective_rpc("clear_persistent_hooks")
    self._has_persistent_hooks = False  # type: ignore[reportAttributeAccessIssue]


def _llm_prefetch_params(self: LLM, names: list[str]) -> None:
    """Pre-fetch model parameters across all TP/PP ranks."""
    if not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    self.collective_rpc("prefetch_parameters", args=(names,))


def _llm_clear_prefetched(self: LLM) -> None:
    """Remove all pre-fetched parameters."""
    self.collective_rpc("clear_prefetched_params")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup.

    Patches ``EngineArgs.create_engine_config`` to inject the worker
    extension and eager mode, ``AsyncLLM.generate`` and ``LLM.generate``
    to retrieve per-request activations for both online and offline
    usage.  Also patches the OpenAI-compatible response builders so
    activations are included in HTTP responses from ``vllm serve``.

    Use ``extra_args={"output_residual_stream": True | list[int]}`` in
    SamplingParams to request activations.
    """
    global _original_create_engine_config
    global _original_generate, _original_llm_generate
    global _original_completion_response, _original_chat_full_generator
    global _original_register_routers

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate  # type: ignore[reportAttributeAccessIssue]

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    # Add persistent hook methods to LLM.
    LLM.register_hooks = _llm_register_hooks  # type: ignore[reportAttributeAccessIssue]
    LLM.collect_hook_results = _llm_collect_hook_results  # type: ignore[reportAttributeAccessIssue]
    LLM.clear_hooks = _llm_clear_hooks  # type: ignore[reportAttributeAccessIssue]
    LLM.prefetch_params = _llm_prefetch_params  # type: ignore[reportAttributeAccessIssue]
    LLM.clear_prefetched = _llm_clear_prefetched  # type: ignore[reportAttributeAccessIssue]

    # Patch OpenAI-compatible response builders so activations survive
    # HTTP serialization.  Wrapped in try/except because these modules
    # are only available when running as an API server.
    try:
        from vllm.entrypoints.openai.completion.serving import (
            OpenAIServingCompletion,
        )

        _original_completion_response = (
            OpenAIServingCompletion.request_output_to_completion_response
        )
        OpenAIServingCompletion.request_output_to_completion_response = (
            _patched_completion_response
        )
    except Exception:
        pass

    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )

        _original_chat_full_generator = OpenAIServingChat.chat_completion_full_generator
        OpenAIServingChat.chat_completion_full_generator = _patched_chat_full_generator
        _original_chat_stream_generator = (
            OpenAIServingChat.chat_completion_stream_generator
        )
        OpenAIServingChat.chat_completion_stream_generator = (
            _patched_chat_stream_generator
        )
    except Exception:
        pass

    # Patch the serve router registration to inject our hooks API.
    try:
        import vllm.entrypoints.serve as _serve_mod

        _original_register_routers = _serve_mod.register_vllm_serve_api_routers
        _serve_mod.register_vllm_serve_api_routers = _patched_register_routers
    except Exception:
        pass
