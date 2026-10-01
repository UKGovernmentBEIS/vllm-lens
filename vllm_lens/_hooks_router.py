"""FastAPI router for persistent hook management (Garçon-style).

Separate module so that ``from __future__ import annotations`` in
``_activations_plugin.py`` does not interfere with FastAPI's
annotation-based dependency injection.
"""

import json
import pickle
from collections.abc import Iterator
from typing import Any

import cloudpickle
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_lens._helpers._serialize import serialize_hook_results
from vllm_lens._helpers.types import Hook

router = APIRouter(prefix="/v1/hooks", tags=["vllm-lens"])

# Binary activation transport (issue #31). Separate prefix so it can be included
# independently of the hook-management router.
activations_router = APIRouter(prefix="/v1/activations", tags=["vllm-lens"])


_ACT_CHUNK_BYTES = 1 << 20  # 1 MiB


@activations_router.get("/{handle}")
async def get_activation(handle: str) -> StreamingResponse:
    """Stream a binary activation payload parked by an ``activations_transport="binary"`` capture.

    Returns the raw ``zstd`` tensor bytes as ``application/octet-stream``; the
    dtype/shape metadata needed to reconstruct the tensor rides in
    ``X-VLLM-Lens-*`` headers (and is also echoed in the completion JSON that
    handed out the handle, so a client already has it). An unknown or expired
    handle is a flat 404 — indistinguishable from a never-existed one, so it
    leaks nothing. ``handle`` is only ever a dict key here (no filesystem, no
    ``collective_rpc``), so there is no traversal or injection surface.

    Auth: this router is included into vLLM's OpenAI app, whose
    ``AuthenticationMiddleware`` guards every ``/v1`` path — so this endpoint
    requires the same API key as ``/v1/completions``. The handle is an
    unguessable (192-bit) bearer capability scoped to a single capture; with a
    single shared server API key it is the per-request boundary. ``no-store``
    keeps intermediaries from caching model internals; per-principal
    ownership-binding for multi-tenant keying is left as a follow-up (issue #31).

    The payload is streamed in chunks rather than copied whole into a second
    response buffer, so serving a multi-GB capture doesn't double peak memory.
    """
    from vllm_lens._helpers._activation_store import store

    result = store.get(handle)
    if result is None:
        raise HTTPException(404, "Unknown or expired activation handle")
    payload, meta = result
    headers = {
        "Content-Length": str(len(payload)),
        "Cache-Control": "no-store, private",
        "X-Content-Type-Options": "nosniff",
        "X-VLLM-Lens-Dtype": str(meta.get("dtype", "")),
        "X-VLLM-Lens-Original-Dtype": str(meta.get("original_dtype", "")),
        "X-VLLM-Lens-Shape": json.dumps(meta.get("shape", [])),
        "X-VLLM-Lens-Compression": str(meta.get("compression", "")),
    }

    def _chunks() -> "Iterator[bytes]":
        for i in range(0, len(payload), _ACT_CHUNK_BYTES):
            yield payload[i : i + _ACT_CHUNK_BYTES]

    return StreamingResponse(
        _chunks(), media_type="application/octet-stream", headers=headers
    )


def _engine_client(request: Request):
    return request.app.state.engine_client


@router.post("/register")
async def register_hooks(raw_request: Request):
    body = await raw_request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    hooks_raw = body.get("hooks")
    if hooks_raw is None:
        raise HTTPException(400, "Missing 'hooks' in request body")
    if not isinstance(hooks_raw, list) or any(
        not isinstance(h, dict) for h in hooks_raw
    ):
        raise HTTPException(400, "'hooks' must be a list of objects")
    hooks = [Hook.model_validate(h) for h in hooks_raw]
    payload = cloudpickle.dumps(hooks)
    engine = _engine_client(raw_request)
    await engine.collective_rpc("set_persistent_hooks", args=(payload,))
    engine._has_persistent_hooks = True
    prefetch = body.get("prefetch_params")
    if prefetch:
        await engine.collective_rpc("prefetch_parameters", args=(prefetch,))
    return JSONResponse({"status": "ok", "count": len(hooks)})


@router.post("/collect")
async def collect_hook_results(raw_request: Request):
    raw_list = await _engine_client(raw_request).collective_rpc("get_all_hook_results")
    # Merge across PP ranks: each rank returns {req_id: {hook_idx: saved}}.
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in raw_list or ():
        if raw is None:
            continue
        rank_data: dict[str, dict[str, dict[str, Any]]] = pickle.loads(raw)
        for req_id, hook_data in rank_data.items():
            if req_id not in merged:
                merged[req_id] = {}
            for hook_idx, saved in hook_data.items():
                if hook_idx not in merged[req_id]:
                    merged[req_id][hook_idx] = {}
                for key, val in saved.items():
                    existing = merged[req_id][hook_idx]
                    if (
                        key in existing
                        and isinstance(existing[key], list)
                        and isinstance(val, list)
                    ):
                        existing[key].extend(val)
                    else:
                        existing[key] = val
    serialized = {
        req_id: serialize_hook_results(hook_data)
        for req_id, hook_data in merged.items()
    }
    return JSONResponse({"results": serialized})


@router.post("/clear")
async def clear_hooks(raw_request: Request):
    engine = _engine_client(raw_request)
    await engine.collective_rpc("clear_persistent_hooks")
    engine._has_persistent_hooks = False
    return JSONResponse({"status": "ok"})


@router.post("/clear_results")
async def clear_hook_results(raw_request: Request):
    """Drop accumulated results but keep the hooks registered.

    ``/collect`` never drains, so results grow unboundedly across requests.
    Call this to reset accumulation (e.g. between chat turns) without
    re-uploading the hooks.
    """
    await _engine_client(raw_request).collective_rpc("clear_persistent_hook_results")
    return JSONResponse({"status": "ok"})


@router.post("/prefetch")
async def prefetch_params(raw_request: Request):
    body = await raw_request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    names = body.get("params")
    if names is None:
        raise HTTPException(400, "Missing 'params' in request body")
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise HTTPException(400, "'params' must be a list of strings")
    await _engine_client(raw_request).collective_rpc(
        "prefetch_parameters", args=(names,)
    )
    return JSONResponse({"status": "ok", "params": names})


@router.post("/clear_prefetched")
async def clear_prefetched(raw_request: Request):
    await _engine_client(raw_request).collective_rpc("clear_prefetched_params")
    return JSONResponse({"status": "ok"})


@router.get("/ui/{filename:path}")
async def serve_ui(filename: str):
    """Serve static HTML files (e.g. emotion_tracker.html) from CWD."""
    import os

    from fastapi.responses import FileResponse

    root = os.path.realpath(os.getcwd())
    path = os.path.realpath(os.path.join(root, filename))
    # Confine to CWD — reject path-traversal escapes (e.g. "../../etc/passwd").
    if os.path.commonpath([root, path]) != root:
        raise HTTPException(403, "Path outside serving directory")
    if not os.path.isfile(path):
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(path)
