"""FastAPI router for persistent hook management (Garçon-style).

Separate module so that ``from __future__ import annotations`` in
``_activations_plugin.py`` does not interfere with FastAPI's
annotation-based dependency injection.
"""

import pickle
from typing import Any

import cloudpickle
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm_lens._helpers._serialize import serialize_hook_results
from vllm_lens._helpers.types import Hook

router = APIRouter(prefix="/v1/hooks", tags=["vllm-lens"])


def _engine_client(request: Request):
    return request.app.state.engine_client


@router.post("/register")
async def register_hooks(raw_request: Request):
    body = await raw_request.json()
    hooks_raw = body.get("hooks")
    if hooks_raw is None:
        raise HTTPException(400, "Missing 'hooks' in request body")
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
                merged[req_id][hook_idx].update(saved)
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


@router.post("/prefetch")
async def prefetch_params(raw_request: Request):
    body = await raw_request.json()
    names = body.get("params")
    if names is None:
        raise HTTPException(400, "Missing 'params' in request body")
    await _engine_client(raw_request).collective_rpc(
        "prefetch_parameters", args=(names,)
    )
    return JSONResponse({"status": "ok", "params": names})


@router.post("/clear_prefetched")
async def clear_prefetched(raw_request: Request):
    await _engine_client(raw_request).collective_rpc("clear_prefetched_params")
    return JSONResponse({"status": "ok"})
