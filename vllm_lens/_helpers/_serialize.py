"""Tensor serialization helpers for vllm-lens activations and hook results."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import torch
import zstandard as zstd

if TYPE_CHECKING:
    from vllm_lens._helpers._activation_store import ActivationStore

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

# torch dtypes that numpy cannot represent natively.
_TORCH_TO_NUMPY_VIEW: dict[torch.dtype, np.dtype[Any]] = {
    torch.bfloat16: np.dtype(np.int16),
}


def _encode_tensor(tensor: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    """Compress a tensor to zstd bytes + the metadata needed to round-trip it.

    Shared by the base64 (:func:`serialize_tensor`) and binary
    (:func:`serialize_tensor_binary`) transports so both encode bit-identically.
    """
    t = tensor.detach().cpu()
    original_dtype = str(t.dtype)

    view_dtype = _TORCH_TO_NUMPY_VIEW.get(t.dtype)
    if view_dtype is not None:
        arr = t.view(torch.int16).numpy()
    else:
        arr = t.numpy()

    compressed = _ZSTD_COMPRESSOR.compress(arr.tobytes())
    meta = {
        "dtype": str(arr.dtype),
        "original_dtype": original_dtype,
        "shape": list(arr.shape),
        "compression": "zstd",
    }
    return compressed, meta


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Convert a single torch.Tensor to a JSON-serializable base64 dict.

    Preserves the tensor's native dtype (including bfloat16) and applies
    zstd compression before base64-encoding for ~4-6× size reduction
    compared to the previous float32 + uncompressed path.

    Output::

        {
            "data": "<b64-of-zstd-compressed-bytes>",
            "dtype": "int16",          # numpy storage dtype
            "original_dtype": "torch.bfloat16",  # for faithful round-trip
            "shape": [...],
            "compression": "zstd",
        }
    """
    compressed, meta = _encode_tensor(tensor)
    return {"data": base64.b64encode(compressed).decode("ascii"), **meta}


def deserialize_tensor(d: dict[str, Any]) -> torch.Tensor:
    """Convert a base64 dict back to a torch.Tensor.

    Handles both the new compressed/native-dtype format and the legacy
    uncompressed float32 format for backward compatibility.
    """
    return tensor_from_bytes(base64.b64decode(d["data"]), d)


def tensor_from_bytes(raw: bytes, meta: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from transport bytes + its metadata.

    ``raw`` is the payload exactly as produced by :func:`_encode_tensor` (a zstd
    frame when ``meta["compression"] == "zstd"``). Shared by the base64 path
    (after ``b64decode``) and the binary-handle path (after fetching the bytes
    from ``GET /v1/activations/{handle}``), so both reconstruct identically.
    """
    if meta.get("compression") == "zstd":
        raw = _ZSTD_DECOMPRESSOR.decompress(raw)

    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    t = torch.from_numpy(arr.copy())

    original_dtype = meta.get("original_dtype")
    if original_dtype == "torch.bfloat16":
        t = t.view(torch.bfloat16)
    elif original_dtype == "torch.float16":
        t = t.to(torch.float16)

    return t


def serialize_activations(tensor_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat dict of torch tensors to a JSON-serializable form.

    Input::

        {"residual_stream": Tensor(n_layers, total_pos, d)}

    Output::

        {"residual_stream": {"data": "<b64>", "dtype": "...", "shape": [...]}}
    """
    return {name: serialize_tensor(t) for name, t in tensor_dict.items()}


def serialize_tensor_binary(
    tensor: torch.Tensor, store: ActivationStore
) -> dict[str, Any]:
    """Park a tensor's zstd bytes in ``store``; return a JSON handle descriptor.

    The heavy bytes never enter the JSON body — the client fetches them from
    ``GET /v1/activations/{handle}`` and reconstructs with the metadata here.
    Shape identical to :func:`serialize_tensor` minus ``data``, plus
    ``handle`` / ``nbytes`` / ``transport``.
    """
    compressed, meta = _encode_tensor(tensor)
    handle = store.put(compressed, meta)
    return {"handle": handle, "nbytes": len(compressed), "transport": "binary", **meta}


def serialize_activations_binary(
    tensor_dict: dict[str, Any], store: ActivationStore
) -> dict[str, Any]:
    """Binary-transport counterpart of :func:`serialize_activations`.

    All tensors of one response are stored in a single atomic
    :meth:`ActivationStore.put_many`, so a later tensor's insert can never
    evict an earlier one's handle before the client has fetched it.
    """
    names = list(tensor_dict)
    encoded = [_encode_tensor(tensor_dict[n]) for n in names]
    handles = store.put_many(encoded)
    return {
        name: {
            "handle": handle,
            "nbytes": len(compressed),
            "transport": "binary",
            **meta,
        }
        for name, handle, (compressed, meta) in zip(names, handles, encoded)
    }


def decode_activations(
    response_json: dict[str, Any],
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Decode activations from an HTTP API response, base64 or binary.

    Takes the raw JSON response dict from ``/v1/completions`` or
    ``/v1/chat/completions`` and converts any activation tensors back to
    ``torch.Tensor``. Two transports are supported per entry:

    - **base64** (default): the entry carries ``data``; decoded in-process.
    - **binary** (``transport="binary"``): the entry carries a ``handle`` and no
      ``data``; ``fetch_bytes(handle)`` must be supplied to pull the raw bytes
      from ``GET /v1/activations/{handle}``. :class:`VLLMLensClient` wires this
      automatically; a bare call without ``fetch_bytes`` raises a clear error
      rather than silently dropping the tensor.

    If no ``activations`` key is present, returns an empty dict.
    """
    raw = response_json.get("activations")
    if raw is None:
        return {}

    out: dict[str, Any] = {}
    for name, encoded in raw.items():
        if "data" in encoded:
            out[name] = deserialize_tensor(encoded)
        elif "handle" in encoded:
            if fetch_bytes is None:
                raise ValueError(
                    f"activation {name!r} uses binary transport "
                    f"(handle={encoded['handle']!r}); pass fetch_bytes= or use "
                    "VLLMLensClient, which fetches it for you"
                )
            out[name] = tensor_from_bytes(fetch_bytes(encoded["handle"]), encoded)
        else:
            raise ValueError(
                f"activation {name!r} has neither 'data' nor 'handle' — "
                "unrecognised transport"
            )
    return out


_JSON_SAFE_TYPES = (str, int, float, bool, type(None))


def _serialize_value(v: Any) -> Any:
    """Serialize a single value from ``ctx.saved`` for JSON transport."""
    if isinstance(v, torch.Tensor):
        return {"__type__": "tensor", **serialize_tensor(v)}
    if isinstance(v, _JSON_SAFE_TYPES):
        return v
    if isinstance(v, (list, dict)):
        # Recurse for containers — fall through to cloudpickle if nested
        # values aren't JSON-safe.
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            pass
    return {
        "__type__": "cloudpickle",
        "data": base64.b64encode(cloudpickle.dumps(v)).decode("ascii"),
    }


def _deserialize_value(v: Any) -> Any:
    """Deserialize a single value from serialized hook results."""
    if isinstance(v, dict) and "__type__" in v:
        if v["__type__"] == "tensor":
            return deserialize_tensor(v)
        if v["__type__"] == "cloudpickle":
            return cloudpickle.loads(base64.b64decode(v["data"]))
    return v


def serialize_hook_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Serialize hook results for JSON transport.

    Input (from worker): ``{hook_index_str: ctx.saved_dict}``.
    Tensors are serialized via :func:`serialize_tensor`, JSON-safe values
    pass through, and other types are cloudpickle'd.
    """
    return {
        hook_idx: {k: _serialize_value(v) for k, v in saved.items()}
        for hook_idx, saved in results.items()
    }


def deserialize_hook_results(
    raw: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Deserialize hook results from JSON transport."""
    return {
        hook_idx: {k: _deserialize_value(v) for k, v in saved.items()}
        for hook_idx, saved in raw.items()
    }
