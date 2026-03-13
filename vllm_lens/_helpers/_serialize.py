"""Tensor serialization helpers for vllm-lens activations."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import torch
import zstandard as zstd

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

# torch dtypes that numpy cannot represent natively.
_TORCH_TO_NUMPY_VIEW: dict[torch.dtype, np.dtype[Any]] = {
    torch.bfloat16: np.dtype(np.int16),
}


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
    t = tensor.detach().cpu()
    original_dtype = str(t.dtype)

    view_dtype = _TORCH_TO_NUMPY_VIEW.get(t.dtype)
    if view_dtype is not None:
        arr = t.view(torch.int16).numpy()
    else:
        arr = t.numpy()

    raw = arr.tobytes()
    compressed = _ZSTD_COMPRESSOR.compress(raw)

    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": str(arr.dtype),
        "original_dtype": original_dtype,
        "shape": list(arr.shape),
        "compression": "zstd",
    }


def deserialize_tensor(d: dict[str, Any]) -> torch.Tensor:
    """Convert a base64 dict back to a torch.Tensor.

    Handles both the new compressed/native-dtype format and the legacy
    uncompressed float32 format for backward compatibility.
    """
    raw = base64.b64decode(d["data"])

    if d.get("compression") == "zstd":
        raw = _ZSTD_DECOMPRESSOR.decompress(raw)

    arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"])
    t = torch.from_numpy(arr.copy())

    original_dtype = d.get("original_dtype")
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


def decode_activations(response_json: dict[str, Any]) -> dict[str, Any]:
    """Decode base64-encoded activations from an HTTP API response.

    Takes the raw JSON response dict from ``/v1/completions`` or
    ``/v1/chat/completions`` and converts any base64 activation tensors
    back to ``torch.Tensor``.

    Returns a dict like::

        {"residual_stream": Tensor(n_layers, total_pos, d)}

    If no ``activations`` key is present, returns an empty dict.
    """
    raw = response_json.get("activations")
    if raw is None:
        return {}
    return {name: deserialize_tensor(encoded) for name, encoded in raw.items()}
