"""Round-trip tests for tensor serialization helpers."""

from __future__ import annotations

import base64

import pytest
import torch

from vllm_lens._helpers._serialize import (
    decode_activations,
    deserialize_tensor,
    serialize_activations,
    serialize_tensor,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_trip(dtype: torch.dtype) -> None:
    """serialize_tensor → deserialize_tensor preserves values for all dtypes."""
    original = torch.randn(4, 8, dtype=dtype)
    serialized = serialize_tensor(original)
    recovered = deserialize_tensor(serialized)

    assert recovered.dtype == dtype
    assert recovered.shape == original.shape
    assert torch.equal(recovered, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_trip_activations(dtype: torch.dtype) -> None:
    """serialize_activations → decode_activations preserves values."""
    tensors = {"residual_stream": torch.randn(2, 4, 8, dtype=dtype)}
    serialized = serialize_activations(tensors)
    response_json = {"activations": serialized}
    recovered = decode_activations(response_json)

    assert "residual_stream" in recovered
    assert recovered["residual_stream"].dtype == dtype
    assert torch.equal(recovered["residual_stream"], tensors["residual_stream"])


def test_zstd_compression_present() -> None:
    """Serialized output includes compression metadata."""
    t = torch.randn(4, 8)
    serialized = serialize_tensor(t)
    assert serialized["compression"] == "zstd"
    assert "original_dtype" in serialized


def test_zstd_reduces_size() -> None:
    """Compressed payload is smaller than uncompressed for typical tensors."""
    t = torch.zeros(32, 128, 4096)  # highly compressible
    serialized = serialize_tensor(t)
    compressed_size = len(base64.b64decode(serialized["data"]))
    raw_size = t.nelement() * t.element_size()
    assert compressed_size < raw_size / 2


def test_backward_compat_legacy_format() -> None:
    """deserialize_tensor handles old format (no compression, no original_dtype)."""
    original = torch.randn(4, 8)
    arr = original.numpy()
    legacy = {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }
    recovered = deserialize_tensor(legacy)
    assert recovered.dtype == torch.float32
    assert torch.equal(recovered, original)


def test_decode_activations_empty() -> None:
    """decode_activations returns empty dict when no activations key."""
    assert decode_activations({}) == {}
    assert decode_activations({"other": "data"}) == {}


def test_bfloat16_cpu_operations() -> None:
    """bfloat16 tensors returned by deserialization support CPU operations."""
    original = torch.randn(4, 8, dtype=torch.bfloat16)
    recovered = deserialize_tensor(serialize_tensor(original))

    result = recovered.float()
    assert result.dtype == torch.float32
    assert torch.allclose(result, original.float())

    dot = (recovered @ recovered.T).float()
    expected = (original @ original.T).float()
    assert torch.allclose(dot, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_original_dtype_field(dtype: torch.dtype) -> None:
    """original_dtype field is always set and round-trips correctly."""
    serialized = serialize_tensor(torch.randn(2, 3, dtype=dtype))
    assert serialized["original_dtype"] == str(dtype)
