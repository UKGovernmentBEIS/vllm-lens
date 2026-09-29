"""Binary activation transport — issue #31.

CPU-only (no GPU / vLLM): exercises the process-local :class:`ActivationStore`,
the ``GET /v1/activations/{handle}`` route, and the serialize/decode handle path.
The full online capture round-trip over a live server is covered by the
GPU-gated ``test_activations_cross_api`` suite.
"""

from __future__ import annotations

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vllm_lens._helpers._activation_store as store_mod
import vllm_lens._hooks_router as hooks_router
from vllm_lens._helpers._activation_store import ActivationStore
from vllm_lens._helpers._serialize import (
    decode_activations,
    serialize_activations,
    serialize_activations_binary,
    serialize_tensor_binary,
    tensor_from_bytes,
)

# ---------------------------------------------------------------------------
# ActivationStore
# ---------------------------------------------------------------------------


def test_store_put_get_roundtrip() -> None:
    store = ActivationStore()
    handle = store.put(b"hello", {"dtype": "int16"})
    assert store.get(handle) == (b"hello", {"dtype": "int16"})


def test_store_unknown_handle_is_none() -> None:
    assert ActivationStore().get("does-not-exist") is None


def test_store_ttl_expiry() -> None:
    now = {"t": 0.0}
    store = ActivationStore(ttl_s=50, clock=lambda: now["t"])
    handle = store.put(b"x", {})
    now["t"] = 40.0
    assert store.get(handle) is not None  # still live
    now["t"] = 60.0
    assert store.get(handle) is None  # past TTL (put@0 + 50)
    assert len(store) == 0  # dropped lazily on the expired read


def test_store_byte_cap_evicts_oldest_first() -> None:
    store = ActivationStore(max_bytes=10)
    h1 = store.put(b"aaaaa", {})  # total 5
    h2 = store.put(b"bbbbb", {})  # total 10
    h3 = store.put(b"ccccc", {})  # total 15 > 10 → evict oldest (h1)
    assert store.get(h1) is None
    assert store.get(h2) is not None
    assert store.get(h3) is not None
    assert store.total_bytes == 10


def test_store_newest_is_never_evicted_by_its_own_insert() -> None:
    store = ActivationStore(max_bytes=4)
    handle = store.put(b"payload-bigger-than-cap", {})
    assert store.get(handle) == (b"payload-bigger-than-cap", {})


def test_store_put_many_is_atomic_per_batch() -> None:
    """A multi-tensor response never evicts its own earlier tensors."""
    store = ActivationStore(max_bytes=10)
    h_old = store.put(b"oooo", {})
    h1, h2, h3 = store.put_many([(b"aaaaa", {}), (b"bbbbb", {}), (b"ccccc", {})])
    # Batch alone (15 bytes) exceeds the cap: the older entry goes, the whole
    # batch stays.
    assert store.get(h_old) is None
    assert [store.get(h) is not None for h in (h1, h2, h3)] == [True, True, True]
    assert store.total_bytes == 15


def test_store_put_many_batch_is_evicted_by_later_captures() -> None:
    store = ActivationStore(max_bytes=10)
    h1, h2 = store.put_many([(b"aaaaa", {}), (b"bbbbb", {})])
    h3 = store.put(b"ccccc", {})  # 15 > 10 → evict oldest (h1) only
    assert store.get(h1) is None
    assert store.get(h2) is not None and store.get(h3) is not None


def test_serialize_activations_binary_uses_single_batch() -> None:
    store = ActivationStore(max_bytes=1)  # every tensor alone exceeds the cap
    acts = {
        "attn_q": torch.randn(4, 8),
        "attn_k": torch.randn(4, 8),
        "residual_stream": torch.randn(2, 4, 8),
    }
    desc = serialize_activations_binary(acts, store)
    # All three handles must still be live — per-tensor puts would have
    # evicted the first two.
    for name, t in acts.items():
        got = store.get(desc[name]["handle"])
        assert got is not None, f"{name} was evicted by its sibling insert"
        assert torch.equal(tensor_from_bytes(got[0], got[1]), t)


def test_store_handles_are_unique_and_high_entropy() -> None:
    store = ActivationStore()
    handles = {store.put(b"x", {}) for _ in range(64)}
    assert len(handles) == 64  # no collisions
    assert all(len(h) >= 24 for h in handles)  # url-safe base64 of 24 bytes


def test_store_sweeps_expired_on_any_get() -> None:
    now = {"t": 0.0}
    store = ActivationStore(ttl_s=50, clock=lambda: now["t"])
    store.put(b"aaaaa", {})
    store.put(b"bbbbb", {})
    assert len(store) == 2
    now["t"] = 100.0  # both past TTL
    # A GET for an unrelated handle still releases the expired memory — TTL does
    # not depend on another insert or a fetch of the exact handle.
    assert store.get("unrelated-handle") is None
    assert len(store) == 0
    assert store.total_bytes == 0


def test_store_max_entries_evicts_oldest() -> None:
    store = ActivationStore(max_entries=3)
    handles = [store.put(bytes([i]), {}) for i in range(5)]
    assert len(store) == 3
    assert store.get(handles[0]) is None  # two oldest evicted by the count cap
    assert store.get(handles[1]) is None
    assert store.get(handles[4]) is not None  # newest kept


# ---------------------------------------------------------------------------
# serialize (handle descriptor) round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_binary_descriptor_carries_no_data(dtype: torch.dtype) -> None:
    store = ActivationStore()
    desc = serialize_tensor_binary(torch.randn(3, 4, dtype=dtype), store)
    assert "data" not in desc  # heavy bytes are NOT in the JSON
    assert desc["transport"] == "binary"
    assert desc["handle"] and desc["nbytes"] > 0
    assert desc["original_dtype"] == str(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_binary_roundtrip_is_bit_exact(dtype: torch.dtype) -> None:
    store = ActivationStore()
    original = torch.randn(2, 5, 8, dtype=dtype)
    desc = serialize_tensor_binary(original, store)
    payload, _meta = store.get(desc["handle"])  # type: ignore[misc]
    recovered = tensor_from_bytes(payload, desc)
    assert recovered.dtype == dtype
    assert torch.equal(recovered, original)


# ---------------------------------------------------------------------------
# decode_activations (both transports)
# ---------------------------------------------------------------------------


def test_decode_base64_is_backward_compatible() -> None:
    original = {"residual_stream": torch.randn(2, 3, 4)}
    resp = {"activations": serialize_activations(original)}
    out = decode_activations(resp)  # no fetch_bytes needed for base64
    assert torch.equal(out["residual_stream"], original["residual_stream"])


def test_decode_binary_uses_fetch_bytes() -> None:
    store = ActivationStore()
    original = {"residual_stream": torch.randn(2, 3, 4)}
    resp = {"activations": serialize_activations_binary(original, store)}

    def fetch(handle: str) -> bytes:
        payload, _ = store.get(handle)  # type: ignore[misc]
        return payload

    out = decode_activations(resp, fetch_bytes=fetch)
    assert torch.equal(out["residual_stream"], original["residual_stream"])


def test_decode_binary_without_fetch_bytes_raises_clearly() -> None:
    store = ActivationStore()
    resp = {
        "activations": serialize_activations_binary(
            {"residual_stream": torch.randn(2, 2)}, store
        )
    }
    with pytest.raises(ValueError, match="binary transport"):
        decode_activations(resp)


# ---------------------------------------------------------------------------
# GET /v1/activations/{handle}
# ---------------------------------------------------------------------------


def _client_for(store: ActivationStore) -> TestClient:
    store_mod.store = store  # the route reads the module-global store
    app = FastAPI()
    app.include_router(hooks_router.activations_router)
    return TestClient(app)


def test_endpoint_streams_octet_stream_and_reconstructs() -> None:
    store = ActivationStore()
    original = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    desc = serialize_tensor_binary(original, store)

    resp = _client_for(store).get(f"/v1/activations/{desc['handle']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["x-vllm-lens-original-dtype"] == "torch.bfloat16"

    recovered = tensor_from_bytes(resp.content, desc)
    assert torch.equal(recovered, original)


def test_endpoint_unknown_handle_is_404() -> None:
    resp = _client_for(ActivationStore()).get("/v1/activations/nope")
    assert resp.status_code == 404


def test_endpoint_sets_no_store_and_nosniff() -> None:
    store = ActivationStore()
    desc = serialize_tensor_binary(torch.randn(2, 2), store)
    resp = _client_for(store).get(f"/v1/activations/{desc['handle']}")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store, private"
    assert resp.headers["x-content-type-options"] == "nosniff"
