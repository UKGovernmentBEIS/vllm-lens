"""Process-local, bounded, TTL'd store for binary activation payloads.

Background (issue #31): activations returned over the OpenAI-compatible API are
serialized as ``zstd`` + base64 inside the JSON body. base64 inflates the
(already compressed) payload by ~33% and forces the whole blob to be
materialized and parsed in one piece on both ends, which caps inline-capture
throughput well below the forward pass.

When a client negotiates binary transport (``activations_transport="binary"`` in
the request's ``extra_args`` / ``vllm_xargs``), the raw ``zstd`` tensor bytes are
parked here under an unguessable handle and the JSON response carries only that
handle plus the dtype/shape metadata. The client then fetches the bytes from
``GET /v1/activations/{handle}`` as ``application/octet-stream`` — no base64, no
one-piece JSON. The default (base64-in-JSON) path is untouched.

Security / resource properties:

- **Unguessable handles** (``secrets.token_urlsafe``): a client cannot read
  another request's activations by iterating ids.
- **Bounded memory**: a total-byte cap plus a per-entry TTL. Every insert first
  drops expired entries, then evicts oldest-first until back under the cap. The
  most-recently-stored entry is never evicted by its own insert, so a single
  capture larger than the cap is still retrievable once (the cap is a soft
  ceiling, honoured against everything except the newest entry).
- **Bounded lifetime**: an entry is unreadable after its TTL and is dropped
  lazily on the next access or insert.
- **Process-local**: a handle is only valid on the replica that produced it. In
  a multi-replica / load-balanced deployment either pin activation fetches to
  the producing replica (sticky routing) or keep the default base64 transport.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

_DEFAULT_TTL_S = 300.0
_DEFAULT_MAX_BYTES = 8 * 1024**3  # 8 GiB
_DEFAULT_MAX_ENTRIES = 512  # bound entry count (and per-entry overhead) too
# Handle entropy: 24 bytes → 192 bits, url-safe base64. Far beyond guessable.
_HANDLE_NBYTES = 24


@dataclass
class _Entry:
    payload: bytes
    meta: dict[str, Any]
    expires_at: float
    nbytes: int


class ActivationStore:
    """A thread-safe, bounded, TTL'd bytes store keyed by unguessable handles."""

    def __init__(
        self,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        ttl_s: float = _DEFAULT_TTL_S,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_bytes = int(max_bytes)
        self._ttl_s = float(ttl_s)
        self._max_entries = int(max_entries)
        self._clock = clock
        self._lock = threading.Lock()
        # Insertion-ordered so eviction is oldest-first.
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._total_bytes = 0

    def put(self, payload: bytes, meta: dict[str, Any]) -> str:
        """Store ``payload`` under a fresh handle and return the handle."""
        return self.put_many([(payload, meta)])[0]

    def put_many(self, items: Sequence[tuple[bytes, dict[str, Any]]]) -> list[str]:
        """Store several payloads atomically; return their handles in order.

        All of one response's tensors go in under a single sweep/evict pass
        and none of them can be evicted by that pass — so a multi-tensor
        capture (e.g. ``attn_q`` + ``attn_k`` + ``residual_stream``) is
        either fully retrievable or, if the store is later over cap, evicted
        oldest-first as a unit relative to other captures.
        """
        now = self._clock()
        entries = [
            (
                secrets.token_urlsafe(_HANDLE_NBYTES),
                _Entry(
                    payload=payload,
                    meta=dict(meta),
                    expires_at=now + self._ttl_s,
                    nbytes=len(payload),
                ),
            )
            for payload, meta in items
        ]
        with self._lock:
            self._drop_expired_locked(now)
            for handle, entry in entries:
                self._entries[handle] = entry
                self._total_bytes += entry.nbytes
            self._evict_to_fit_locked(protect=len(entries))
        return [handle for handle, _ in entries]

    def get(self, handle: str) -> tuple[bytes, dict[str, Any]] | None:
        """Return ``(payload, meta)`` for a live handle, else ``None``.

        A miss (unknown or expired handle) is indistinguishable to the caller,
        so a 404 leaks nothing about whether a handle ever existed.
        """
        now = self._clock()
        with self._lock:
            # Sweep on read too, so expired payloads are released even when no
            # new capture arrives to trigger put()'s sweep (bounded work: the
            # store holds at most ``max_entries``).
            self._drop_expired_locked(now)
            entry = self._entries.get(handle)
            if entry is None:
                return None
            return entry.payload, dict(entry.meta)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    # -- internals (call with the lock held) --------------------------------

    def _remove_locked(self, handle: str) -> None:
        entry = self._entries.pop(handle, None)
        if entry is not None:
            self._total_bytes -= entry.nbytes

    def _drop_expired_locked(self, now: float) -> None:
        expired = [h for h, e in self._entries.items() if e.expires_at <= now]
        for h in expired:
            self._remove_locked(h)

    def _evict_to_fit_locked(self, *, protect: int = 1) -> None:
        # Oldest-first on either the byte cap or the entry-count cap, but never
        # evict the ``protect`` newest entries (the batch just added): this
        # keeps the most recent capture retrievable even if it alone exceeds
        # ``max_bytes`` (a single all-layer capture can be large — rejecting
        # it would defeat the feature). Peak retained memory is therefore
        # ``max_bytes + largest_single_capture``; the count cap bounds
        # per-entry object/metadata overhead independently.
        while (
            self._total_bytes > self._max_bytes
            or len(self._entries) > self._max_entries
        ) and len(self._entries) > protect:
            oldest, _ = next(iter(self._entries.items()))
            self._remove_locked(oldest)


def _store_from_env() -> ActivationStore:
    max_bytes = int(os.environ.get("VLLM_LENS_ACT_MAX_BYTES", _DEFAULT_MAX_BYTES))
    ttl_s = float(os.environ.get("VLLM_LENS_ACT_TTL_S", _DEFAULT_TTL_S))
    max_entries = int(os.environ.get("VLLM_LENS_ACT_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES))
    return ActivationStore(max_bytes=max_bytes, ttl_s=ttl_s, max_entries=max_entries)


# Process-wide singleton shared by the response patches (producers) and the
# ``GET /v1/activations/{handle}`` route (consumer). Configured once from env.
store = _store_from_env()
