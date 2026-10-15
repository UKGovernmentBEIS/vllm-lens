"""Position-selective residual-stream capture (issue #40), CPU only.

Drives the worker's capture hook the way vLLM's scheduler would (one call per
layer per forward, chunked prefill, mixed-length batches, decode steps) with
vLLM's two imports stubbed out, then checks the payload builder, the plugin's
cross-rank merge and trim, the wire round trip and the client encoding. The
value-level checks against a live engine are the GPU suites.
"""

from __future__ import annotations

import importlib
import json
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens._helpers._activation_store import ActivationStore
from vllm_lens._helpers._positions import (
    LAST_PROMPT_TOKEN,
    POSITIONS_KEY,
    POSITIONS_META_KEY,
    normalize_capture_positions,
    order_positions,
    parse_capture_positions,
    positions_in_slice,
    select_rows,
)
from vllm_lens._helpers._serialize import (
    decode_activations,
    serialize_activations,
    serialize_activations_binary,
    tensor_from_bytes,
)
from vllm_lens.client import VLLMLensClient

HIDDEN = 4
LAYERS = [2, 5]
CAPTURE = {"output_residual_stream": LAYERS}


# ---------------------------------------------------------------------------
# Selector parsing and entry-point validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("null", None),
        (LAST_PROMPT_TOKEN, LAST_PROMPT_TOKEN),
        (json.dumps(LAST_PROMPT_TOKEN), LAST_PROMPT_TOKEN),
        ([5, 2, 5], [2, 5]),
        ((5, 2), [2, 5]),
        ("[5, 2, 5]", [2, 5]),
        ([0], [0]),
    ],
)
def test_parse_capture_positions(value, expected):
    assert parse_capture_positions(value) == expected


@pytest.mark.parametrize(
    "value",
    [[-1], [1.5], [True], ["3"], [], "last", "last token", "{}", {"a": 1}, 3, "3"],
)
def test_parse_rejects_invalid_selectors(value):
    with pytest.raises(ValueError, match=POSITIONS_KEY):
        parse_capture_positions(value)


def test_normalize_rewrites_extra_args_in_place():
    extra = {"output_residual_stream": [2], POSITIONS_KEY: "[5, 2, 5]"}
    normalize_capture_positions(extra)
    assert extra[POSITIONS_KEY] == [2, 5]
    extra = {
        "output_residual_stream": [2],
        POSITIONS_KEY: json.dumps(LAST_PROMPT_TOKEN),
    }
    normalize_capture_positions(extra)
    assert extra[POSITIONS_KEY] == LAST_PROMPT_TOKEN
    extra = {"output_residual_stream": [2], POSITIONS_KEY: None}
    normalize_capture_positions(extra)
    assert POSITIONS_KEY not in extra
    for untouched in (None, {}, {"output_residual_stream": [2]}):
        normalize_capture_positions(untouched)


def test_normalize_requires_residual_stream_capture():
    with pytest.raises(ValueError, match="requires output_residual_stream"):
        normalize_capture_positions({POSITIONS_KEY: LAST_PROMPT_TOKEN})


def test_positions_in_slice_resolves_last_prompt_token_once():
    # A 10-token prompt prefilled in chunks of 4, 4, 2, then two decode steps.
    slices = [(0, 4), (4, 4), (8, 2), (10, 1), (11, 1)]
    hits = [positions_in_slice(LAST_PROMPT_TOKEN, s, n, 10) for s, n in slices]
    assert hits == [[], [], [9], [], []]
    hits = [positions_in_slice([0, 3, 4, 9, 11, 500], s, n, 10) for s, n in slices]
    assert hits == [[0, 3], [4], [9], [], [11]]


def test_select_rows_views_contiguous_runs_and_gathers_gaps():
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    run = select_rows(hidden, start=2, abs_start=10, positions=[11, 12])
    assert run.data_ptr() == hidden[3].data_ptr()  # a view, no copy
    torch.testing.assert_close(run, hidden[3:5])
    gathered = select_rows(hidden, start=2, abs_start=10, positions=[10, 13])
    torch.testing.assert_close(gathered, hidden[[2, 5]])
    assert select_rows(hidden, 2, 10, []).shape == (0, 4)


def test_order_positions_keeps_latest_row_per_position():
    assert order_positions([3, 7, 3, 1]) == ([1, 3, 7], [3, 2, 1])
    assert order_positions([]) == ([], [])


# ---------------------------------------------------------------------------
# Worker capture hook under a fake scheduler
# ---------------------------------------------------------------------------


@pytest.fixture
def worker(monkeypatch):
    """Import the worker extension with vLLM's two imports stubbed out."""
    forward_context = ModuleType("vllm.forward_context")
    forward_context.is_forward_context_available = lambda: True
    forward_context.get_forward_context = lambda: None
    model_utils = ModuleType("vllm.model_executor.models.utils")
    model_utils.PPMissingLayer = type("PPMissingLayer", (), {})
    monkeypatch.setitem(sys.modules, forward_context.__name__, forward_context)
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)
    sys.modules.pop("vllm_lens._worker_ext", None)
    module = importlib.import_module("vllm_lens._worker_ext")
    yield module
    sys.modules.pop("vllm_lens._worker_ext", None)


class FakeEngine:
    """Run the capture hook as vLLM would: once per layer for every forward.

    Each hidden-state row encodes ``[request index, absolute position, layer,
    salt]`` so a captured row identifies exactly which token it came from.
    """

    def __init__(self, worker, monkeypatch, requests: dict[str, tuple[int, dict]]):
        self.worker = worker
        self.monkeypatch = monkeypatch
        self.ids = list(requests)
        self.requests = {
            req_id: SimpleNamespace(
                sampling_params=SimpleNamespace(extra_args=dict(extra)),
                num_computed_tokens=0,
                num_prompt_tokens=n_prompt,
            )
            for req_id, (n_prompt, extra) in requests.items()
        }
        ext = worker.HiddenStatesExtension()
        ext.model_runner = SimpleNamespace(
            input_batch=SimpleNamespace(num_reqs=0, req_ids=[]),
            requests=self.requests,
            model=object(),
        )
        ext._captured_states = {}
        ext._captured_positions = {}
        ext._steering_data = {}
        ext._hook_data = {}
        ext._persistent_hooks = []
        ext._hook_contexts = {}
        ext._persistent_hook_contexts = {}
        ext._prefetched_params = {}
        ext._should_capture = True
        self.ext = ext

    def step(self, scheduled: dict[str, int], salt: float = 0.0) -> None:
        req_ids = list(scheduled)
        starts, rows = [0], []
        for req_id in req_ids:
            state = self.requests[req_id]
            for k in range(scheduled[req_id]):
                rows.append(
                    [self.ids.index(req_id), state.num_computed_tokens + k, 0, salt]
                )
            starts.append(starts[-1] + scheduled[req_id])
        base = torch.tensor(rows, dtype=torch.float32)
        batch = self.ext.model_runner.input_batch
        batch.num_reqs, batch.req_ids = len(req_ids), req_ids
        query_start_loc = torch.tensor(starts)
        self.monkeypatch.setattr(
            self.worker,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata={"attn": SimpleNamespace(query_start_loc=query_start_loc)}
            ),
        )
        for layer_idx in LAYERS:
            hidden = base.clone()
            hidden[:, 2] = layer_idx
            assert self.worker._hook_inner(self.ext, layer_idx, hidden) is None
        for req_id, n in scheduled.items():
            self.requests[req_id].num_computed_tokens += n

    def prefill(self, req_id: str, chunks: list[int], decode: int = 0) -> None:
        for chunk in chunks:
            self.step({req_id: chunk})
        for _ in range(decode):
            self.step({req_id: 1})

    def activations(self, req_id: str) -> dict:
        return self.ext._build_payload(req_id)["activations"]


def _check_rows(acts: dict, req_index: int, positions: list[int]) -> None:
    rs = acts["residual_stream"]
    assert rs.shape == (len(LAYERS), len(positions), HIDDEN)
    for li, layer_idx in enumerate(LAYERS):
        assert rs[li, :, 0].tolist() == [req_index] * len(positions)
        assert rs[li, :, 1].tolist() == positions
        assert rs[li, :, 2].tolist() == [layer_idx] * len(positions)


def test_default_capture_is_unchanged(worker, monkeypatch):
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, CAPTURE)})
    engine.prefill("a-0", [4, 4, 2], decode=2)
    acts = engine.activations("a-0")
    _check_rows(acts, 0, list(range(12)))
    assert POSITIONS_META_KEY not in acts
    assert engine.ext._captured_positions == {}


def test_last_prompt_token_ignores_chunk_boundaries(worker, monkeypatch):
    selector = {**CAPTURE, POSITIONS_KEY: LAST_PROMPT_TOKEN}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.prefill("a-0", [4, 4])
    # Chunk endpoints 3 and 7 are not the last prompt token: nothing kept yet.
    assert torch.cat(engine.ext._captured_states["a-0"][LAYERS[0]]).shape == (0, HIDDEN)
    engine.prefill("a-0", [2], decode=2)
    acts = engine.activations("a-0")
    _check_rows(acts, 0, [9])
    assert acts[POSITIONS_META_KEY] == [9]


def test_explicit_positions_span_prompt_and_decode(worker, monkeypatch):
    # Unsorted, duplicated and never-computed entries, as a bypassing caller
    # could pass them straight to the worker.
    selector = {**CAPTURE, POSITIONS_KEY: [11, 0, 5, 9, 500, 5]}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.prefill("a-0", [4, 4, 2], decode=3)
    acts = engine.activations("a-0")
    _check_rows(acts, 0, [0, 5, 9, 11])
    assert acts[POSITIONS_META_KEY] == [0, 5, 9, 11]


def test_mixed_batch_keeps_requests_apart(worker, monkeypatch):
    engine = FakeEngine(
        worker,
        monkeypatch,
        {
            "a-0": (10, {**CAPTURE, POSITIONS_KEY: LAST_PROMPT_TOKEN}),
            "b-0": (3, CAPTURE),
            "c-0": (6, {**CAPTURE, POSITIONS_KEY: [2, 7]}),
        },
    )
    engine.step({"a-0": 4, "b-0": 3, "c-0": 6})
    engine.step({"a-0": 4, "b-0": 1, "c-0": 1})
    engine.step({"a-0": 2, "b-0": 1, "c-0": 1})
    engine.step({"a-0": 1, "b-0": 1, "c-0": 1})
    a, b, c = (engine.activations(r) for r in ("a-0", "b-0", "c-0"))
    _check_rows(a, 0, [9])
    assert a[POSITIONS_META_KEY] == [9]
    _check_rows(b, 1, list(range(6)))
    assert POSITIONS_META_KEY not in b
    _check_rows(c, 2, [2, 7])
    assert c[POSITIONS_META_KEY] == [2, 7]


def test_only_selected_rows_reach_cpu(worker, monkeypatch):
    copied: list[int] = []
    original_cpu = torch.Tensor.cpu
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda t: (copied.append(t.shape[0]), original_cpu(t))[1]
    )
    selector = {**CAPTURE, POSITIONS_KEY: LAST_PROMPT_TOKEN}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (40, selector)})
    engine.prefill("a-0", [16, 16, 8], decode=2)
    assert max(copied) == 1  # never a whole chunk
    rs_long = engine.activations("a-0")["residual_stream"]
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.prefill("a-0", [10], decode=2)
    rs_short = engine.activations("a-0")["residual_stream"]
    assert rs_long.shape == rs_short.shape == (len(LAYERS), 1, HIDDEN)
    assert (
        rs_long.numel() * rs_long.element_size()
        == rs_short.numel() * rs_short.element_size()
    )


def test_non_capturing_tp_rank_stores_nothing(worker, monkeypatch):
    selector = {**CAPTURE, POSITIONS_KEY: LAST_PROMPT_TOKEN}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.ext._should_capture = False
    engine.prefill("a-0", [10], decode=1)
    assert engine.ext._captured_states == {}
    assert engine.ext._captured_positions == {}


def test_selector_matching_nothing_yields_empty_rows(worker, monkeypatch):
    selector = {**CAPTURE, POSITIONS_KEY: [500]}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.prefill("a-0", [10], decode=2)
    acts = engine.activations("a-0")
    assert acts["residual_stream"].shape == (len(LAYERS), 0, HIDDEN)
    assert acts[POSITIONS_META_KEY] == []


def test_recomputed_positions_keep_the_latest_row(worker, monkeypatch):
    # Preemption recomputes the prompt from position 0; the second pass wins.
    selector = {**CAPTURE, POSITIONS_KEY: [1, 5]}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.prefill("a-0", [4, 6])
    engine.requests["a-0"].num_computed_tokens = 0
    engine.step({"a-0": 10}, salt=1.0)
    acts = engine.activations("a-0")
    _check_rows(acts, 0, [1, 5])
    assert acts["residual_stream"][:, :, 3].tolist() == [[1.0, 1.0]] * len(LAYERS)
    assert acts[POSITIONS_META_KEY] == [1, 5]


def test_clear_captured_states_drops_positions(worker, monkeypatch):
    selector = {**CAPTURE, POSITIONS_KEY: LAST_PROMPT_TOKEN}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.prefill("a-0", [10])
    assert "a-0" in engine.ext._captured_positions
    engine.ext.clear_captured_states("a")
    assert engine.ext._captured_states == {}
    assert engine.ext._captured_positions == {}


# ---------------------------------------------------------------------------
# Plugin side: cross-rank merge, trim, entry points
# ---------------------------------------------------------------------------


def _rank_payload(n_layers: int, positions: list[int] | None, req: str = "a") -> bytes:
    acts: dict = {
        "residual_stream": torch.full(
            (n_layers, len(positions or [3]), HIDDEN), float(n_layers)
        )
    }
    if positions is not None:
        acts[POSITIONS_META_KEY] = positions
    return pickle.dumps({req: {"activations": acts}})


def test_merge_carries_positions_across_ranks(worker, monkeypatch):
    selector = {**CAPTURE, POSITIONS_KEY: LAST_PROMPT_TOKEN}
    engine = FakeEngine(worker, monkeypatch, {"a-0": (10, selector)})
    engine.prefill("a-0", [10], decode=1)
    rank0 = engine.ext.get_captured_states("a")
    assert rank0 is not None and engine.ext.get_captured_states("a") is None
    # TP=2: rank 1 does not capture and returns None.
    merged = plugin._merge_captured_states([rank0, None])
    assert merged is not None and merged[POSITIONS_META_KEY] == [9]
    assert merged["residual_stream"].shape == (len(LAYERS), 1, HIDDEN)
    # PP=2: each stage holds its own layers; the layer dim concatenates.
    stage0 = pickle.dumps(
        {
            "activations": {
                "residual_stream": torch.zeros(1, 1, HIDDEN),
                POSITIONS_META_KEY: [9],
            }
        }
    )
    stage1 = pickle.dumps(
        {
            "activations": {
                "residual_stream": torch.ones(2, 1, HIDDEN),
                POSITIONS_META_KEY: [9],
            }
        }
    )
    merged = plugin._merge_captured_states([stage0, stage1])
    assert merged is not None and merged["residual_stream"].shape == (3, 1, HIDDEN)
    assert merged[POSITIONS_META_KEY] == [9]


def test_batched_merge_carries_positions_per_request():
    out = plugin._merge_captured_states_batch(
        [_rank_payload(1, [9]), _rank_payload(2, [9])], ["a", "missing"]
    )
    assert list(out) == ["a"]
    assert out["a"]["residual_stream"].shape == (3, 1, HIDDEN)
    assert out["a"][POSITIONS_META_KEY] == [9]
    plain = plugin._merge_captured_states_batch([_rank_payload(1, None)], ["a"])
    assert POSITIONS_META_KEY not in plain["a"]


def test_trim_drops_rows_from_the_surplus_pass():
    acts = {
        "residual_stream": torch.arange(3.0).view(1, 3, 1),
        POSITIONS_META_KEY: [0, 9, 12],
    }
    plugin._trim_activations(acts, expected_len=12)
    assert acts["residual_stream"].flatten().tolist() == [0.0, 1.0]
    assert acts[POSITIONS_META_KEY] == [0, 9]
    kept = {"residual_stream": torch.zeros(1, 2, 1), POSITIONS_META_KEY: [0, 9]}
    rs = kept["residual_stream"]
    plugin._trim_activations(kept, expected_len=12)
    assert kept["residual_stream"] is rs and kept[POSITIONS_META_KEY] == [0, 9]
    plain = {"residual_stream": torch.zeros(1, 13, 1)}
    plugin._trim_activations(plain, expected_len=12)
    assert plain["residual_stream"].shape == (1, 12, 1)


def test_offline_entry_point_validates_before_any_rpc():
    calls: list[str] = []
    llm = SimpleNamespace(collective_rpc=lambda name, args=(): calls.append(name))
    bad = SimpleNamespace(
        extra_args={"output_residual_stream": [2], POSITIONS_KEY: [-1]},
        skip_reading_prefix_cache=False,
    )
    with pytest.raises(ValueError, match=POSITIONS_KEY):
        plugin._prepare_offline_params(llm, [bad])  # type: ignore[arg-type]
    assert calls == []
    good = SimpleNamespace(
        extra_args={"output_residual_stream": [2], POSITIONS_KEY: "[5, 2]"},
        skip_reading_prefix_cache=False,
    )
    state = plugin._prepare_offline_params(llm, [good])  # type: ignore[arg-type]
    assert good.extra_args[POSITIONS_KEY] == [2, 5]
    assert good.skip_reading_prefix_cache is True
    assert state["wants_activations"] and calls == ["install_hooks"]


async def test_async_entry_point_attaches_trimmed_selection(monkeypatch):
    calls: list[tuple[str, tuple]] = []
    payload = pickle.dumps(
        {
            "activations": {
                "residual_stream": torch.arange(3.0).view(1, 3, 1),
                POSITIONS_META_KEY: [0, 9, 12],
            }
        }
    )

    async def collective_rpc(name, args=()):
        calls.append((name, args))
        if name == "get_captured_states":
            return [payload, None]
        return None

    async def original_generate(self, prompt, sampling_params, request_id, **kwargs):
        yield SimpleNamespace(
            finished=True,
            prompt_token_ids=[0] * 10,
            outputs=[SimpleNamespace(token_ids=[1, 2, 3])],
        )

    monkeypatch.setattr(plugin, "_original_generate", original_generate)
    engine = SimpleNamespace(collective_rpc=collective_rpc, _hooks_installed=True)
    params = SimpleNamespace(
        extra_args={"output_residual_stream": [2], POSITIONS_KEY: "[12, 9, 0]"},
        skip_reading_prefix_cache=False,
    )
    outputs = [o async for o in plugin._patched_generate(engine, "hi", params, "req")]  # type: ignore[arg-type]
    assert params.extra_args[POSITIONS_KEY] == [0, 9, 12]
    acts = outputs[-1].activations
    # 10 prompt + 3 generated tokens compute positions 0..11; 12 is surplus.
    assert acts["residual_stream"].flatten().tolist() == [0.0, 1.0]
    assert acts[POSITIONS_META_KEY] == [0, 9]
    assert [name for name, _ in calls] == [
        "get_captured_states",
        "clear_captured_states",
    ]

    bad = SimpleNamespace(
        extra_args={POSITIONS_KEY: LAST_PROMPT_TOKEN}, skip_reading_prefix_cache=False
    )
    with pytest.raises(ValueError, match="requires output_residual_stream"):
        async for _ in plugin._patched_generate(engine, "hi", bad, "req"):  # type: ignore[arg-type]
            pass


# ---------------------------------------------------------------------------
# Wire format and client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transport", ["base64", "binary"])
def test_positions_survive_both_transports(transport):
    acts = {
        "residual_stream": torch.randn(2, 1, HIDDEN, dtype=torch.bfloat16),
        POSITIONS_META_KEY: [9],
    }
    store = ActivationStore()
    if transport == "binary":
        serialized = serialize_activations_binary(acts, store)
        assert "handle" in serialized["residual_stream"]
    else:
        serialized = serialize_activations(acts)
    response = {"activations": json.loads(json.dumps(serialized))}  # JSON-safe
    decoded = decode_activations(
        response,
        fetch_bytes=lambda h: store.get(h)[0],  # type: ignore[index]
    )
    assert torch.equal(decoded["residual_stream"], acts["residual_stream"])
    assert decoded[POSITIONS_META_KEY] == [9]
    if transport == "binary":
        raw, meta = store.get(serialized["residual_stream"]["handle"])  # type: ignore[misc]
        assert torch.equal(tensor_from_bytes(raw, meta), acts["residual_stream"])


def test_client_encodes_capture_positions():
    client = VLLMLensClient("http://localhost:8000")
    xargs = client._build_xargs(None, [2], None, capture_positions=LAST_PROMPT_TOKEN)
    assert xargs == {"output_residual_stream": "[2]", POSITIONS_KEY: LAST_PROMPT_TOKEN}
    xargs = client._build_xargs(None, [2], None, capture_positions=[5, 0])
    assert xargs[POSITIONS_KEY] == "[5, 0]"
    assert parse_capture_positions(xargs[POSITIONS_KEY]) == [0, 5]
    assert POSITIONS_KEY not in client._build_xargs(None, [2], None)
