"""Generic hook eligibility and dispatch without vLLM or a GPU."""

import importlib.util
import pickle
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import cloudpickle

from vllm_lens import Hook, SteeringVector


@pytest.fixture
def worker(monkeypatch):
    forward_context = ModuleType("vllm.forward_context")
    forward_context.is_forward_context_available = lambda: True
    forward_context.get_forward_context = lambda: None
    model_utils = ModuleType("vllm.model_executor.models.utils")
    model_utils.PPMissingLayer = type("PPMissingLayer", (), {})
    monkeypatch.setitem(sys.modules, forward_context.__name__, forward_context)
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)
    path = Path(__file__).resolve().parents[2] / "vllm_lens" / "_worker_ext.py"
    spec = importlib.util.spec_from_file_location("worker_hook_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class QueryStarts:
    """Track metadata reads that would synchronize a CUDA tensor."""

    def __init__(self, values, allowed=True):
        self.values = values
        self.allowed = allowed
        self.reads = 0

    def __getitem__(self, index):
        def tolist():
            assert self.allowed, "Unexpected boundary read"
            self.reads += 1
            return self.values[index]

        return SimpleNamespace(tolist=tolist)


def make_extension(worker, monkeypatch, starts, *, persistent=(), per_request=((),)):
    req_ids = [f"r{i}-internal" for i in range(len(per_request))]
    requests = {
        req_id: SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))
        for req_id in req_ids
    }
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(req_ids), req_ids=req_ids),
        requests=requests,
        model=object(),
    )
    # vLLM keeps one forward context for every layer of a forward pass.
    forward_context = SimpleNamespace(
        attn_metadata={"attention": SimpleNamespace(query_start_loc=starts)}
    )
    monkeypatch.setattr(worker, "get_forward_context", lambda: forward_context)
    return SimpleNamespace(
        model_runner=runner,
        _persistent_hooks=list(persistent),
        _hook_data={f"r{i}": list(hooks) for i, hooks in enumerate(per_request)},
        _steering_data={},
        _hook_contexts={},
        _persistent_hook_contexts={},
        _prefetched_params={},
        _captured_states={},
    )


def test_one_plan_and_boundary_read_per_forward_pass(worker, monkeypatch):
    seen = []
    hooks = [
        Hook(
            fn=lambda ctx, h: seen.append(("pre", ctx.layer_idx)),
            layer_indices=[0, 1],
            pre=True,
        ),
        Hook(
            fn=lambda ctx, h: seen.append(("post", ctx.layer_idx)), layer_indices=[0, 1]
        ),
    ]
    starts = QueryStarts([0, 2, 3])
    extension = make_extension(worker, monkeypatch, starts, per_request=[hooks, []])
    extension.model_runner.requests["r1-internal"].sampling_params.extra_args = {
        "output_residual_stream": "[1]"
    }
    lookups = []
    find = worker._find_hook_configs_no_persistent
    monkeypatch.setattr(
        worker,
        "_find_hook_configs_no_persistent",
        lambda ext, req_id, extra: lookups.append(req_id) or find(ext, req_id, extra),
    )
    hidden = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    for layer in range(4):
        worker._pre_hook_inner(extension, layer, hidden)
        worker._hook_inner(extension, layer, hidden)

    assert lookups == ["r0-internal", "r1-internal"]
    assert starts.reads == 1
    assert seen == [("pre", 0), ("post", 0), ("pre", 1), ("post", 1)]
    assert list(extension._captured_states) == ["r1-internal"]
    torch.testing.assert_close(
        extension._captured_states["r1-internal"][1][0], hidden[2:]
    )


def test_new_forward_pass_rebuilds_plan(worker, monkeypatch):
    starts = QueryStarts([0, 2])
    extension = make_extension(worker, monkeypatch, starts)
    hidden = torch.ones(2, 1)
    assert worker._hook_inner(extension, 0, hidden) is None
    # Registrations change between scheduler steps, with a new forward context.
    extension._hook_data["r0"] = [Hook(fn=lambda _, h: h * 3, layer_indices=[0])]
    next_context = SimpleNamespace(
        attn_metadata={"attention": SimpleNamespace(query_start_loc=starts)}
    )
    monkeypatch.setattr(worker, "get_forward_context", lambda: next_context)
    torch.testing.assert_close(worker._hook_inner(extension, 0, hidden), hidden * 3)


def test_plan_tracks_joining_finishing_and_reordered_requests(worker, monkeypatch):
    starts = QueryStarts([0, 2, 3])
    extension = make_extension(worker, monkeypatch, starts, per_request=[[], []])
    extension._persistent_hooks = [Hook(fn=lambda _, h: h + 1, layer_indices=[0, 1])]
    extension.model_runner.requests["new-internal"] = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args={"output_residual_stream": [0]})
    )
    orders = [
        (["r0-internal", "r1-internal"], [0, 2, 3]),
        (["r1-internal", "new-internal", "r0-internal"], [0, 1, 4, 6]),
        (["new-internal", "r1-internal"], [0, 2, 3]),
    ]
    previous_plan = None
    for req_ids, boundaries in orders:
        extension.model_runner.input_batch.req_ids = req_ids
        extension.model_runner.input_batch.num_reqs = len(req_ids)
        locations = QueryStarts(boundaries)
        context = SimpleNamespace(
            attn_metadata={"attention": SimpleNamespace(query_start_loc=locations)}
        )
        monkeypatch.setattr(worker, "get_forward_context", lambda: context)
        hidden = torch.arange(boundaries[-1], dtype=torch.float32).reshape(-1, 1)
        for layer in (0, 1):
            torch.testing.assert_close(
                worker._hook_inner(extension, layer, hidden), hidden + 1
            )
        plan = extension._step_plan
        assert plan is not previous_plan
        assert plan.req_ids == req_ids
        assert [plan.span(i) for i in range(len(req_ids))] == list(
            zip(boundaries, boundaries[1:])
        )
        assert locations.reads == 1
        previous_plan = plan


@pytest.mark.parametrize("kind", ["steering", "hook"])
@pytest.mark.parametrize("operation", ["clear", "replace"])
def test_request_updates_release_cached_payload(worker, monkeypatch, kind, operation):
    extension = make_extension(worker, monkeypatch, QueryStarts([0, 2]))
    payload = torch.ones(1, 2)
    reference = weakref.ref(payload)
    if kind == "steering":
        extension._steering_data["r0"] = [
            SteeringVector(activations=payload, layer_indices=[3])
        ]
        cleanup = worker.HiddenStatesExtension.clear_steering_data
    else:
        extension._hook_data["r0"] = [
            Hook(fn=lambda _, h, vector=payload: h + vector, layer_indices=[3])
        ]
        cleanup = worker.HiddenStatesExtension.clear_hook_data
    del payload
    assert worker._get_step_plan(extension) is not None
    assert reference() is not None

    if operation == "clear":
        cleanup(extension, "r0")
    else:
        monkeypatch.setattr(worker, "_get_total_num_layers", lambda _: 4)
        extension.model_runner.model = torch.nn.Linear(2, 2)
        if kind == "steering":
            replacement = SteeringVector(
                activations=torch.zeros(1, 2), layer_indices=[3]
            )
            worker.HiddenStatesExtension.set_steering_data(
                extension, "r0", pickle.dumps([replacement])
            )
        else:
            replacement = Hook(fn=lambda *_: None, layer_indices=[3])
            worker.HiddenStatesExtension.set_hook_data(
                extension, "r0", cloudpickle.dumps([replacement])
            )

    assert extension._step_plan is None
    assert reference() is None


def test_steering_for_other_layers_skips_clone_and_boundaries(worker, monkeypatch):
    starts = QueryStarts([0, 2], allowed=False)
    extension = make_extension(worker, monkeypatch, starts)
    extension._steering_data["r0"] = [
        SteeringVector(activations=torch.ones(1, 2), layer_indices=[5])
    ]
    monkeypatch.setattr(
        torch.Tensor, "clone", lambda *_: pytest.fail("Inactive steering cloned")
    )
    assert worker._hook_inner(extension, 3, torch.ones(2, 2)) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ([1, 2], frozenset({1, 2})),
        ("[1, 2]", frozenset({1, 2})),
        (True, True),
        ("not json", True),
    ],
)
def test_capture_layers_parsing(worker, value, expected):
    extra = {} if value is None else {"output_residual_stream": value}
    assert worker._capture_layers(extra) == expected
