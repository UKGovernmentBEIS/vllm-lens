"""Generic hook eligibility and dispatch without vLLM or a GPU."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


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

        def item():
            assert self.allowed, "Unexpected boundary read"
            self.reads += 1
            return self.values[index]

        return SimpleNamespace(tolist=tolist, item=item)


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


def count_cpu_copies(monkeypatch):
    copies = []
    cpu = torch.Tensor.cpu

    def counted(tensor, *args, **kwargs):
        copies.append(tensor.shape[0])
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted)
    return copies


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "requested,expected_rows",
    # Adjacent capturing requests share one slice; gaps are gathered first.
    [((True, True, True), 6), ((True, None, "[0]"), 5), ((None, "[0]", None), 1)],
)
def test_capture_copies_each_layer_once(
    worker, monkeypatch, fused, requested, expected_rows
):
    starts = QueryStarts([0, 2, 3, 6])
    extension = make_extension(worker, monkeypatch, starts, per_request=[[], [], []])
    for i, value in enumerate(requested):
        if value is not None:
            extension.model_runner.requests[
                f"r{i}-internal"
            ].sampling_params.extra_args = {"output_residual_stream": value}
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    output = (hidden, torch.full_like(hidden, 100)) if fused else hidden
    stream = hidden + 100 if fused else hidden
    copies = count_cpu_copies(monkeypatch)

    assert worker._hook_inner(extension, 0, output) is None
    assert copies == [expected_rows]

    spans = [(0, 2), (2, 3), (3, 6)]
    captured = extension._captured_states
    assert set(captured) == {f"r{i}-internal" for i, v in enumerate(requested) if v}
    pointers = set()
    for i, (start, end) in enumerate(spans):
        if requested[i] is None:
            continue
        (activation,) = captured[f"r{i}-internal"][0]
        torch.testing.assert_close(activation, stream[start:end])
        pointers.add(activation.untyped_storage().data_ptr())
        if len(set(captured)) > 1:
            # Split results own their rows. (A lone result is the copy
            # itself on GPU; on CPU, .cpu() returns the input unchanged.)
            assert activation.untyped_storage().nbytes() == activation.nbytes
    assert len(pointers) == len(set(captured))


def test_no_capture_at_layer_skips_stream_sum_and_copy(worker, monkeypatch):
    starts = QueryStarts([0, 2], allowed=False)
    extension = make_extension(worker, monkeypatch, starts)
    extension.model_runner.requests["r0-internal"].sampling_params.extra_args = {
        "output_residual_stream": [5]
    }
    hidden = torch.ones(2, 2)
    copies = count_cpu_copies(monkeypatch)
    monkeypatch.setattr(
        torch.Tensor, "__add__", lambda *_: pytest.fail("Summed unused stream")
    )
    assert worker._hook_inner(extension, 3, (hidden, hidden)) is None
    assert copies == []
    assert extension._captured_states == {}
