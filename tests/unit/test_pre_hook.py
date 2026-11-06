"""CPU regressions for the production decoder pre-hook and its registration."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_lens import Hook, SteeringVector


class DecoderLayer(torch.nn.Module):
    def forward(self, positions, hidden_states, residual=None, *extra, **kwargs):
        self.received = (positions, hidden_states, residual, extra, kwargs)
        return hidden_states, residual


@pytest.fixture
def worker(monkeypatch):
    # As in test_plugin_registration, stub only the imported vLLM surface.
    # Load the actual production module without leaving stub imports cached.
    context = SimpleNamespace(
        attn_metadata={
            "attention": SimpleNamespace(query_start_loc=torch.tensor([0, 2, 3]))
        }
    )
    surfaces = {
        "vllm.forward_context": {
            "get_forward_context": lambda: context,
            "is_forward_context_available": lambda: True,
        },
        "vllm.model_executor.models.utils": {"PPMissingLayer": torch.nn.Identity},
    }
    for name, attributes in surfaces.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "vllm_lens" / "_worker_ext.py"
    spec = importlib.util.spec_from_file_location("vllm_lens._worker_ext", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def extension(worker, monkeypatch):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([DecoderLayer() for _ in range(3)])
    ext = worker.HiddenStatesExtension()
    ext.model_runner = SimpleNamespace(
        model=model,
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["target-0", "other-0"]),
        requests={
            req: SimpleNamespace(
                sampling_params=SimpleNamespace(
                    extra_args={"output_residual_stream": [0, 1, 2]}
                )
            )
            for req in ("target-0", "other-0")
        },
    )
    ext.model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 3, get_num_layers=lambda _: 3
    )
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    ext.rank = 0
    ext._prefetched_params = {"test": torch.ones(1)}
    monkeypatch.setenv("VLLM_LENS_LAYER_PATH", "layers.{i}")
    ext.install_hooks()
    ext.install_hooks()  # Installation remains idempotent.
    return ext


def inputs(style, residual_kind):
    positions = torch.arange(3)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    residual = torch.full_like(hidden, 100) if residual_kind == "tensor" else None
    marker = object()
    kwargs = {"untargeted": marker}
    if style == "positional":
        args = (positions, hidden)
        if residual_kind != "omitted":
            args += (residual, marker, None)
    elif style == "keyword":
        args = ()
        kwargs.update(positions=positions, hidden_states=hidden)
    elif style == "hidden_keyword":
        args = (positions,)
        kwargs["hidden_states"] = hidden
    else:
        args = (positions, hidden)
    if style != "positional" and residual_kind != "omitted":
        kwargs["residual"] = residual
    return args, kwargs, positions, hidden, residual


STYLES = ["positional", "keyword", "hidden_keyword", "residual_keyword"]


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("residual_kind", ["omitted", "none", "tensor"])
@pytest.mark.parametrize("mode", ["replace", "inplace", "capture", "no_hooks"])
def test_installed_pre_hook(extension, caplog, style, residual_kind, mode):
    args, kwargs, positions, hidden, residual = inputs(style, residual_kind)
    original = hidden.clone()
    seen = []

    def pre(ctx, h):
        seen.append(h.clone())
        ctx.saved["input"] = h.clone()
        assert ctx.layer_idx == 0
        assert ctx.seq_len == 2
        assert ctx.model is extension.model_runner.model
        assert ctx._prefetched is extension._prefetched_params
        if mode == "replace":
            return h + 10
        if mode == "inplace":
            h.add_(10)
        return None

    if mode != "no_hooks":
        extension._hook_data = {"target": [Hook(fn=pre, layer_indices=[0], pre=True)]}
    layer = extension.model_runner.model.layers[0]
    output = layer(*args, **kwargs)
    expected = original.clone()
    if mode in ("replace", "inplace"):
        expected[:2] += 10
    torch.testing.assert_close(output[0], expected)
    assert output[1] is residual
    received = layer.received
    assert received[0] is positions
    assert received[4]["untargeted"] is kwargs["untargeted"]
    assert received[3] == args[3:]
    if mode != "inplace":
        torch.testing.assert_close(hidden, original)
    if mode in ("capture", "no_hooks", "inplace"):
        assert output[0] is hidden
    if mode != "no_hooks":
        assert len(seen) == 1
        torch.testing.assert_close(seen[0], original[:2])
        assert set(extension._hook_contexts["target-0"]) == {0}
    stream = expected if residual is None else expected + residual
    torch.testing.assert_close(extension._captured_states["target-0"][0][0], stream[:2])
    torch.testing.assert_close(extension._captured_states["other-0"][0][0], stream[2:])
    assert not caplog.records


@pytest.mark.parametrize("style", STYLES)
def test_installed_pre_hook_failure_preserves_input(extension, caplog, style):
    args, kwargs, _, hidden, residual = inputs(style, "tensor")
    original = hidden.clone()

    def fail(ctx, h):
        raise RuntimeError("intentional user hook failure")

    extension._hook_data = {
        "target": [
            Hook(fn=lambda ctx, h: h + 10, layer_indices=[0], pre=True),
            Hook(fn=fail, layer_indices=[0], pre=True),
        ]
    }
    output = extension.model_runner.model.layers[0](*args, **kwargs)
    assert output[0] is hidden
    assert output[1] is residual
    torch.testing.assert_close(hidden, original)
    assert "pre-hook error on layer 0, skipping" in caplog.text
    assert "intentional user hook failure" in caplog.text


def test_installed_mixed_layer_calls_preserve_hook_order_and_context(extension, caplog):
    events = []

    def record(name, transform):
        def hook(ctx, h):
            events.append((name, ctx.layer_idx, ctx.seq_len, h.clone()))
            ctx.saved.setdefault("layers", []).append(ctx.layer_idx)
            return transform(h)

        return hook

    layers = [0, 1, 2]
    extension._persistent_hooks = [
        Hook(fn=record("persistent_post", lambda h: None), layer_indices=layers),
        Hook(
            fn=record("persistent_pre", lambda h: h + 1), layer_indices=layers, pre=True
        ),
    ]
    extension._hook_data = {
        "target": [
            Hook(fn=record("post", lambda h: None), layer_indices=layers),
            Hook(
                fn=record("inplace", lambda h: h.mul_(2)),
                layer_indices=layers,
                pre=True,
            ),
            Hook(
                fn=record("returned", lambda h: h + 3), layer_indices=layers, pre=True
            ),
        ]
    }
    extension._steering_data = {
        "target": [
            SteeringVector(activations=torch.full((3, 2), 5.0), layer_indices=layers)
        ]
    }
    args, kwargs, positions, hidden, residual = inputs("positional", "tensor")
    expected = hidden.clone()
    for idx, layer in enumerate(extension.model_runner.model.layers):
        if idx == 0:
            hidden, actual_residual = layer(*args, **kwargs)
        elif idx == 1:
            hidden, actual_residual = layer(
                positions=positions, hidden_states=hidden, residual=residual
            )
        else:
            hidden, actual_residual = layer(positions, hidden, residual=residual)
        expected += 1  # Persistent pre-hook also runs on the other request.
        expected[:2] = expected[:2] * 2 + 3 + 5
        torch.testing.assert_close(hidden, expected)
        assert actual_residual is residual
        torch.testing.assert_close(
            extension._captured_states["target-0"][idx][0], (expected + residual)[:2]
        )
        names = [
            name
            for name, layer_idx, length, _ in events
            if layer_idx == idx and length == 2
        ]
        assert names == [
            "persistent_pre",
            "inplace",
            "returned",
            "persistent_post",
            "post",
        ]
    for store in (extension._hook_contexts, extension._persistent_hook_contexts):
        contexts = store["target-0"]
        assert len({id(ctx) for ctx in contexts.values()}) == len(contexts)
        for ctx in contexts.values():
            assert ctx.saved["layers"] == layers
    assert set(extension._hook_contexts["target-0"]) == {0, 1, 2}
    assert set(extension._persistent_hook_contexts["target-0"]) == {0, 1}
    assert not caplog.records


@pytest.mark.parametrize("style", STYLES)
def test_replacement_preserves_untargeted_containers(worker, extension, style):
    args, kwargs, _, hidden, residual = inputs(style, "tensor")
    original_kwargs = kwargs.copy()
    extension._hook_data = {
        "target": [Hook(fn=lambda ctx, h: h + 1, layer_indices=[0], pre=True)]
    }
    new_args, new_kwargs = worker._make_pre_hook(extension, 0)(None, args, kwargs)
    if "hidden_states" in kwargs:
        assert new_args is args
        assert new_kwargs is not kwargs
        modified = new_kwargs["hidden_states"]
    else:
        assert new_args is not args
        assert new_kwargs is kwargs
        modified = new_args[1]
    assert len(new_args) == len(args)
    assert new_kwargs.keys() == kwargs.keys()
    for index, value in enumerate(args):
        if index != 1:
            assert new_args[index] is value
    for key, value in kwargs.items():
        assert value is original_kwargs[key]
        if key != "hidden_states":
            assert new_kwargs[key] is value
    torch.testing.assert_close(modified[:2], hidden[:2] + 1)
    torch.testing.assert_close(modified[2:], hidden[2:])
    assert residual.eq(100).all()


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(
    "mode", ["no_hooks", "capture", "unmatched", "no_context", "failure"]
)
def test_no_replacement_keeps_original_containers(
    worker, extension, monkeypatch, style, mode
):
    args, kwargs, _, hidden, _ = inputs(style, "tensor")
    original = hidden.clone()
    original_kwargs = kwargs.copy()

    def pre(ctx, h):
        if mode == "failure":
            raise RuntimeError("intentional failure")
        ctx.saved["input"] = h.clone()

    if mode != "no_hooks":
        extension._hook_data = {
            "target": [
                Hook(
                    fn=pre, layer_indices=[1] if mode == "unmatched" else [0], pre=True
                )
            ]
        }
    if mode == "no_context":
        monkeypatch.setattr(worker, "is_forward_context_available", lambda: False)
    assert worker._make_pre_hook(extension, 0)(None, args, kwargs) is None
    assert kwargs.keys() == original_kwargs.keys()
    assert all(kwargs[key] is value for key, value in original_kwargs.items())
    torch.testing.assert_close(hidden, original)

    # Verify PyTorch retains the exact args/kwargs containers on a no-op,
    # including when the production hook catches a user-hook exception.
    received = []

    def observe(module, call_args, call_kwargs):
        received.append((call_args, call_kwargs))

    layer = extension.model_runner.model.layers[0]
    layer.register_forward_pre_hook(observe, with_kwargs=True, prepend=True)
    layer.register_forward_pre_hook(observe, with_kwargs=True)
    layer(*args, **kwargs)
    assert received[0][0] is received[1][0]
    assert received[0][1] is received[1][1]
