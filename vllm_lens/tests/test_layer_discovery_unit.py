"""CPU-only unit tests for decoder-layer discovery helpers.

Exercises the pure logic in ``_worker_ext`` on hand-built ``nn.Module``
trees and fake registries — no GPU or model download required.  The
end-to-end registry test on a real model lives in ``test_layer_discovery.py``.
"""

from types import SimpleNamespace

import pytest
from torch import nn
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._worker_ext import (
    _LAYER_PATH_ENV,
    LayerDiscoveryError,
    _discover_layer_modules,
    _is_decoder_mixer,
    _layers_from_path_template,
    _layers_from_registry,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.attn = nn.Identity()


class _Model(nn.Module):
    """``model.layers.{i}`` layout, with optional PP-missing stages."""

    def __init__(self, n: int, missing: frozenset[int] = frozenset()) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [PPMissingLayer() if i in missing else _Block() for i in range(n)]
        )


def _registry_for(model: _Model, mixer_cls: type) -> dict[str, nn.Module]:
    reg: dict[str, nn.Module] = {}
    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer, PPMissingLayer):
            reg[f"model.layers.{i}.self_attn.attn"] = mixer_cls()
    return reg


class _FakeMixer(nn.Module):
    pass


class _FakeEncoderMixer(nn.Module):
    attn_type = "encoder"


def _fake_worker(
    model: nn.Module,
    registry: dict[str, nn.Module] | None,
    total: int,
    local: int | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        get_total_num_hidden_layers=lambda: total,
        get_num_layers=lambda _pc: total if local is None else local,
    )
    return SimpleNamespace(
        model_runner=SimpleNamespace(model=model),
        model_config=cfg,
        parallel_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(static_forward_context=registry or {}),
    )


# --------------------------------------------------------------------------
# _is_decoder_mixer
# --------------------------------------------------------------------------


def test_mixer_allowlist_excludes_unknown_types():
    assert _is_decoder_mixer(_FakeMixer(), (_FakeMixer,))
    assert not _is_decoder_mixer(nn.Identity(), (_FakeMixer,))


def test_mixer_allowlist_excludes_non_decoder_attn_type():
    assert not _is_decoder_mixer(_FakeEncoderMixer(), (_FakeEncoderMixer,))


# --------------------------------------------------------------------------
# _layers_from_registry
# --------------------------------------------------------------------------


def test_registry_maps_prefix_to_decoder_layer(monkeypatch):
    monkeypatch.setattr(
        "vllm_lens._worker_ext._decoder_mixer_types", lambda: (_FakeMixer,)
    )
    model = _Model(4)
    out = _layers_from_registry(model, _registry_for(model, _FakeMixer))
    assert sorted(out) == [0, 1, 2, 3]
    assert out[2] is model.model.layers[2]


def test_registry_ignores_non_allowlisted_entries(monkeypatch):
    """A KV-cache dummy registered one past the last layer must be dropped."""
    monkeypatch.setattr(
        "vllm_lens._worker_ext._decoder_mixer_types", lambda: (_FakeMixer,)
    )
    model = _Model(4)
    model.cache_only_layers = nn.ModuleDict({"4": nn.Identity()})
    reg = _registry_for(model, _FakeMixer)
    reg["cache_only_layers.4"] = nn.Identity()  # not in the allowlist
    assert sorted(_layers_from_registry(model, reg)) == [0, 1, 2, 3]


def test_registry_rejects_prefix_without_index(monkeypatch):
    monkeypatch.setattr(
        "vllm_lens._worker_ext._decoder_mixer_types", lambda: (_FakeMixer,)
    )
    model = _Model(1)
    model.model.final_attn = _FakeMixer()
    with pytest.raises(LayerDiscoveryError, match="no integer layer index"):
        _layers_from_registry(model, {"model.final_attn": _FakeMixer()})


# --------------------------------------------------------------------------
# _layers_from_path_template
# --------------------------------------------------------------------------


def test_path_template_resolves_and_skips_pp_missing():
    model = _Model(4, missing={0, 1})
    out = _layers_from_path_template(model, "model.layers.{i}", 4)
    assert sorted(out) == [2, 3]
    assert out[3] is model.model.layers[3]


def test_path_template_requires_placeholder():
    with pytest.raises(LayerDiscoveryError, match="{i}"):
        _layers_from_path_template(_Model(2), "model.layers", 2)


def test_path_template_names_bad_path():
    with pytest.raises(LayerDiscoveryError, match="model.blocks.0"):
        _layers_from_path_template(_Model(2), "model.blocks.{i}", 2)


# --------------------------------------------------------------------------
# _discover_layer_modules (orchestration + count check)
# --------------------------------------------------------------------------


def test_discover_uses_registry_by_default(monkeypatch):
    monkeypatch.delenv(_LAYER_PATH_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_lens._worker_ext._decoder_mixer_types", lambda: (_FakeMixer,)
    )
    model = _Model(3)
    worker = _fake_worker(model, _registry_for(model, _FakeMixer), total=3)
    assert sorted(_discover_layer_modules(worker)) == [0, 1, 2]


def test_discover_env_override_bypasses_registry(monkeypatch):
    monkeypatch.setenv(_LAYER_PATH_ENV, "model.layers.{i}")
    model = _Model(3)
    # Registry deliberately garbage — must not be consulted.
    worker = _fake_worker(model, {"nonsense": nn.Identity()}, total=3)
    assert sorted(_discover_layer_modules(worker)) == [0, 1, 2]


def test_discover_errors_on_empty_registry(monkeypatch):
    monkeypatch.delenv(_LAYER_PATH_ENV, raising=False)
    worker = _fake_worker(_Model(3), {}, total=3)
    with pytest.raises(LayerDiscoveryError, match=_LAYER_PATH_ENV):
        _discover_layer_modules(worker)


def test_discover_errors_on_count_mismatch(monkeypatch):
    """Registry finds fewer layers than the config declares → loud error."""
    monkeypatch.delenv(_LAYER_PATH_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_lens._worker_ext._decoder_mixer_types", lambda: (_FakeMixer,)
    )
    model = _Model(4)
    reg = _registry_for(model, _FakeMixer)
    del reg["model.layers.1.self_attn.attn"]
    worker = _fake_worker(model, reg, total=4)
    with pytest.raises(LayerDiscoveryError, match="found 3 layer"):
        _discover_layer_modules(worker)


def test_discover_errors_on_out_of_range_index(monkeypatch):
    monkeypatch.delenv(_LAYER_PATH_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_lens._worker_ext._decoder_mixer_types", lambda: (_FakeMixer,)
    )
    model = _Model(2)
    model.model.layers.append(_Block())  # module tree has 3, config says 2
    reg = _registry_for(model, _FakeMixer)
    worker = _fake_worker(model, reg, total=2)
    with pytest.raises(LayerDiscoveryError, match="out of range: \\[2\\]"):
        _discover_layer_modules(worker)


def test_discover_pp_expects_local_count_only(monkeypatch):
    """Under PP this rank owns a subset; the check uses the local count."""
    monkeypatch.delenv(_LAYER_PATH_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_lens._worker_ext._decoder_mixer_types", lambda: (_FakeMixer,)
    )
    model = _Model(4, missing={0, 1})
    worker = _fake_worker(model, _registry_for(model, _FakeMixer), total=4, local=2)
    assert sorted(_discover_layer_modules(worker)) == [2, 3]
