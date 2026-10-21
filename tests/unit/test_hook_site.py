"""CPU tests for ``Hook.site`` and the ``HookContext`` offset fields."""

from __future__ import annotations

import pydantic
import pytest

from vllm_lens._helpers.types import Hook, HookContext


def _noop(ctx, x):
    return None


def test_default_site_is_layer() -> None:
    assert Hook(fn=_noop, layer_indices=[0]).site == "layer"


def test_mlp_act_site_json_round_trip() -> None:
    hook = Hook(fn=_noop, layer_indices=[2, 3], site="mlp_act")
    restored = Hook.model_validate(hook.model_dump())
    assert restored.site == "mlp_act"
    assert restored.layer_indices == [2, 3]
    assert restored.has_layer(3)


def test_pre_rejected_for_mlp_act() -> None:
    with pytest.raises(pydantic.ValidationError, match="pre=True is not supported"):
        Hook(fn=_noop, layer_indices=[0], site="mlp_act", pre=True)


def test_unknown_site_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        Hook(fn=_noop, layer_indices=[0], site="attn")  # type: ignore[arg-type]


def test_hook_context_defaults() -> None:
    ctx = HookContext()
    assert ctx.position_offset == 0
    assert ctx.neuron_offset == 0
