"""Pydantic models for vllm-lens steering vectors and hooks."""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any, Self

import cloudpickle
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from vllm_lens._helpers._serialize import deserialize_tensor, serialize_tensor


class SteeringVector(BaseModel):
    """A steering vector that modifies the residual stream during inference.

    Supports automatic serialization/deserialization of ``torch.Tensor``
    activations for JSON transport (HTTP API) and direct ``torch.Tensor``
    values for in-process usage (offline ``LLM`` / ``AsyncLLMEngine``).

    Example (offline)::

        sv = SteeringVector(
            activations=torch.randn(1, 4096),
            layer_indices=[18],
            scale=2.0,
        )

    Example (JSON round-trip)::

        data = sv.model_dump()          # base64-encoded activations
        sv2 = SteeringVector.model_validate(data)  # decoded back to tensor
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    activations: torch.Tensor
    """Steering activations.  Shape ``(n_layers, hidden_dim)`` for broadcast
    or ``(n_layers, n_positions, hidden_dim)`` for position-specific."""

    layer_indices: list[int]
    """Which model layers this steering vector applies to.  Length must
    match ``activations.shape[0]``."""

    scale: float = 1.0
    """Scalar multiplier applied to the steering vector before addition."""

    norm_match: bool = False
    """If True, rescale the modified hidden state to preserve the original
    per-token L2 norm."""

    position_indices: list[int] | None = None
    """Absolute token positions for 3D activations.  ``None`` means broadcast
    (2D) or sequential ``0..n_positions-1`` (3D)."""

    @field_validator("activations", mode="before")
    @classmethod
    def _deserialize_activations(cls, v: Any) -> torch.Tensor:
        """Accept base64 dicts (from JSON transport) or raw tensors."""
        if isinstance(v, dict) and "data" in v:
            return deserialize_tensor(v)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(
            f"activations must be a torch.Tensor or a base64 dict, got {type(v)}"
        )

    @field_serializer("activations")
    def _serialize_activations(self, v: torch.Tensor, _info: Any) -> dict[str, Any]:
        """Serialize tensor to base64 dict for JSON transport."""
        return serialize_tensor(v)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        """Validate activation tensor shape matches layer_indices."""
        if self.activations.dim() not in (2, 3):
            raise ValueError(
                f"activations must be 2D or 3D, got {self.activations.dim()}D"
            )
        if self.activations.shape[0] != len(self.layer_indices):
            raise ValueError(
                f"activations dim 0 ({self.activations.shape[0]}) must match "
                f"len(layer_indices) ({len(self.layer_indices)})"
            )
        return self

    @property
    def layer_index_map(self) -> dict[int, int]:
        """Maps actual model layer index to index into ``activations`` dim-0."""
        if not hasattr(self, "_layer_index_map_cache"):
            object.__setattr__(
                self,
                "_layer_index_map_cache",
                {li: i for i, li in enumerate(self.layer_indices)},
            )
        return self._layer_index_map_cache  # type: ignore[reportAttributeAccessIssue]



class HookContext:
    """Mutable context passed to hook functions during forward passes.

    Created per (hook, request) pair.  ``saved`` persists across layers
    for the same hook, so hooks can accumulate data as each layer fires.
    ``layer_idx`` and ``seq_len`` are updated by the dispatcher before
    each call.
    """

    __slots__ = ("layer_idx", "seq_len", "saved", "model")

    def __init__(self) -> None:
        self.layer_idx: int = 0
        self.seq_len: int = 0
        self.saved: dict[str, Any] = {}
        self.model: Any = None
        """The full model (e.g. ``LlamaForCausalLM``).  Set by the
        dispatcher.  Useful for accessing ``lm_head``, layer norm, etc."""

    def get_parameter(self, name: str) -> torch.Tensor:
        """Get a model parameter by name, auto-gathering across TP ranks.

        With pipeline parallelism, the parameter must be on this PP
        stage.  In practice, hook a layer on the same PP stage as the
        parameter (e.g. hook the last layer to access ``lm_head.weight``).

        Example::

            weight = ctx.get_parameter("lm_head.weight")
            logits = hidden_states @ weight.T
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_tp_group
        from vllm.model_executor.models.utils import PPMissingLayer

        obj: Any = self.model
        parts = name.split(".")
        for attr in parts:
            obj = getattr(obj, attr)
            if isinstance(obj, PPMissingLayer):
                raise AttributeError(
                    f"Parameter {name!r} is not on this pipeline-parallel "
                    f"stage. Hook a layer on the same PP stage as the "
                    f"target parameter (e.g. last layer for lm_head)."
                )

        param = torch.as_tensor(obj)

        # TP gather if sharded.
        tp_group = get_tp_group()
        module: Any = self.model
        for attr in parts[:-1]:
            module = getattr(module, attr)
        tp_size = getattr(module, "tp_size", 1)
        if tp_size > 1:
            gathered = [torch.empty_like(param) for _ in range(tp_size)]
            dist.all_gather(gathered, param, group=tp_group.device_group)
            gather_dim = getattr(module, "gather_dim", 0)
            param = torch.cat(gathered, dim=gather_dim)

        return param


class Hook(BaseModel):
    """A user-defined hook that runs on specified layers during inference.

    The callable ``fn`` receives a :class:`HookContext` and the per-request
    hidden-states slice (shape ``(seq_len, hidden_dim)``).  Return a tensor
    to modify hidden states, or ``None`` to leave them unchanged.

    ``fn`` must be deterministic across TP ranks — non-deterministic ops
    (e.g. ``torch.randn``) will cause divergence.

    Example::

        def ablate_neuron(ctx, h):
            ctx.saved[f"pre_L{ctx.layer_idx}"] = h[:, 42].cpu()
            h = h.clone()
            h[:, 42] = 0
            return h

        hook = Hook(fn=ablate_neuron, layer_indices=[15, 16])

    For HTTP transport, ``fn`` is serialized via cloudpickle.  This means
    **arbitrary code execution** on the server — only use with trusted clients.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fn: Callable[[HookContext, torch.Tensor], torch.Tensor | None]
    """Hook function.  Signature: ``(ctx, hidden_states) -> Tensor | None``."""

    layer_indices: list[int]
    """Which model layers this hook runs on."""

    pre: bool = False
    """If True, run as a pre-hook (before the layer forward pass) instead of
    a post-hook (after).  Pre-hooks receive the layer's input hidden states
    and can modify them before the layer processes them.  Useful for
    corrupting embeddings or patching inputs to specific layers."""

    @field_validator("fn", mode="before")
    @classmethod
    def _deserialize_fn(cls, v: Any) -> Callable:
        """Accept cloudpickle base64 dicts (from JSON transport) or raw callables."""
        if isinstance(v, dict) and "cloudpickle" in v:
            return cloudpickle.loads(base64.b64decode(v["cloudpickle"]))
        if isinstance(v, (bytes, bytearray)):
            return cloudpickle.loads(v)
        if callable(v):
            return v
        raise ValueError(f"fn must be a callable or a cloudpickle dict, got {type(v)}")

    @field_serializer("fn")
    def _serialize_fn(self, v: Callable, _info: Any) -> dict[str, str]:
        """Serialize callable to cloudpickle base64 dict for JSON transport."""
        return {"cloudpickle": base64.b64encode(cloudpickle.dumps(v)).decode("ascii")}

    @model_validator(mode="after")
    def _check_layers(self) -> Self:
        """Validate layer_indices is non-empty and cache as a set."""
        if not self.layer_indices:
            raise ValueError("layer_indices must be non-empty")
        # Cache as frozenset for O(1) membership tests on the hot path.
        object.__setattr__(self, "_layer_set", frozenset(self.layer_indices))
        return self

    def has_layer(self, layer_idx: int) -> bool:
        """O(1) layer membership test (uses cached frozenset)."""
        return layer_idx in self._layer_set  # type: ignore[reportAttributeAccessIssue]
