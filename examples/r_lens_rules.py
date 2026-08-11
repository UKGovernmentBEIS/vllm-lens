"""LRP backward rules for fitting an R-lens (``jacobian_lens_fit.py --rules lrp``).

An R-lens is a J-lens fit with Layer-wise Relevance Propagation (LRP) rules
installed in the backward pass, which makes the readout markedly more faithful
on early layers (see "R-lens: making J-lens more faithful on early layers",
https://www.greaterwrong.com/posts/nv8oedrnLXKRzNEL9). The rules are pure
stop-gradients: every patched forward is numerically unchanged, so the fit
loop, the ``.pt`` format and the readout side need no changes — only the
quantity transported by ``torch.autograd.backward`` differs.

Dense-model recipe (the only one implemented; MoE fails fast):

- LN-rule on residual-stream RMSNorms: treat the normalization denominator as
  a constant, i.e. detach the ``rsqrt`` factor.
- Identity-rule on the gated MLP's activation: detach the nonlinear factor of
  ``act(g) = g * m(g)`` so the backward is a per-element linear map ``m(g)``.
- Half-rule on the multiplicative gate: split relevance 50/50 across the two
  branches of ``act(gate) * up`` instead of double-counting through the
  product.

Attention, q/k norms and all linear layers keep ordinary gradients (the LRP
0-rule for a linear map *is* the gradient), so nothing else is patched.

This module is dependency-free (torch + stdlib) and duck-types against
prime-rl's custom-impl decoder layers: RMSNorms expose ``weight`` /
``variance_epsilon``, gated MLPs expose ``gate_proj`` / ``up_proj`` /
``down_proj`` / ``gate_act_fn``. Rules are bound per *instance* (never on the
class): decoder layers also hold q/k RMSNorms inside ``self_attn`` that must
keep the true gradient.
"""

import math
import types

import torch
from torch import nn

_RMSNORM_ATTRS = ("weight", "variance_epsilon")
_MLP_ATTRS = ("gate_proj", "up_proj", "down_proj", "gate_act_fn")

# transformers' ACT2FN entries are its own activation classes; match by name so
# the dispatch survives transformers versions (isinstance covers plain torch).
_SILU_CLASS_NAMES = {"SiLU", "SiLUActivation"}
_GELU_EXACT_CLASS_NAMES = {"GELU", "GELUActivation"}
_GELU_TANH_CLASS_NAMES = {"PytorchGELUTanh", "NewGELUActivation", "GELUTanh"}


def lrp_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """RMSNorm forward with the LN-rule installed (LRP).

    Identical op sequence to prime-rl's reference RMSNorm forward — fp32
    compute, cast back before the weight multiply — with one ``.detach()`` on
    the ``rsqrt`` factor, so the output is bitwise-equal and only the backward
    changes: the normalization denominator is treated as a constant.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = (
        hidden_states * torch.rsqrt(variance + self.variance_epsilon).detach()
    )
    return self.weight * hidden_states.to(input_dtype)


def _classify_gate_act(act_fn) -> str:
    """Map a gate activation module to an identity-rule kind.

    Only activations that factor exactly as ``g * m(g)`` with ``m`` smooth
    everywhere are supported — the identity rule then needs no division and no
    near-zero guard. Anything else raises at install time, before any compute.
    """
    name = type(act_fn).__name__
    if isinstance(act_fn, nn.SiLU) or name in _SILU_CLASS_NAMES:
        return "silu"
    if isinstance(act_fn, nn.GELU):
        return "gelu_tanh" if act_fn.approximate == "tanh" else "gelu_exact"
    if name in _GELU_EXACT_CLASS_NAMES:
        return "gelu_exact"
    if name in _GELU_TANH_CLASS_NAMES:
        return "gelu_tanh"
    raise NotImplementedError(
        f"R-lens identity rule: unsupported gate activation {name!r} "
        "(supported: SiLU, GELU, GELU-tanh)"
    )


def _identity_rule_act(kind: str, g: torch.Tensor) -> torch.Tensor:
    """``act(g)`` with its nonlinear factor detached (LRP identity rule).

    Each supported activation is written as ``g * m(g)`` and ``m(g)`` is
    detached, so the forward value is unchanged while the backward becomes the
    per-element linear map ``m(g)`` (e.g. ``sigmoid(g)`` for SiLU instead of
    the full SiLU derivative).
    """
    if kind == "silu":
        return g * torch.sigmoid(g).detach()
    if kind == "gelu_exact":
        return g * (0.5 * (1.0 + torch.erf(g / math.sqrt(2.0)))).detach()
    if kind == "gelu_tanh":
        c = math.sqrt(2.0 / math.pi)
        inner = c * (g + 0.044715 * g.pow(3))
        return g * (0.5 * (1.0 + torch.tanh(inner))).detach()
    raise NotImplementedError(f"unknown identity-rule kind {kind!r}")


def lrp_gated_mlp_forward(self, x: torch.Tensor, routed_experts=None) -> torch.Tensor:
    """Gated-MLP forward with identity + half rules installed (LRP).

    ``0.5 * (a * b.detach() + a.detach() * b)`` is bitwise ``a * b`` in the
    forward while each branch receives half of the ordinary product gradient.
    """
    g = self.gate_proj(x)
    u = self.up_proj(x)
    a = _identity_rule_act(self._lrp_act_kind, g)
    h = 0.5 * (a * u.detach() + a.detach() * u)
    return self.down_proj(h)


def _bind(module: nn.Module, fn) -> None:
    if "forward" in module.__dict__:
        raise RuntimeError(
            f"LRP rules already installed on {type(module).__name__} "
            "(instance forward is already overridden)"
        )
    module.forward = types.MethodType(fn, module)


def install_lrp_rules(decoder_layers) -> int:
    """Install LRP rules on each decoder layer's residual-stream norms and MLP.

    Patches ``input_layernorm`` / ``post_attention_layernorm`` (LN-rule) and
    ``mlp`` (identity + half rules) per instance; q/k norms inside
    ``self_attn`` and everything outside ``decoder_layers`` are untouched.
    Forward values are unchanged. Returns the number of modules patched.
    """
    n_patched = 0
    for i, layer in enumerate(decoder_layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name, None)
            if norm is None or any(not hasattr(norm, a) for a in _RMSNORM_ATTRS):
                raise ValueError(
                    f"layer {i}: {name} ({type(norm).__name__}) does not look "
                    f"like an RMSNorm (needs {_RMSNORM_ATTRS})"
                )
            _bind(norm, lrp_rmsnorm_forward)
            n_patched += 1
        mlp = getattr(layer, "mlp", None)
        if mlp is None or any(not hasattr(mlp, a) for a in _MLP_ATTRS):
            raise ValueError(
                f"layer {i}: mlp ({type(mlp).__name__}) is not a dense gated "
                f"MLP (needs {_MLP_ATTRS}) — MoE models are not supported by "
                "the R-lens fit yet"
            )
        mlp._lrp_act_kind = _classify_gate_act(mlp.gate_act_fn)
        _bind(mlp, lrp_gated_mlp_forward)
        n_patched += 1
    return n_patched
