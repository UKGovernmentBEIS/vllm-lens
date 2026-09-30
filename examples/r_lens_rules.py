"""LRP backward rules for fitting an R-lens (``jacobian_lens_fit.py --rules lrp``).

An R-lens is a J-lens fit with Layer-wise Relevance Propagation (LRP) rules
installed in the backward pass, which makes the readout markedly more faithful
on early layers (see "R-lens: making J-lens more faithful on early layers",
https://www.greaterwrong.com/posts/nv8oedrnLXKRzNEL9). The rules are pure
stop-gradients: every patched forward returns the original module's value
*exactly* (the value is computed by the module's own kernels; a surrogate
carries only the backward), so the fit loop, the ``.pt`` format and the
readout side need no changes — only the quantity transported by
``torch.autograd.backward`` differs.

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


def _exact_value(true_value: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """Return exactly ``true_value`` in the forward, ``surrogate``'s backward.

    ``surrogate - surrogate.detach()`` is exactly zero elementwise in the
    forward (for finite values), so the result equals ``true_value`` while
    gradients flow only through ``surrogate``.
    """
    return true_value.detach() + (surrogate - surrogate.detach())


# Structural parity check tolerance (relative to the output's max magnitude).
# Both sides are computed with the module's own kernels, so genuine matches are
# at or near bitwise; anything past bf16 rounding means the module computes a
# different function from the one the surrogate linearizes.
_PARITY_RTOL = 1e-2


def _check_same_function(
    module: nn.Module, ours: torch.Tensor, ref: torch.Tensor, what: str
) -> None:
    """Once per instance: assert the surrogate's structure matches the module.

    ``_exact_value`` guarantees the *forward* is the original module's value
    regardless, but the *backward* is the surrogate's — so if the module's
    real forward is not the function the surrogate re-expresses (an extra
    scale, a ``(1 + weight)`` norm, a bias path, ...), the fit would run with
    silently wrong gradients. Compare the surrogate-structured value against
    the module's own forward on the first call and fail loudly on mismatch.
    """
    if getattr(module, "_lrp_parity_checked", False):
        return
    ours32, ref32 = ours.detach().float(), ref.detach().float()
    scale = ref32.abs().max().clamp_min(1e-6)
    err = (ours32 - ref32).abs().max() / scale
    if not torch.isfinite(err) or err > _PARITY_RTOL:
        raise RuntimeError(
            f"R-lens: {type(module).__name__}.{what} does not match the structure "
            f"the LRP surrogate assumes (max rel. diff {err.item():.3e} > "
            f"{_PARITY_RTOL}). The module computes a different function from "
            "the one the rules linearize, so its backward would be wrong. "
            "Add explicit support for this module type in r_lens_rules.py."
        )
    module._lrp_parity_checked = True


def lrp_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """RMSNorm forward with the LN-rule installed (LRP).

    The value is the module's original forward, exactly (whatever kernel it
    uses — reference path, quack, hub); the backward comes from a
    reference-path surrogate with one ``.detach()`` on the ``rsqrt`` factor,
    treating the normalization denominator as a constant.
    """
    input_dtype = hidden_states.dtype
    hs = hidden_states.to(torch.float32)
    variance = hs.pow(2).mean(-1, keepdim=True)
    hs = hs * torch.rsqrt(variance + self.variance_epsilon).detach()
    surrogate = self.weight * hs.to(input_dtype)
    with torch.no_grad():
        true_value = self._lrp_orig_forward(hidden_states)
    _check_same_function(self, surrogate, true_value, "forward")
    return _exact_value(true_value, surrogate)


def _classify_gate_act(act_fn) -> str:
    """Map a gate activation module to an identity-rule kind.

    Only activations that factor exactly as ``g * m(g)`` with ``m`` smooth
    everywhere are supported — the identity rule then needs no division and no
    near-zero guard. Anything else raises at install time, before any compute.
    The classification selects the *backward* multiplier only; the forward
    value always comes from the module's own ``gate_act_fn``.
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

    The value of ``act(gate) * up`` comes from the module's own activation
    kernel, exactly; the surrogate applies the identity rule (linearized
    activation) and the half rule (each branch gets half of the ordinary
    product gradient) in the backward only.
    """
    g = self.gate_proj(x)
    u = self.up_proj(x)
    a = _identity_rule_act(self._lrp_act_kind, g)
    surrogate = 0.5 * (a * u.detach() + a.detach() * u)
    with torch.no_grad():
        true_h = self.gate_act_fn(g) * u
    out = self.down_proj(_exact_value(true_h, surrogate))
    if not getattr(self, "_lrp_parity_checked", False):
        with torch.no_grad():
            ref = self._lrp_orig_forward(x)
        _check_same_function(self, out, ref, "forward")
    return out


# HF config keys that mark a mixture-of-experts architecture (transformers'
# naming varies: Mixtral/Qwen-MoE ``num_local_experts`` / ``num_experts``,
# DeepSeek/GLM ``n_routed_experts``).
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def check_config_supports_lrp(config) -> None:
    """Fail fast from the HF config, *before* any weights are materialized.

    ``install_lrp_rules`` also rejects MoE layers, but it runs after
    ``setup_model`` — for a 100B+ MoE that is 25+ minutes of loading and
    weight conversion before the error appears.  Call this on
    ``AutoConfig.from_pretrained(model)`` right after argument parsing.
    Accepts a config object or a plain dict (nested ``text_config`` is
    checked too, for multimodal wrappers).
    """
    cfgs = [config]
    text = getattr(config, "text_config", None)
    if text is None and isinstance(config, dict):
        text = config.get("text_config")
    if text is not None:
        cfgs.append(text)
    for cfg in cfgs:
        for key in _MOE_CONFIG_KEYS:
            n = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)
            if n is not None and int(n) > 1:
                arch = (
                    cfg.get("architectures")
                    if isinstance(cfg, dict)
                    else getattr(cfg, "architectures", None)
                ) or ["?"]
                raise ValueError(
                    f"{arch[0]} is a mixture-of-experts model ({key}={n}) — "
                    "MoE models are not supported by the R-lens fit yet "
                    "(dense gated MLPs only). Use --rules gradient, or a dense "
                    "checkpoint."
                )


def _check_unpatched(module: nn.Module) -> None:
    if "forward" in module.__dict__:
        raise RuntimeError(
            f"LRP rules already installed on {type(module).__name__} "
            "(instance forward is already overridden)"
        )


def install_lrp_rules(decoder_layers) -> int:
    """Install LRP rules on each decoder layer's residual-stream norms and MLP.

    Patches ``input_layernorm`` / ``post_attention_layernorm`` (LN-rule) and
    ``mlp`` (identity + half rules) per instance; q/k norms inside
    ``self_attn`` and everything outside ``decoder_layers`` are untouched.
    Forward values are unchanged. Returns the number of modules patched.

    Every target is validated before anything is bound, so a failure (MoE
    layer, unsupported activation, double install) leaves the model unpatched.
    Structure is verified on each module's *first forward* (parameters may not
    be materialized at install time under FSDP): the surrogate's value must
    match the module's own forward, else ``RuntimeError`` — see
    :func:`_check_same_function`.
    """
    norms: list[nn.Module] = []
    mlps: list[tuple[nn.Module, str]] = []
    for i, layer in enumerate(decoder_layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name, None)
            if norm is None or any(not hasattr(norm, a) for a in _RMSNORM_ATTRS):
                raise ValueError(
                    f"layer {i}: {name} ({type(norm).__name__}) does not look "
                    f"like an RMSNorm (needs {_RMSNORM_ATTRS})"
                )
            _check_unpatched(norm)
            norms.append(norm)
        mlp = getattr(layer, "mlp", None)
        if mlp is None or any(not hasattr(mlp, a) for a in _MLP_ATTRS):
            raise ValueError(
                f"layer {i}: mlp ({type(mlp).__name__}) is not a dense gated "
                f"MLP (needs {_MLP_ATTRS}) — MoE models are not supported by "
                "the R-lens fit yet"
            )
        _check_unpatched(mlp)
        mlps.append((mlp, _classify_gate_act(mlp.gate_act_fn)))
    for norm in norms:
        norm._lrp_orig_forward = norm.forward
        norm.forward = types.MethodType(lrp_rmsnorm_forward, norm)
    for mlp, kind in mlps:
        mlp._lrp_act_kind = kind
        mlp._lrp_orig_forward = mlp.forward
        mlp.forward = types.MethodType(lrp_gated_mlp_forward, mlp)
    return len(norms) + len(mlps)
