"""CPU-only unit tests for the R-lens LRP rules (``examples/r_lens_rules.py``).

The stand-in modules replicate the attribute surface of prime-rl's custom-impl
decoder layers (RMSNorm: ``weight``/``variance_epsilon``; gated MLP:
``gate_proj``/``up_proj``/``down_proj``/``gate_act_fn``), with the RMSNorm
reference forward verbatim, so what is asserted here is exactly what the fit
env will execute.
"""

import copy
import math

import pytest
import torch
from torch import nn

from r_lens_rules import (
    _identity_rule_act,
    install_lrp_rules,
    lrp_gated_mlp_forward,
    lrp_rmsnorm_forward,
)

D = 8
D_FF = 16
EPS = 1e-6


class _RMSNorm(nn.Module):
    """prime-rl's reference RMSNorm forward, verbatim."""

    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _MLP(nn.Module):
    def __init__(self, act=None):
        super().__init__()
        self.gate_proj = nn.Linear(D, D_FF, bias=False)
        self.up_proj = nn.Linear(D, D_FF, bias=False)
        self.down_proj = nn.Linear(D_FF, D, bias=False)
        self.gate_act_fn = act if act is not None else nn.SiLU()

    def forward(self, x, routed_experts=None):
        return self.down_proj(self.gate_act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = _RMSNorm(D)
        self.k_norm = _RMSNorm(D)


class _DecoderLayer(nn.Module):
    def __init__(self, act=None):
        super().__init__()
        self.input_layernorm = _RMSNorm(D)
        self.self_attn = _Attn()
        self.post_attention_layernorm = _RMSNorm(D)
        self.mlp = _MLP(act)

    def forward(self, x):
        # A decoder block with attention stood in by the normed residual
        # branch — keeps input_layernorm in the differentiated graph.
        h = x + self.input_layernorm(x)
        return h + self.mlp(self.post_attention_layernorm(h))


def _randomize(module: nn.Module) -> nn.Module:
    """Randomize all params (unit norm weights would mask placement bugs)."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p))
    return module


def _patched_norm(norm: _RMSNorm) -> _RMSNorm:
    norm = copy.deepcopy(norm)
    norm._lrp_orig_forward = norm.forward
    norm.forward = lrp_rmsnorm_forward.__get__(norm)
    return norm


def _patched_mlp(mlp: _MLP, kind: str) -> _MLP:
    mlp = copy.deepcopy(mlp)
    mlp._lrp_act_kind = kind
    mlp.forward = lrp_gated_mlp_forward.__get__(mlp)
    return mlp


ACTS = {
    "silu": (
        nn.SiLU(),
        torch.sigmoid,  # the detached multiplier m(g) in act(g) = g * m(g)
    ),
    "gelu_exact": (
        nn.GELU(),
        lambda g: 0.5 * (1.0 + torch.erf(g / math.sqrt(2.0))),
    ),
    "gelu_tanh": (
        nn.GELU(approximate="tanh"),
        lambda g: (
            0.5
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (g + 0.044715 * g.pow(3))))
        ),
    ),
}


# --- (a) forward parity -------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_forward_bitwise_equal(dtype):
    torch.manual_seed(0)
    norm = _randomize(_RMSNorm(D))
    x = torch.randn(3, 5, D).to(dtype)
    assert torch.equal(_patched_norm(norm)(x), norm(x))


@pytest.mark.parametrize("kind", list(ACTS))
def test_mlp_forward_matches(kind):
    # The value is computed by the module's own gate_act_fn (straight-through
    # construction), so the patched forward is exactly the unpatched one.
    torch.manual_seed(1)
    mlp = _randomize(_MLP(ACTS[kind][0]))
    x = torch.randn(3, 5, D)
    assert torch.equal(_patched_mlp(mlp, kind)(x), mlp(x))


def test_no_grad_input_flows_through():
    # Layers at or below the fit's earliest source layer see grad-less inputs
    # (and the fit freezes every parameter before the forward).
    torch.manual_seed(2)
    layer = _DecoderLayer().requires_grad_(False)
    install_lrp_rules([layer])
    x = torch.randn(2, D)
    y = layer.mlp(layer.post_attention_layernorm(x))
    assert not y.requires_grad


# --- (b) RMSNorm LN-rule backward ---------------------------------------------


def test_rmsnorm_ln_rule_backward():
    torch.manual_seed(3)
    norm = _randomize(_RMSNorm(D))
    x0 = torch.randn(4, D)
    v = torch.randn(4, D)
    w = norm.weight.detach()
    r = torch.rsqrt(x0.pow(2).mean(-1, keepdim=True) + EPS)

    def grad_of(module):
        x = x0.clone().requires_grad_(True)
        module(x).backward(v)
        return x.grad

    true_grad = grad_of(norm)
    lrp_grad = grad_of(_patched_norm(norm))
    # LN-rule: denominator constant => elementwise v * w * r.
    torch.testing.assert_close(lrp_grad, v * w * r)
    # The analytic true gradient has the extra mean-projection term...
    expected_true = v * w * r - x0 * r.pow(3) * (v * w * x0).mean(-1, keepdim=True)
    torch.testing.assert_close(true_grad, expected_true)
    # ...which is generically nonzero, so the two backward rules differ.
    assert not torch.allclose(lrp_grad, true_grad)


# --- (c) gated-MLP identity + half rule backward ------------------------------


@pytest.mark.parametrize("kind", list(ACTS))
def test_mlp_identity_half_rule_backward(kind):
    torch.manual_seed(4)
    act, m = ACTS[kind]
    mlp = _randomize(_MLP(act))
    x0 = torch.randn(4, D)
    v = torch.randn(4, D)
    wg, wu, wd = (
        mlp.gate_proj.weight.detach(),
        mlp.up_proj.weight.detach(),
        mlp.down_proj.weight.detach(),
    )
    g, u, vd = x0 @ wg.T, x0 @ wu.T, v @ wd

    def grad_of(module):
        x = x0.clone().requires_grad_(True)
        module(x).backward(v)
        return x.grad

    lrp_grad = grad_of(_patched_mlp(mlp, kind))
    # Identity rule: act backward is m(g); half rule: 0.5 per gate branch.
    expected = 0.5 * (vd * u * m(g)) @ wg + 0.5 * (vd * (g * m(g))) @ wu
    torch.testing.assert_close(lrp_grad, expected)
    assert not torch.allclose(lrp_grad, grad_of(mlp))


@pytest.mark.parametrize("kind", list(ACTS))
def test_identity_rule_finite_at_zero(kind):
    g = torch.zeros(3, requires_grad=True)
    out = _identity_rule_act(kind, g)
    assert torch.equal(out, torch.zeros(3))
    out.sum().backward()
    assert torch.isfinite(g.grad).all()


# --- (d) end-to-end replica of the fit loop -----------------------------------


def test_fit_loop_replica_matches_closed_form():
    torch.manual_seed(5)
    layer = _DecoderLayer()
    for mod in (layer.post_attention_layernorm, layer.mlp):
        _randomize(mod)
    install_lrp_rules([layer])
    norm, mlp = layer.post_attention_layernorm, layer.mlp

    def chain(x):  # the post-attention half of a decoder block
        return x + mlp(norm(x))

    x0 = torch.randn(D)
    dim_batch = 3  # < D so several backwards reuse the retained graph
    xb = x0.repeat(dim_batch, 1).requires_grad_(True)
    with torch.enable_grad():
        y = chain(xb)
    lens = torch.zeros(D, D)
    b = torch.arange(dim_batch)
    starts = list(range(0, D, dim_batch))
    for bi, start in enumerate(starts):
        n = min(dim_batch, D - start)
        xb.grad = None
        cot = torch.zeros_like(y)
        cot[b[:n], start + b[:n]] = 1.0
        torch.autograd.backward(
            y, grad_tensors=cot, retain_graph=(bi < len(starts) - 1), inputs=[xb]
        )
        lens[start : start + n] = xb.grad[:n]

    # Closed form: R = I + J_mlp @ J_norm with the LRP rules installed.
    w = norm.weight.detach()
    r = torch.rsqrt(x0.pow(2).mean() + EPS)
    xn = w * x0 * r
    wg, wu, wd = (
        mlp.gate_proj.weight.detach(),
        mlp.up_proj.weight.detach(),
        mlp.down_proj.weight.detach(),
    )
    g, u = wg @ xn, wu @ xn
    j_mlp = wd @ (
        0.5 * torch.diag(u * torch.sigmoid(g)) @ wg
        + 0.5 * torch.diag(g * torch.sigmoid(g)) @ wu
    )
    j_norm = r * torch.diag(w)
    torch.testing.assert_close(lens, torch.eye(D) + j_mlp @ j_norm)

    # And it differs from the true Jacobian of the unpatched chain.
    plain = _DecoderLayer()
    plain.load_state_dict(layer.state_dict())
    true_jac = torch.autograd.functional.jacobian(
        lambda x: x + plain.mlp(plain.post_attention_layernorm(x)), x0
    )
    assert not torch.allclose(lens, true_jac)


def test_fit_style_multi_source_hook_rooted():
    """Full replica of the fitter's capture pattern on a 3-layer stack.

    Frozen params, grad-less input, graph rooted by a forward hook calling
    ``requires_grad_(True)`` on the earliest source's output, two intermediate
    sources harvested per backward with ``inputs=srcs`` and retained-graph
    reuse — exactly ``jacobian_lens_fit.py``'s loop. Verified against
    ``torch.autograd.functional.jacobian`` of the same patched blocks.
    """
    torch.manual_seed(7)
    layers = nn.ModuleList([_randomize(_DecoderLayer()) for _ in range(3)])
    layers.requires_grad_(False)
    install_lrp_rules(layers)

    acts = {}

    def make_hook(idx):
        def hook(_m, _in, out):
            if idx == 0 and not out.requires_grad:
                out.requires_grad_(True)  # root the graph at the earliest source
            acts[idx] = out

        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(3)]
    dim_batch = 3
    x = torch.randn(D).repeat(dim_batch, 1)  # identical rows, requires_grad=False
    with torch.enable_grad():
        y = layers[2](layers[1](layers[0](x)))
    for h in handles:
        h.remove()
    target, srcs = acts[2], [acts[0], acts[1]]
    assert y.requires_grad and not x.requires_grad
    for s in srcs:
        s.retain_grad()
    lens = {0: torch.zeros(D, D), 1: torch.zeros(D, D)}
    b = torch.arange(dim_batch)
    starts = list(range(0, D, dim_batch))
    for bi, start in enumerate(starts):
        n = min(dim_batch, D - start)
        for s in srcs:
            s.grad = None
        cot = torch.zeros_like(target)
        cot[b[:n], start + b[:n]] = 1.0
        torch.autograd.backward(
            target, grad_tensors=cot, retain_graph=(bi < len(starts) - 1), inputs=srcs
        )
        for lyr, s in zip((0, 1), srcs):
            lens[lyr][start : start + n] = s.grad[:n]

    # Reference: independent autograd through the same patched blocks.
    out0, out1 = acts[0][0].detach(), acts[1][0].detach()
    j1 = torch.autograd.functional.jacobian(layers[1], out0)
    j2 = torch.autograd.functional.jacobian(layers[2], out1)
    torch.testing.assert_close(lens[1], j2)
    torch.testing.assert_close(lens[0], j2 @ j1)


# --- (e) install scope and failure modes --------------------------------------


def test_install_scope_and_count():
    layers = [_DecoderLayer(), _DecoderLayer()]
    outside_norm = _RMSNorm(D)  # stands in for the final model.norm
    assert install_lrp_rules(layers) == 6
    for layer in layers:
        for mod in (layer.input_layernorm, layer.post_attention_layernorm, layer.mlp):
            assert "forward" in mod.__dict__
        for mod in (layer.self_attn.q_norm, layer.self_attn.k_norm):
            assert "forward" not in mod.__dict__
    assert "forward" not in outside_norm.__dict__
    with pytest.raises(RuntimeError, match="already installed"):
        install_lrp_rules(layers)


def test_install_rejects_moe_mlp():
    class _MoEStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([_MLP()])

    layer = _DecoderLayer()
    layer.mlp = _MoEStub()
    with pytest.raises(ValueError, match="MoE"):
        install_lrp_rules([layer])


def test_install_rejects_unknown_activation():
    layer = _DecoderLayer(act=nn.Tanh())
    with pytest.raises(NotImplementedError, match="Tanh"):
        install_lrp_rules([layer])


def test_failed_install_leaves_model_unpatched():
    # Validation runs over every layer before anything is bound, so a bad
    # layer anywhere leaves the whole model untouched.
    good, bad = _DecoderLayer(), _DecoderLayer(act=nn.Tanh())
    with pytest.raises(NotImplementedError):
        install_lrp_rules([good, bad])
    for layer in (good, bad):
        for mod in (layer.input_layernorm, layer.post_attention_layernorm, layer.mlp):
            assert "forward" not in mod.__dict__


def test_forward_hooks_still_fire_on_patched_module():
    # The fit's activation capture is a forward hook; it must survive patching.
    torch.manual_seed(6)
    layer = _DecoderLayer()
    install_lrp_rules([layer])
    seen = []
    layer.mlp.register_forward_hook(lambda m, i, o: seen.append(o))
    out = layer.mlp(torch.randn(2, D))
    assert len(seen) == 1 and torch.equal(seen[0], out)
