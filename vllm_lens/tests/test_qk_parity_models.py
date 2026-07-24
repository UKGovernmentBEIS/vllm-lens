"""Opt-in multi-architecture parity sweep for attention Q/K capture.

Compares reconstructed attention patterns (``vllm_lens.attention``)
against HuggingFace eager attention (``output_attentions=True``) across
architectures that exercise every reconstruction code path: GQA/MHA,
non-RoPE positions, alternative module layouts, logit soft-capping,
sliding windows, custom scales, attention sinks, hybrid
linear-attention models, and the MLA refusal.

Deliberately opt-in — it downloads several models (some gated: needs an
HF token), loads each into both vLLM and HF, and wants a large GPU
(gpt-oss-20b needs ~45 GB for the HF side alone):

    VLLM_LENS_QK_PARITY=1 pytest vllm_lens/tests/test_qk_parity_models.py -v

Engines are created strictly sequentially (one per test): concurrent
vLLM engine boots on one host can die silently during attention-kernel
JIT initialization.
"""

import gc
import os

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens.attention import attention_patterns

pytestmark = pytest.mark.skipif(
    not os.environ.get("VLLM_LENS_QK_PARITY"),
    reason="opt-in sweep: set VLLM_LENS_QK_PARITY=1 "
    "(downloads several models; needs a large GPU and an HF token)",
)

PROMPT = (
    "The quick brown fox jumps over the lazy dog. In a distant kingdom, an "
    "old librarian catalogued forbidden books about astronomy, medicine, and "
    "the forgotten art of navigation across the winter sea."
)

# (model, engine kwargs, env overrides)
MODELS = [
    pytest.param("Qwen/Qwen2.5-0.5B-Instruct", {}, {}, id="qwen2.5-gqa"),
    pytest.param("unsloth/Llama-3.2-1B-Instruct", {}, {}, id="llama3.2-gqa"),
    pytest.param("Qwen/Qwen3-0.6B", {}, {}, id="qwen3-qk-norm"),
    pytest.param("gpt2", {"max_model_len": 1024}, {}, id="gpt2-mha-learned-positions"),
    pytest.param("facebook/opt-125m", {}, {}, id="opt-decoder-layout"),
    pytest.param(
        "google/gemma-2-2b-it",
        {},
        # FlashAttention-3 crashes at init on gemma-2's logit soft-cap.
        {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"},
        id="gemma2-softcap-sliding-window",
    ),
    pytest.param(
        "ibm-granite/granite-3.3-2b-instruct",
        {},
        {},
        id="granite-attention-multiplier-scale",
    ),
    pytest.param("openai/gpt-oss-20b", {}, {}, id="gpt-oss-sinks-sliding-window"),
    pytest.param(
        "zai-org/GLM-4-9B-0414",
        {},
        {},
        id="glm4-extreme-gqa-partial-rope",
    ),
]


def _capture_all_layers(model_name: str, engine_kwargs: dict) -> tuple[dict, int]:
    """Run one prompt through a fresh engine with output_qk=True."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    ids = tok(PROMPT).input_ids

    llm = LLM(
        model=model_name,
        dtype="auto",
        gpu_memory_utilization=0.5,
        **{"max_model_len": 2048, **engine_kwargs},
    )
    try:
        sp = SamplingParams(
            temperature=0.0, max_tokens=1, extra_args={"output_qk": True}
        )
        out = llm.generate([{"prompt_token_ids": ids}], sp)[0]
        acts = out.activations  # type: ignore[attr-defined]
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    assert acts is not None and "attn_q" in acts
    return acts, len(ids)


def _hf_attentions(model_name: str, layers: list[int], n: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    ids = tok(PROMPT).input_ids
    assert len(ids) == n
    hf = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    try:
        with torch.no_grad():
            out = hf(
                torch.tensor([ids], device="cuda"),
                output_attentions=True,
                use_cache=False,
            )
        return {layer: out.attentions[layer][0].float().cpu() for layer in layers}
    finally:
        del hf
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize(("model_name", "engine_kwargs", "env"), MODELS)
def test_parity_with_hf(model_name, engine_kwargs, env, monkeypatch):
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    acts, n = _capture_all_layers(model_name, engine_kwargs)
    layers = list(acts["qk_layers"])
    pick = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
    hf_by_layer = _hf_attentions(model_name, pick, n)

    for layer in pick:
        got = attention_patterns(acts, layer)[:, :n, :n]
        want = hf_by_layer[layer]
        assert got.shape == want.shape, f"L{layer}: {got.shape} vs {want.shape}"

        mean_abs_diff = (got - want).abs().mean().item()
        assert mean_abs_diff < 1e-2, f"L{layer} mean abs diff {mean_abs_diff:.6f}"

        # Tie-tolerant argmax agreement: near-uniform rows (common in
        # layer 0) flip argmax on bf16 noise, so a row agrees if our
        # pick's HF probability is within 5e-3 of HF's row max.
        hf_at_pick = want.gather(-1, got.argmax(-1).unsqueeze(-1)).squeeze(-1)
        agreement = ((want.max(-1).values - hf_at_pick) < 5e-3).float().mean().item()
        assert agreement > 0.9, f"L{layer} row-argmax agreement {agreement:.2%}"


def test_hybrid_model_captures_attention_layers_only():
    """Qwen3.5 hybrid: only the full-attention layers register/capture."""
    from transformers import AutoConfig

    model_name = "Qwen/Qwen3.5-35B-A3B"
    acts, _ = _capture_all_layers(model_name, {"gpu_memory_utilization": 0.85})
    layers = list(acts["qk_layers"])
    total = AutoConfig.from_pretrained(model_name).num_hidden_layers
    # A strict subset of decoder layers, with valid global indices.
    assert 0 < len(layers) < total
    assert all(0 <= layer < total for layer in layers)
    # Reconstructions are valid (sub-)distributions on the last layer.
    weights = attention_patterns(acts, layers[-1])
    assert not weights.isnan().any()
    assert (weights.sum(-1) <= 1.0 + 1e-4).all()


def test_mla_model_refused_with_clear_error():
    """DeepSeek MLA: residual capture works, output_qk raises loudly."""
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite-Chat",
        dtype="auto",
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
    )
    try:
        sp_rs = SamplingParams(
            temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [2]}
        )
        out = llm.generate(["Hello world"], sp_rs)[0]
        assert "residual_stream" in out.activations  # type: ignore[attr-defined]

        sp_qk = SamplingParams(
            temperature=0.0, max_tokens=1, extra_args={"output_qk": True}
        )
        with pytest.raises(Exception, match="MLA"):
            llm.generate(["Hello world"], sp_qk)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
