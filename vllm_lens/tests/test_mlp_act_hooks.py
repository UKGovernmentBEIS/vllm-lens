"""GPU tests for ``Hook(site="mlp_act")`` — per-neuron MLP activation hooks.

The hooked tensor is the output of ``layer.mlp.act_fn`` (``silu(gate) * up``),
i.e. the input to ``down_proj``.  Checked against HF in fp32, plus the closed
form of a single-neuron zero ablation on the layer's residual output.
"""

import gc

import pytest
import torch
from vllm import LLM, RequestOutput, SamplingParams

from vllm_lens import Hook

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT, PROMPTS, REFERENCE_DTYPE

NEURON = 123


@pytest.fixture(scope="module")
def llm_model():
    llm = LLM(
        model=MODEL_NAME,
        dtype=REFERENCE_DTYPE,
        gpu_memory_utilization=0.3,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_model_tp2():
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 requires at least 2 GPUs")
    llm = LLM(
        model=MODEL_NAME,
        dtype=REFERENCE_DTYPE,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=2,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _generate_one(
    llm: LLM, extra_args: dict, max_tokens: int = 1, ignore_eos: bool = False
) -> RequestOutput:
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        extra_args=extra_args,
    )
    return llm.generate([PROMPT], sp)[0]


def _hf_down_proj_input(model, tokenizer, prompt: str) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    down_proj = model.model.layers[LAYER_IDX].mlp.down_proj
    handle = down_proj.register_forward_pre_hook(
        lambda _m, args: captured.append(args[0][0].detach().float().cpu())
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return captured[0]


class TestMlpActHooks:
    def test_capture_matches_hf_down_proj_input(self, llm_model, hf_model):
        """Per-request slices in a mixed batch equal HF's ``down_proj`` input."""
        model, tokenizer = hf_model

        def capture(ctx, acts):
            ctx.saved.setdefault("parts", []).append(acts.detach().float().cpu())
            ctx.saved["neuron_offset"] = ctx.neuron_offset

        params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                extra_args={
                    "apply_hooks": [
                        Hook(fn=capture, layer_indices=[LAYER_IDX], site="mlp_act")
                    ]
                },
            )
            for _ in PROMPTS
        ]
        outputs = llm_model.generate(list(PROMPTS), params)

        for prompt, out in zip(PROMPTS, outputs):
            saved = out.hook_results["0"]  # type: ignore[reportAttributeAccessIssue]
            assert saved["neuron_offset"] == 0
            ref = _hf_down_proj_input(model, tokenizer, prompt)
            got = torch.cat(saved["parts"], dim=0)[: ref.shape[0]]
            assert got.shape == ref.shape, f"{prompt!r}: {got.shape} vs {ref.shape}"
            diff = (got - ref).abs().mean().item()
            scale = ref.abs().mean().item()
            assert diff < 1e-2 * scale, f"{prompt!r}: diff {diff:.3e}, ref {scale:.3e}"

    @pytest.mark.parametrize("site", ["mlp_act", "layer"])
    def test_position_offset_across_prefill_and_decode(self, llm_model, site):
        def record(ctx, acts):
            ctx.saved.setdefault("calls", []).append(
                (ctx.position_offset, ctx.seq_len, acts.shape[0])
            )

        out = _generate_one(
            llm_model,
            {"apply_hooks": [Hook(fn=record, layer_indices=[LAYER_IDX], site=site)]},
            max_tokens=4,
            ignore_eos=True,
        )
        calls = out.hook_results["0"]["calls"]  # type: ignore[reportAttributeAccessIssue]
        n_prompt = len(out.prompt_token_ids or [])
        assert calls[0] == (0, n_prompt, n_prompt)
        assert len(calls) == 4, calls
        for j, call in enumerate(calls[1:], start=1):
            assert call == (n_prompt + j - 1, 1, 1), calls

    def test_zero_ablation_closed_form(self, llm_model, hf_model):
        """Zeroing neuron k shifts the layer's residual output by -a_k ⊗ w_down[:, k]."""
        model, _ = hf_model

        def ablate(ctx, acts, _k=NEURON):
            ctx.saved.setdefault("a_k", []).append(acts[:, _k].detach().float().cpu())
            acts = acts.clone()
            acts[:, _k] = 0
            return acts

        base = _generate_one(llm_model, {"output_residual_stream": [LAYER_IDX]})
        ablated = _generate_one(
            llm_model,
            {
                "output_residual_stream": [LAYER_IDX],
                "apply_hooks": [
                    Hook(fn=ablate, layer_indices=[LAYER_IDX], site="mlp_act")
                ],
            },
        )
        # Compare prompt (prefill) tokens only.
        n = len(base.prompt_token_ids or [])
        base_rs = base.activations["residual_stream"][0, :n].float()  # type: ignore[reportAttributeAccessIssue]
        ablated_rs = ablated.activations["residual_stream"][0, :n].float()  # type: ignore[reportAttributeAccessIssue]
        a_k = torch.cat(ablated.hook_results["0"]["a_k"], dim=0)[:n]  # type: ignore[reportAttributeAccessIssue]
        w_col = model.model.layers[LAYER_IDX].mlp.down_proj.weight[:, NEURON]
        expected = torch.outer(a_k, w_col.detach().float().cpu())

        assert a_k.abs().max() > 0, "neuron is inactive on this prompt"
        torch.testing.assert_close(base_rs - ablated_rs, expected, rtol=5e-2, atol=1e-3)

    def test_observing_hook_changes_nothing(self, llm_model):
        def observe(ctx, acts):
            ctx.saved["seen"] = ctx.saved.get("seen", 0) + acts.shape[0]

        base = _generate_one(llm_model, {"output_residual_stream": [LAYER_IDX]})
        observed = _generate_one(
            llm_model,
            {
                "output_residual_stream": [LAYER_IDX],
                "apply_hooks": [
                    Hook(fn=observe, layer_indices=[LAYER_IDX], site="mlp_act")
                ],
            },
        )
        assert observed.hook_results["0"]["seen"] > 0  # type: ignore[reportAttributeAccessIssue]
        assert torch.equal(
            observed.activations["residual_stream"],  # type: ignore[reportAttributeAccessIssue]
            base.activations["residual_stream"],  # type: ignore[reportAttributeAccessIssue]
        )


class TestMlpActHooksTP2:
    def test_capture_tp2_shards_match_hf(self, llm_model_tp2, hf_model):
        """Each rank sees its column shard; offsets tile the full width."""
        model, tokenizer = hf_model

        def capture(ctx, acts):
            key = f"cols_{ctx.neuron_offset}"
            ctx.saved.setdefault(key, []).append(acts.detach().float().cpu())

        out = _generate_one(
            llm_model_tp2,
            {
                "apply_hooks": [
                    Hook(fn=capture, layer_indices=[LAYER_IDX], site="mlp_act")
                ]
            },
        )
        saved = out.hook_results["0"]  # type: ignore[reportAttributeAccessIssue]
        ref = _hf_down_proj_input(model, tokenizer, PROMPT)
        offsets = sorted(int(k.removeprefix("cols_")) for k in saved)
        width = ref.shape[1] // 2
        assert offsets == [0, width], offsets
        got = torch.cat(
            [torch.cat(saved[f"cols_{o}"], dim=0)[: ref.shape[0]] for o in offsets],
            dim=1,
        )
        assert got.shape == ref.shape
        diff = (got - ref).abs().mean().item()
        assert diff < 1e-2 * ref.abs().mean().item()

    def test_zero_ablation_tp2_on_second_rank(self, llm_model_tp2, hf_model):
        """``neuron_offset`` lets a hook target a neuron owned by rank 1."""
        model, _ = hf_model
        mlp = model.model.layers[LAYER_IDX].mlp
        neuron = mlp.down_proj.weight.shape[1] - 1 - NEURON  # in rank 1's shard

        def ablate(ctx, acts, _k=neuron):
            local = _k - ctx.neuron_offset
            if not 0 <= local < acts.shape[1]:
                return None
            ctx.saved.setdefault("a_k", []).append(acts[:, local].float().cpu())
            acts = acts.clone()
            acts[:, local] = 0
            return acts

        base = _generate_one(llm_model_tp2, {"output_residual_stream": [LAYER_IDX]})
        ablated = _generate_one(
            llm_model_tp2,
            {
                "output_residual_stream": [LAYER_IDX],
                "apply_hooks": [
                    Hook(fn=ablate, layer_indices=[LAYER_IDX], site="mlp_act")
                ],
            },
        )
        n = len(base.prompt_token_ids or [])
        base_rs = base.activations["residual_stream"][0, :n].float()  # type: ignore[reportAttributeAccessIssue]
        ablated_rs = ablated.activations["residual_stream"][0, :n].float()  # type: ignore[reportAttributeAccessIssue]
        a_k = torch.cat(ablated.hook_results["0"]["a_k"], dim=0)[:n]  # type: ignore[reportAttributeAccessIssue]
        w_col = mlp.down_proj.weight[:, neuron].detach().float().cpu()

        assert a_k.abs().max() > 0, "neuron is inactive on this prompt"
        torch.testing.assert_close(
            base_rs - ablated_rs, torch.outer(a_k, w_col), rtol=5e-2, atol=1e-3
        )
