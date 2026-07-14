import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import LAYER_IDX, MODEL_NAME, NUM_LAYERS, PROMPT, PROMPTS

# Layers straddling the PP=2 stage boundary (stage 0 = [0, NUM_LAYERS//2)),
# so the per-request cross-rank concat in _merge_captured_states_batch is
# actually exercised (a single layer lives on one stage only).
_CROSS_STAGE_LAYERS = [LAYER_IDX, NUM_LAYERS - 4]


@pytest.fixture(scope="module")
def llm_model():
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_model_tp2():
    """Tensor-parallel (TP=2) offline engine for batched-fetch coverage."""
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
        tensor_parallel_size=2,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_model_pp2():
    """Pipeline-parallel (PP=2) offline engine for batched-fetch coverage.

    Exercises ``get_captured_states_batch`` + ``_merge_captured_states_batch``
    across PP stages (per-request cross-rank concat).
    """
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
        pipeline_parallel_size=2,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _get_hf_acts(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    num_tokens = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER_IDX + 1][0, :num_tokens].float()


def _get_vllm_acts(llm: LLM, prompts: list[str], max_tokens: int) -> list[torch.Tensor]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": [LAYER_IDX]},
    )
    outputs = llm.generate(prompts, sampling_params)
    return [out.activations["residual_stream"] for out in outputs]  # type: ignore[reportAttributeAccessIssue]


def _assert_close(vllm_acts: torch.Tensor, hf_acts: torch.Tensor):
    assert vllm_acts.shape == hf_acts.shape, (
        f"Shape mismatch: vLLM {vllm_acts.shape} vs HF {hf_acts.shape}"
    )
    mean_abs_diff = (vllm_acts - hf_acts.to(vllm_acts.device)).abs().mean().item()
    assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"


class TestOfflineMatchesTransformers:
    def test_single_prompt_1_token(self, llm_model, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        streams = _get_vllm_acts(llm_model, [PROMPT], max_tokens=1)
        vllm_acts = streams[0][0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    def test_single_prompt_10_tokens(self, llm_model, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        streams = _get_vllm_acts(llm_model, [PROMPT], max_tokens=10)
        vllm_acts = streams[0][0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    def test_batch_prompts_10_tokens(self, llm_model, hf_model):
        model, tokenizer = hf_model

        streams = _get_vllm_acts(llm_model, PROMPTS, max_tokens=10)

        for i, prompt in enumerate(PROMPTS):
            num_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            hf_acts = _get_hf_acts(model, tokenizer, prompt)
            vllm_acts = streams[i][0, :num_tokens].float()
            _assert_close(vllm_acts, hf_acts)

    def test_activation_shape_matches_token_count(self, llm_model):
        """Activation seq dim must equal prompt + generated - 1."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        outputs = llm_model.generate(PROMPTS, sampling_params)
        for output in outputs:
            n_prompt = len(output.prompt_token_ids)
            n_gen = len(output.outputs[0].token_ids)
            expected = n_prompt + n_gen - 1
            rs = output.activations["residual_stream"]
            assert rs.shape[1] == expected, (
                f"Expected seq_len={expected} (prompt={n_prompt}, gen={n_gen}), "
                f"got {rs.shape[1]}"
            )


def _capture_batch(
    llm: LLM, prompts: list[str], layers: list[int]
) -> list[torch.Tensor]:
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"output_residual_stream": layers},
    )
    return [out.activations["residual_stream"] for out in llm.generate(prompts, sp)]  # type: ignore[reportAttributeAccessIssue]


def _assert_batched_matches_individual(llm: LLM, layers: list[int]) -> None:
    """Batched capture of many prompts must equal capturing each alone.

    Isolates the batched-fetch demux/merge from parallel-vs-HF numerics:
    both paths run the same forward on the same engine, so any difference is
    a request-routing or cross-rank-merge error, not TP/PP fp drift.
    """
    batched = _capture_batch(llm, PROMPTS, layers)
    assert len(batched) == len(PROMPTS)
    for i, prompt in enumerate(PROMPTS):
        single = _capture_batch(llm, [prompt], layers)[0]
        assert batched[i].shape == single.shape, (
            f"prompt {i}: batched shape {batched[i].shape} != single "
            f"{single.shape} — request demux is misrouted"
        )
        assert batched[i].shape[0] == len(layers), (
            f"prompt {i}: expected {len(layers)} captured layers, got "
            f"{batched[i].shape[0]} — cross-rank merge dropped/duplicated layers"
        )
        # Scale-relative tolerance: batched vs single differ only by batched-GEMM
        # fp jitter (≈1%), whereas a misrouted request would differ by ~100%.
        b, s = batched[i].float(), single.float()
        rel = (b - s).abs().mean().item() / (s.abs().mean().item() + 1e-6)
        assert rel < 0.05, (
            f"prompt {i}: batched vs single relative mean abs diff {rel:.4f} "
            f"too large — wrong request's activations returned"
        )


class TestBatchedFetch:
    """Coverage for the batched ``get_captured_states_batch`` offline path.

    On ``LLM.generate`` the plugin fetches every request's captured
    activations in one RPC and demultiplexes them by request id. These verify
    each request gets *its own* activations (no cross-request mixing/dropping)
    and that the cross-rank batched merge is correct under TP and PP.

    (End-to-end value correctness vs HuggingFace is already covered — at TP=1
    by ``TestOfflineMatchesTransformers`` and under PP by
    ``test_activations_pp.py`` — both of which now route through the batched
    fetch on the offline path.)
    """

    def test_batched_fetch_matches_individual(self, llm_model):
        _assert_batched_matches_individual(llm_model, [LAYER_IDX])

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="TP=2 requires at least 2 GPUs"
    )
    def test_batched_fetch_matches_individual_tp2(self, llm_model_tp2):
        # Capture runs on TP rank 0 only; verify the batch demux still routes
        # every request correctly under TP.
        _assert_batched_matches_individual(llm_model_tp2, _CROSS_STAGE_LAYERS)

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="PP=2 requires at least 2 GPUs"
    )
    def test_batched_fetch_matches_individual_pp2(self, llm_model_pp2):
        # _CROSS_STAGE_LAYERS straddle both PP stages, so this exercises the
        # per-request cross-rank concat in _merge_captured_states_batch.
        _assert_batched_matches_individual(llm_model_pp2, _CROSS_STAGE_LAYERS)
