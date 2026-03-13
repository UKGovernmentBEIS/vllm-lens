"""Tests for activation_oracle.py prompt construction and activation collection.

Verifies that create_oracle_prompt() produces the same oracle prompt string
as the original ao_original_demo.py logic, and that collect_activations()
produces activations similar to the original demo's HuggingFace-based approach.
"""

import gc
import os

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import AsyncEngineArgs, AsyncLLMEngine

from vllm_lens._examples.activation_oracle import (
    LAYER,
    TARGET_MESSAGES,
    AOConfig,
    collect_activations,
    create_oracle_prompt,
    get_target_prompt_input_ids,
)

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
NUM_POSITIONS = 1
ORACLE_QUESTION = "Can you name all people the model is currently thinking about?"


def _check_close(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cos_threshold: float = 0.99,
    abs_threshold: float = 0.5,
) -> None:
    """Assert activations match using cosine similarity and mean absolute difference.

    Comparing vLLM against HuggingFace Transformers activations for an FP8
    model involves different FP8 matmul kernels, attention backends
    (FLASH_ATTN vs SDPA), and LoRA implementations (Punica vs PEFT).
    These differences compound across layers, so we use cosine similarity
    (direction) as the primary metric and a loose absolute threshold as a
    secondary sanity check.
    """
    assert a.shape == b.shape, f"Shape mismatch: vLLM {a.shape} vs HF {b.shape}"
    b_dev = b.to(a.device)

    # Flatten to (num_positions, hidden_dim) for per-position comparison
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b_dev.reshape(-1, b_dev.shape[-1]).float()

    # Cosine similarity per position
    cos_sim = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    # Mean absolute difference per position
    mean_abs = (a_flat - b_flat).abs().mean(dim=-1)

    cos_failed = cos_sim < cos_threshold
    abs_failed = mean_abs > abs_threshold
    failed = cos_failed | abs_failed

    if failed.any():
        lines: list[str] = []
        for pos in failed.nonzero(as_tuple=False).squeeze(-1).tolist():
            parts: list[str] = []
            if cos_failed[pos]:
                parts.append(f"cos_sim={cos_sim[pos]:.6f} < {cos_threshold}")
            if abs_failed[pos]:
                parts.append(f"mean_abs_diff={mean_abs[pos]:.6f} > {abs_threshold}")
            lines.append(f"  position {pos}: {', '.join(parts)}")
        header = f"{len(lines)}/{a_flat.shape[0]} positions failed:"
        pytest.fail("\n".join([header, *lines]))


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
async def engine():
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    eng = AsyncLLMEngine.from_engine_args(engine_args)
    yield eng
    eng.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def ao_config() -> AOConfig:
    """AOConfig with the known HuggingFace Hub values for the Qwen3-8B oracle."""
    return AOConfig(
        hook_onto_layer=1,
        special_token=" ?",
        prefix_template="Layer: {layer}\\n{special_token} * {num_positions} \\n",
        steering_coefficient=1.0,
        act_layer_combinations=[[9], [18], [27]],
    )


class TestCreateOraclePrompt:
    def test_prompt_matches_original_demo(
        self,
        ao_config: AOConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """create_oracle_prompt() must produce the same string as
        ao_original_demo.py (captured 2025-02-25, Qwen3-8B-FP8, device 1)."""
        result = create_oracle_prompt(
            config=ao_config,
            layer=LAYER,
            num_positions=NUM_POSITIONS,
            oracle_question=ORACLE_QUESTION,
            tokenizer=tokenizer,
        )

        # Captured from the original activation oracle demo
        # https://github.com/adamkarvonen/activation_oracles/blob/main/experiments/activation_oracle_demo.ipynb

        expected = (
            "<|im_start|>user\n"
            "Layer: 18\n"
            " ? \n"
            "Can you name all people the model is currently thinking about?"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
        )

        assert result == expected, (
            f"Prompt mismatch.\n"
            f"  create_oracle_prompt: {result!r}\n"
            f"  original demo:        {expected!r}"
        )


class TestGetTargetPromptInputIds:
    def test_prompt_ids_match_original_demo(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """get_target_prompt_input_ids() must produce the same token IDs as
        ao_original_demo.py's tokenization path (captured from Qwen3-8B)."""
        result = get_target_prompt_input_ids(TARGET_MESSAGES, tokenizer)

        # Captured from the original activation oracle demo
        # https://github.com/adamkarvonen/activation_oracles/blob/main/experiments/activation_oracle_demo.ipynb
        # fmt: off
        expected = [
            151644, 872, 198, 785, 54375, 879, 53144, 17280, 1023, 15599,
            264, 5458, 879, 18047, 458, 43345, 13, 2938, 5458, 594,
            1429, 11245, 59972, 572, 151645, 198,
        ]
        # fmt: on

        assert result == expected, (
            f"Token ID mismatch.\n"
            f"  get_target_prompt_input_ids ({len(result)} tokens): {result}\n"
            f"  original demo ({len(expected)} tokens): {expected}"
        )


class TestCollectActivations:
    # Cross-framework activation comparison is skipped for two reasons:
    #
    # A) LoRA during collection — the HF fixture was captured with the
    #    oracle LoRA inadvertently active (PEFT's inject_adapter re-sets
    #    whichever adapter was last loaded).  We now follow the paper
    #    (arXiv:2512.15674) and collect from the base model (no LoRA),
    #    so the fixture is no longer a valid reference.
    #
    # B) Cross-framework numerical differences — even ignoring the LoRA
    #    mismatch, vLLM and HF produce ~0.03–0.27 mean-abs-diff per
    #    position at layer 18 (cosine sim ~0.997+).  Three sources:
    #      1. FP8 matmul kernels — vLLM custom CUDA vs HF/PyTorch
    #      2. Attention backends — FLASH_ATTN (vLLM) vs SDPA (HF)
    #      3. LoRA implementations — Punica GPU (vLLM) vs PEFT (HF)
    #
    # Re-enable when a vLLM-generated reference fixture is available.
    @pytest.mark.skip(
        reason="HF fixture uses LoRA during collection + cross-framework FP8 diffs"
    )
    async def test_activations_similar_original_demo(
        self,
        engine: AsyncLLMEngine,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """collect_activations() via vLLM must produce activations similar to
        the original demo's HuggingFace-based approach (Qwen3-8B-FP8)."""
        input_ids = get_target_prompt_input_ids(TARGET_MESSAGES, tokenizer)
        result = await collect_activations(engine, input_ids, LAYER)

        fixture_path = os.path.join(
            os.path.dirname(__file__), "utils", "original_demo_activations.pt"
        )
        original = torch.load(fixture_path, weights_only=True)

        _check_close(result, original)
