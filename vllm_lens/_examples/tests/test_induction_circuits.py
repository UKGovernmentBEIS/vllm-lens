"""Tests for induction_circuits.py: correctness, robustness, and validity evidence.

Three independent layers of evidence, matching the PR write-up:

1. ``TestInductionTargets`` -- pure unit tests for the target-derivation
   logic (no GPU).
2. ``TestDlaMethodologySanityCheck`` -- proves the OV-circuit direct logit
   attribution (DLA) *method itself* is sound, independent of vllm-lens or
   vLLM entirely: applied via TransformerLens directly to GPT-2 small (whose
   induction heads are extensively documented in prior work), does DLA
   surface the same heads as a real attention-pattern-based detector run on
   the exact same model? This isolates "is the math right" from "does vLLM
   capture things correctly".
3. ``TestRobustnessAcrossServingConfigs`` -- proves vllm-lens's capture of
   the *same* quantities (residual stream, per-head OV contributions) is
   consistent across offline sync, chunked prefill, and async serving --
   the "robust across typical serving configurations" requirement.
4. ``TestGroundTruthCorrelation`` -- checks the full vllm-lens pipeline
   (real vLLM engine, Qwen2.5-0.5B-Instruct) against the saved
   attention-pattern ground truth from ``_induction_ground_truth.py``.
"""

from __future__ import annotations

import gc

import pytest
import torch
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from vllm_lens._examples._induction_ground_truth import (
    QWEN_MODEL_NAME,
    build_random_token_id_seqs,
)
from vllm_lens._examples._induction_prompts import (
    induction_targets,
    make_induction_sequence,
)
from vllm_lens._examples.induction_circuits import (
    capture_induction_activations,
    compute_induction_scores,
    correlate_with_ground_truth,
    get_final_norm_and_unembed,
)

from vllm_lens.tests.conftest import NUM_LAYERS

MODEL_NAME = QWEN_MODEL_NAME

# Small/fast settings for the serving-config robustness matrix -- these
# tests only need to show *consistency* across configs, not reproduce the
# full-quality ground-truth comparison (which reuses the QWEN_* constants
# instead, see TestGroundTruthCorrelation).
_ROBUSTNESS_SEQ_LEN = 20
_ROBUSTNESS_N_SEQUENCES = 3
_ROBUSTNESS_SEED_BASE = 12345


# ============================================================
# 1. Pure unit tests
# ============================================================


class TestInductionTargets:
    def test_worked_example(self) -> None:
        assert induction_targets([5, 9, 2, 5, 9]) == [None, None, None, 9, 2]

    def test_no_repeats(self) -> None:
        assert induction_targets([1, 2, 3, 4]) == [None, None, None, None]

    def test_empty(self) -> None:
        assert induction_targets([]) == []

    def test_repeated_random_sequence_has_valid_targets_only_in_second_half(
        self,
    ) -> None:
        seq = make_induction_sequence(vocab_size=1000, seq_len=10, seed=0)
        targets = induction_targets(seq)
        assert targets[:10] == [None] * 10
        assert all(t is not None for t in targets[10:])
        # Second half's targets are exactly the first half's tokens, shifted.
        assert targets[10:] == seq[1:10] + [seq[0]]


# ============================================================
# 2. DLA methodology sanity check (TransformerLens + GPT-2, no vLLM at all)
# ============================================================


class TestDlaMethodologySanityCheck:
    def test_dla_recovers_same_heads_as_attention_pattern_detector(self) -> None:
        """Independent methodology check: does DLA agree with attention patterns?

        Runs entirely through TransformerLens on GPT-2 small (no vLLM, no
        vllm-lens capture code) so this isolates "is the OV-circuit DLA
        *idea* correct" from anything about vLLM's plumbing. Both signals
        are computed from scratch here rather than reusing
        ``_induction_ground_truth.py``'s saved values, to keep this test
        fully self-contained.
        """
        transformer_lens = pytest.importorskip("transformer_lens")
        from transformer_lens.head_detector import (
            compute_head_attention_similarity_score,
            get_induction_head_detection_pattern,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = transformer_lens.HookedTransformer.from_pretrained(
            "gpt2", device=device
        )
        n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

        seqs = [
            make_induction_sequence(vocab_size=model.cfg.d_vocab, seq_len=30, seed=s)
            for s in range(8)
        ]

        dla_scores = torch.zeros(n_layers, n_heads, device=device)
        attn_scores = torch.zeros(n_layers, n_heads, device=device)
        total_positions = 0

        for token_ids in seqs:
            targets = induction_targets(token_ids)
            valid = [p for p, t in enumerate(targets) if t is not None]
            tokens = torch.tensor([token_ids], device=device)

            detection_pattern = get_induction_head_detection_pattern(tokens.cpu()).to(
                device
            )
            _, cache = model.run_with_cache(
                tokens,
                return_type=None,
                names_filter=lambda name: (
                    name.endswith("hook_z")
                    or name.endswith("hook_pattern")
                    or name.endswith("hook_resid_post")
                ),
                remove_batch_dim=True,
            )

            final_residual = cache["resid_post", n_layers - 1].float()
            centered = final_residual - final_residual.mean(dim=-1, keepdim=True)
            inv_std = 1.0 / (centered.pow(2).mean(dim=-1) + model.cfg.eps).sqrt()

            pos_idx = torch.tensor(valid)
            target_ids = torch.tensor([targets[p] for p in valid])
            target_dirs = model.W_U.float().T[target_ids]
            inv_std_v = inv_std[pos_idx]

            for layer in range(n_layers):
                z = cache["z", layer].float()
                w_o = model.W_O[layer].float()
                head_contrib = torch.einsum("thd,hdm->thm", z, w_o)[pos_idx]
                centered_v = head_contrib - head_contrib.mean(dim=-1, keepdim=True)
                scaled = centered_v * inv_std_v[:, None, None]
                dla_scores[layer] += torch.einsum(
                    "vhm,vm->vh", scaled, target_dirs
                ).sum(dim=0)

                layer_patterns = cache["pattern", layer, "attn"]
                for head in range(n_heads):
                    attn_scores[layer, head] += compute_head_attention_similarity_score(
                        layer_patterns[head].clone(),
                        detection_pattern,
                        exclude_bos=True,
                        exclude_current_token=True,
                        error_measure="mul",
                    )
            total_positions += len(valid)

        dla_scores /= total_positions
        attn_scores /= len(seqs)

        k = 10
        dla_top = set(torch.topk(dla_scores.flatten(), k).indices.tolist())
        attn_top = set(torch.topk(attn_scores.flatten(), k).indices.tolist())
        overlap = len(dla_top & attn_top) / k

        # Not a tight bound -- direct vs indirect effects can genuinely
        # diverge (see induction_circuits.py's note on this) -- but for
        # GPT-2 small, whose induction heads are unusually "clean" and
        # well-documented, a strong majority overlap is the expected,
        # literature-consistent result.
        assert overlap >= 0.5, (
            f"DLA top-{k} heads only overlap {overlap:.0%} with the "
            f"attention-pattern detector's top-{k} on GPT-2 small; expected "
            "strong agreement for this well-studied model."
        )


# ============================================================
# 3. Robustness across serving configurations
# ============================================================


@pytest.fixture(scope="module")
def robustness_token_id_seqs() -> list[list[int]]:
    return build_random_token_id_seqs(
        MODEL_NAME,
        _ROBUSTNESS_SEQ_LEN,
        _ROBUSTNESS_N_SEQUENCES,
        _ROBUSTNESS_SEED_BASE,
    )


@pytest.fixture(scope="module")
def default_llm():
    llm = LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3)
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def chunked_llm():
    # seq_len=20 -> 40 total tokens per sequence; force at least one chunk
    # boundary mid-prompt.
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
async def async_engine():
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield engine
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


def _assert_activations_close(
    a: dict, b: dict, *, label_a: str, label_b: str, tol: float = 1.5e-1
) -> None:
    # bf16 accumulates rounding differently depending on batching/kernel path
    # (chunked vs. single-shot prefill, sync vs. async scheduling), and this
    # compounds over NUM_LAYERS layers -- activation_oracle.py's tests
    # document up to ~0.27 mean-abs-diff for cross-framework comparisons;
    # same-framework-different-config noise observed here is ~0.05.
    for key in ("residual_stream", "head_contributions"):
        ta, tb = a[key].float(), b[key].float()
        assert ta.shape == tb.shape, (
            f"{key} shape mismatch: {label_a} {ta.shape} vs {label_b} {tb.shape}"
        )
        mean_abs_diff = (ta - tb).abs().mean().item()
        assert mean_abs_diff < tol, (
            f"{key} mismatch between {label_a} and {label_b}: "
            f"mean abs diff {mean_abs_diff:.6f} >= {tol}"
        )


class TestRobustnessAcrossServingConfigs:
    def test_offline_vs_chunked_prefill(
        self, robustness_token_id_seqs, default_llm, chunked_llm
    ) -> None:
        """Same captured activations whether or not chunked prefill splits the
        prompt across multiple forward passes."""
        default_acts = capture_induction_activations(
            default_llm, robustness_token_id_seqs, NUM_LAYERS
        )
        chunked_acts = capture_induction_activations(
            chunked_llm, robustness_token_id_seqs, NUM_LAYERS
        )
        for i, (d, c) in enumerate(zip(default_acts, chunked_acts)):
            _assert_activations_close(
                d, c, label_a=f"default[{i}]", label_b=f"chunked[{i}]"
            )

    async def test_offline_vs_async(
        self, robustness_token_id_seqs, default_llm, async_engine
    ) -> None:
        """Same captured activations via offline LLM.generate vs AsyncLLMEngine.generate."""
        default_acts = capture_induction_activations(
            default_llm, robustness_token_id_seqs, NUM_LAYERS
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={
                "output_residual_stream": [NUM_LAYERS - 1],
                "output_head_contributions": True,
            },
        )
        async_acts = []
        for i, token_ids in enumerate(robustness_token_id_seqs):
            final = None
            async for output in async_engine.generate(
                {"prompt_token_ids": token_ids},
                sampling_params,
                request_id=f"induction-{i}",
            ):
                final = output
            assert final is not None
            async_acts.append(final.activations)

        for i, (d, a) in enumerate(zip(default_acts, async_acts)):
            _assert_activations_close(
                d, a, label_a=f"offline[{i}]", label_b=f"async[{i}]"
            )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Pipeline/tensor parallelism tests require at least 2 GPUs",
)
class TestRobustnessUnderParallelism:
    def test_offline_vs_tensor_parallel(
        self, robustness_token_id_seqs, default_llm
    ) -> None:
        """Same captured activations under TP=2 as under TP=1.

        Exercises the tensor-parallel head-index-offset/merge logic in
        ``_merge_captured_head_contributions`` (every TP rank captures its
        own local head shard, unlike residual-stream capture).
        """
        default_acts = capture_induction_activations(
            default_llm, robustness_token_id_seqs, NUM_LAYERS
        )

        tp_llm = LLM(
            model=MODEL_NAME,
            dtype="auto",
            gpu_memory_utilization=0.3,
            tensor_parallel_size=2,
        )
        try:
            tp_acts = capture_induction_activations(
                tp_llm, robustness_token_id_seqs, NUM_LAYERS
            )
        finally:
            del tp_llm
            gc.collect()
            torch.cuda.empty_cache()

        for i, (d, t) in enumerate(zip(default_acts, tp_acts)):
            _assert_activations_close(d, t, label_a=f"tp1[{i}]", label_b=f"tp2[{i}]")


# ============================================================
# 4. Ground-truth correlation (full pipeline, saved TransformerLens scores)
# ============================================================


class TestGroundTruthCorrelation:
    @pytest.fixture(scope="class")
    def ground_truth_path(self):
        from pathlib import Path

        path = (
            Path(__file__).parent
            / "utils"
            / "induction_ground_truth_qwen2.5-0.5b-instruct.pt"
        )
        if not path.exists():
            pytest.skip(
                f"Ground truth not found at {path}; run "
                "`python -m vllm_lens._examples._induction_ground_truth` first."
            )
        return path

    def test_scores_correlate_with_attention_pattern_ground_truth(
        self, ground_truth_path
    ) -> None:
        gt = torch.load(ground_truth_path, weights_only=True)
        assert gt["model_name"] == MODEL_NAME

        gamma, eps, unembed = get_final_norm_and_unembed(MODEL_NAME)
        llm = LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3)
        try:
            scores = compute_induction_scores(
                llm, gt["token_id_seqs"], NUM_LAYERS, gamma, eps, unembed
            )
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()

        # Internal validity: a genuine induction signal should be top-heavy,
        # not a flat/degenerate distribution (would indicate a broken
        # computation regardless of ground-truth agreement).
        flat = scores.flatten()
        median = flat.median().item()
        top_score = flat.max().item()
        assert top_score > 3 * max(median, 1e-9), (
            "Induction score distribution isn't top-heavy -- top score "
            f"{top_score:.4f} vs median {median:.4f}"
        )

        spearman, overlap = correlate_with_ground_truth(scores, gt["scores"])
        # See induction_circuits.py's note: DLA (writes) and attention
        # patterns (reads) are related but distinct signals, so this is a
        # loose bound guarding against regressions, not a tight one.
        assert spearman > 0.15, f"Spearman correlation too low: {spearman:.3f}"
        assert overlap > 0, "No overlap at all with ground-truth top heads"
