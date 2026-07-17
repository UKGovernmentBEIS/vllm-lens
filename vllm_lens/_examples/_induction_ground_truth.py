"""Ground-truth induction-head localization via TransformerLens attention patterns.

Deliberately independent of vllm-lens's own residual-stream / OV-circuit
based induction detector: vLLM's fused attention kernels never materialize a
hookable attention-weight tensor, so vllm-lens's detector reads what each
head *writes* into the residual stream, not what it *attends to*. This
module instead reads real attention patterns (which TransformerLens exposes
directly) and scores every head against the canonical attention-pattern
signature of an induction head (Olsson et al. 2022, "In-context Learning and
Induction Heads"), giving an independent check that isn't subject to the
same blind spots as the OV-circuit method.

Token ids must be supplied pre-tokenized (not a text string) so that the
exact same sequence can also be fed to vLLM: ``HookedTransformer.to_tokens``
(and ``head_detector.detect_head``'s public string-based API) retokenizes
independently and could silently diverge from vLLM's token ids, which would
make any cross-method comparison meaningless.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformer_lens.head_detector import (
    compute_head_attention_similarity_score,
    get_induction_head_detection_pattern,
)

from vllm_lens._examples._induction_prompts import make_induction_sequence

# Methodology sanity check: GPT-2 small's induction heads are extensively
# documented in prior interpretability work, so a clean top-heavy score
# distribution here is evidence the scoring method itself is correct,
# independent of anything vllm-lens does.
GPT2_MODEL_NAME = "gpt2"
GPT2_SEQ_LEN = 40
GPT2_N_SEQUENCES = 24
GPT2_SEED_BASE = 1000

# The actual comparison target: same model vllm-lens's own OV-circuit
# detector runs on, so scores are directly comparable head-for-head.
QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_SEQ_LEN = 40
QWEN_N_SEQUENCES = 24
QWEN_SEED_BASE = 0

_GROUND_TRUTH_PATH = (
    Path(__file__).parent
    / "tests"
    / "utils"
    / "induction_ground_truth_qwen2.5-0.5b-instruct.pt"
)


def build_random_token_id_seqs(
    model_name: str,
    seq_len: int,
    n_sequences: int,
    seed_base: int,
) -> list[list[int]]:
    """Build ``n_sequences`` independent induction sequences for ``model_name``'s tokenizer.

    Uses the model's own tokenizer's vocab size and special-token ids, so
    the returned ids are valid inputs for that specific model (in either
    vLLM or TransformerLens).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    exclude = set(tokenizer.all_special_ids or [])
    vocab_size = len(tokenizer)
    return [
        make_induction_sequence(
            vocab_size=vocab_size,
            seq_len=seq_len,
            seed=seed_base + i,
            exclude_token_ids=exclude,
        )
        for i in range(n_sequences)
    ]


def attention_pattern_induction_scores(
    model_name: str,
    token_id_seqs: list[list[int]],
    device: str = "cuda",
) -> torch.Tensor:
    """Average attention-pattern induction score per ``(layer, head)``.

    Loads ``model_name`` into a ``HookedTransformer`` once and, for each
    already-tokenized sequence in ``token_id_seqs``, runs it with a cache,
    builds the induction detection pattern directly from those token ids
    (``get_induction_head_detection_pattern`` is a pure function of the ids,
    not a retokenization), and scores every head's attention pattern against
    it (``exclude_bos=True, exclude_current_token=True, error_measure="mul"``
    -- the standard settings from Olsson et al. 2022 / TransformerLens's
    head-detector). Scores are averaged across all sequences.

    Returns a ``(n_layers, n_heads)`` tensor.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    totals = torch.zeros(n_layers, n_heads)

    for token_ids in token_id_seqs:
        tokens = torch.tensor([token_ids], device=device)
        detection_pattern = get_induction_head_detection_pattern(tokens.cpu()).to(
            device
        )
        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            names_filter=lambda name: name.endswith("hook_pattern"),
            remove_batch_dim=True,
        )
        for layer in range(n_layers):
            layer_patterns = cache["pattern", layer, "attn"]  # (n_heads, dest, src)
            for head in range(n_heads):
                totals[layer, head] += compute_head_attention_similarity_score(
                    layer_patterns[head].clone(),
                    detection_pattern,
                    exclude_bos=True,
                    exclude_current_token=True,
                    error_measure="mul",
                )

    return totals / len(token_id_seqs)


def _report_distribution(model_name: str, scores: torch.Tensor, k: int = 8) -> None:
    n_heads = scores.shape[1]
    flat = scores.flatten()
    top = torch.topk(flat, min(k, flat.numel()))
    median = flat.median().item()
    print(f"\n{model_name}: top-{k} induction scores (median overall = {median:.4f})")
    for idx, val in zip(top.indices.tolist(), top.values.tolist()):
        layer, head = idx // n_heads, idx % n_heads
        print(f"  L{layer}H{head}: {val:.4f}")
    print(f"  max/median ratio: {top.values[0].item() / max(median, 1e-9):.1f}x")


def main() -> None:
    # (a) Methodology sanity check on GPT-2 small.
    gpt2_seqs = build_random_token_id_seqs(
        GPT2_MODEL_NAME, GPT2_SEQ_LEN, GPT2_N_SEQUENCES, GPT2_SEED_BASE
    )
    gpt2_scores = attention_pattern_induction_scores(GPT2_MODEL_NAME, gpt2_seqs)
    _report_distribution(GPT2_MODEL_NAME, gpt2_scores)

    # (b) Ground truth for the model vllm-lens's own detector targets.
    qwen_seqs = build_random_token_id_seqs(
        QWEN_MODEL_NAME, QWEN_SEQ_LEN, QWEN_N_SEQUENCES, QWEN_SEED_BASE
    )
    qwen_scores = attention_pattern_induction_scores(QWEN_MODEL_NAME, qwen_seqs)
    _report_distribution(QWEN_MODEL_NAME, qwen_scores)

    _GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": qwen_scores,
            "token_id_seqs": qwen_seqs,
            "model_name": QWEN_MODEL_NAME,
            "seeds": list(range(QWEN_SEED_BASE, QWEN_SEED_BASE + QWEN_N_SEQUENCES)),
        },
        _GROUND_TRUTH_PATH,
    )
    print(f"\nSaved Qwen2.5-0.5B-Instruct ground truth to {_GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
