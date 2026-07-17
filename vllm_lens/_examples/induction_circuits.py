"""Automatic induction-circuit localization via vllm-lens.

Identifies which (layer, head) pairs implement induction — the classic
in-context "[A][B] ... [A] -> predict [B]" copying circuit (Olsson et al.
2022, "In-context Learning and Induction Heads") — using only vllm-lens's
public activation-capture surface (``extra_args``), so the same computation
runs identically whether the engine is offline/online, sync/async, or under
chunked prefill / tensor / pipeline parallelism.

Why not just read attention patterns?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vLLM's attention runs through fused kernels with no hookable module that
exposes the attention weight matrix, so vllm-lens cannot read attention
patterns directly (see ``_worker_ext.py``'s module docstring). Instead this
uses each head's *write* to the residual stream: ``extra_args={
"output_head_contributions": True}`` captures, per layer and per head, the
exact additive contribution ``z_h @ W_O[:, h_slice].T`` that head makes to
the residual stream (an algebraic decomposition of the linear ``o_proj``
projection — no attention weights needed).

Method: direct logit attribution (DLA) on repeated-random tokens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Build ``random_ids + random_ids`` sequences (``_induction_prompts``) --
   random tokens so correct next-token prediction has no explanation other
   than a genuine copying circuit.
2. Capture, per sequence: each head's residual-stream contribution at every
   layer, and the final decoder layer's *full* residual (needed for the
   RMSNorm scale factor -- see below).
3. Project each head's contribution through the model's final RMSNorm +
   unembedding, reading off the logit assigned to the "induction-correct"
   target token at each position (``_induction_prompts.induction_targets``).
   RMSNorm's scale factor is a fixed value per token position (derived from
   the true, full residual, not from any individual component), so it
   distributes linearly over a sum -- this lets us attribute the *exact*
   direct contribution of a single head to the final logits without
   needing to re-run the model. This is the standard logit-lens / direct
   logit attribution method, applied at head granularity instead of layer
   granularity.
4. Average the target-token logit contribution over all valid positions
   and sequences -> ``induction_score[layer, head]``.

This is compared against ``_induction_ground_truth.py`` -- an entirely
independent method (real attention patterns, via TransformerLens, on the
same model and the same token ids) -- for validation.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Annotated, Any

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.table import Table
from transformers import AutoConfig, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from vllm_lens._examples._induction_ground_truth import (
    QWEN_MODEL_NAME,
    QWEN_N_SEQUENCES,
    QWEN_SEED_BASE,
    QWEN_SEQ_LEN,
    build_random_token_id_seqs,
)
from vllm_lens._examples._induction_prompts import induction_targets

MODEL_NAME = QWEN_MODEL_NAME
SEQ_LEN = QWEN_SEQ_LEN
N_SEQUENCES = QWEN_N_SEQUENCES
SEED_BASE = QWEN_SEED_BASE


# ============================================================
# Final norm + unembedding (needed for the logit-lens projection)
# ============================================================


def get_final_norm_and_unembed(
    model_name: str,
    device: str = "cuda",
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Extract the final RMSNorm weight/eps and unembedding matrix.

    Loaded via a plain HuggingFace model (same pattern already used for
    cross-framework ground truth elsewhere in this repo, e.g.
    ``tests/conftest.py``'s ``hf_model`` fixture) rather than reaching into
    vLLM's internals -- these three tensors are all that's needed, and this
    keeps the projection logic entirely in ordinary, easy-to-audit Python.
    """
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map=device
    ).eval()
    final_norm = hf_model.model.norm  # type: ignore[union-attr]
    gamma = final_norm.weight.detach().clone()
    eps = float(getattr(final_norm, "variance_epsilon", 1e-6))
    unembed = hf_model.get_output_embeddings().weight.detach().clone()  # type: ignore[union-attr]
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return gamma, eps, unembed


# ============================================================
# Activation capture
# ============================================================


def capture_induction_activations(
    llm: LLM,
    token_id_seqs: list[list[int]],
    n_layers: int,
) -> list[dict[str, Any]]:
    """Capture per-head contributions (all layers) + final-layer residual.

    ``max_tokens=1`` -- only the prefill matters; we need activations over
    the prompt itself, not any generated continuation. Only the *last*
    layer's full residual stream is requested (needed for the RMSNorm
    scale factor), since requesting every layer's residual stream in
    addition to every layer's head contributions would roughly double the
    data moved off-GPU for no benefit here.
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={
            "output_residual_stream": [n_layers - 1],
            "output_head_contributions": True,
        },
    )
    prompts = [{"prompt_token_ids": ids} for ids in token_id_seqs]
    outputs = llm.generate(prompts, sampling_params)
    return [output.activations for output in outputs]  # type: ignore[attr-defined]


# ============================================================
# Direct logit attribution
# ============================================================


def direct_logit_attribution(
    final_residual: torch.Tensor,  # (total_pos, hidden)
    head_contributions: torch.Tensor,  # (n_layers, n_heads, total_pos, hidden)
    targets: Sequence[int | None],
    final_norm_weight: torch.Tensor,  # (hidden,)
    final_norm_eps: float,
    unembed_weight: torch.Tensor,  # (vocab, hidden)
) -> tuple[torch.Tensor, int]:
    """Per-head direct logit attribution to the induction-correct target token.

    RMSNorm(x) = x / rms(x) * gamma has no mean-centering, so it is exactly
    linear in x for a *fixed* rms(x) -- and rms(x) here is computed once
    from the true final residual (the sum of every component), not from any
    individual head's contribution. That means each head's contribution can
    be projected through the *same* final norm scale factor as the real
    residual stream and summed independently, giving each head's exact
    direct contribution to the final logits (ignoring indirect effects
    where a later layer reads and transforms this head's output -- the
    standard, accepted scope of "direct" logit attribution).

    Returns ``(per_head_logit_sum, n_valid_positions)`` so callers can
    average correctly across multiple sequences of possibly different
    valid-position counts.
    """
    valid_positions = [p for p, t in enumerate(targets) if t is not None]
    n_layers, n_heads = head_contributions.shape[0], head_contributions.shape[1]
    if not valid_positions:
        return torch.zeros(n_layers, n_heads), 0

    pos_idx = torch.tensor(valid_positions)
    target_ids = torch.tensor([targets[p] for p in valid_positions])

    # Captured activations always come back on CPU (see _worker_ext.py); do
    # the actual math on whichever device the unembedding matrix lives on.
    device = final_norm_weight.device

    residual = final_residual[pos_idx].float().to(device)  # (n_valid, hidden)
    rms = (residual.pow(2).mean(dim=-1) + final_norm_eps).sqrt()  # (n_valid,)
    scale = final_norm_weight.float() / rms.unsqueeze(-1)  # (n_valid, hidden)

    contribs = (
        head_contributions[:, :, pos_idx, :].float().to(device)
    )  # (L, H, n_valid, hidden)
    scaled = contribs * scale  # broadcasts over (L, H)

    target_dirs = unembed_weight[target_ids.to(device)].float()  # (n_valid, hidden)
    per_position = torch.einsum("lhvd,vd->lhv", scaled, target_dirs)
    return per_position.sum(dim=-1), len(valid_positions)


def compute_induction_scores(
    llm: LLM,
    token_id_seqs: list[list[int]],
    n_layers: int,
    final_norm_weight: torch.Tensor,
    final_norm_eps: float,
    unembed_weight: torch.Tensor,
) -> torch.Tensor:
    """Average per-(layer, head) direct logit attribution across sequences."""
    activations = capture_induction_activations(llm, token_id_seqs, n_layers)

    total_score: torch.Tensor | None = None
    total_positions = 0
    for token_ids, act in zip(token_id_seqs, activations):
        targets = induction_targets(token_ids)
        final_residual = act["residual_stream"][0]  # (total_pos, hidden)
        head_contribs = act[
            "head_contributions"
        ]  # (n_layers, n_heads, total_pos, hidden)

        score_sum, n_valid = direct_logit_attribution(
            final_residual,
            head_contribs,
            targets,
            final_norm_weight,
            final_norm_eps,
            unembed_weight,
        )
        total_score = score_sum if total_score is None else total_score + score_sum
        total_positions += n_valid

    assert total_score is not None and total_positions > 0
    return total_score / total_positions


# ============================================================
# Reporting
# ============================================================


def _rank_table(scores: torch.Tensor, title: str, k: int = 12) -> Table:
    n_heads = scores.shape[1]
    flat = scores.flatten()
    top = torch.topk(flat, min(k, flat.numel()))
    table = Table(title=title)
    table.add_column("Rank", justify="right")
    table.add_column("Layer", justify="right")
    table.add_column("Head", justify="right")
    table.add_column("Score", justify="right")
    for rank, (idx, val) in enumerate(
        zip(top.indices.tolist(), top.values.tolist()), 1
    ):
        layer, head = idx // n_heads, idx % n_heads
        table.add_row(str(rank), str(layer), str(head), f"{val:.4f}")
    return table


def correlate_with_ground_truth(
    dla_scores: torch.Tensor, ground_truth_scores: torch.Tensor, k: int = 12
) -> tuple[float, float]:
    """Spearman rank correlation + top-k overlap fraction against ground truth.

    Computed as the Pearson correlation of ranks (``pandas``' own
    ``method="spearman"`` shells out to ``scipy``, an otherwise-unneeded
    dependency for this one call).
    """
    dla_flat = pd.Series(dla_scores.flatten().tolist())
    gt_flat = pd.Series(ground_truth_scores.flatten().tolist())
    spearman = float(dla_flat.rank().corr(gt_flat.rank()))

    n_heads = dla_scores.shape[1]

    def top_k_set(scores: torch.Tensor) -> set[tuple[int, int]]:
        flat = scores.flatten()
        top = torch.topk(flat, min(k, flat.numel())).indices.tolist()
        return {(idx // n_heads, idx % n_heads) for idx in top}

    dla_top = top_k_set(dla_scores)
    gt_top = top_k_set(ground_truth_scores)
    overlap = len(dla_top & gt_top) / k
    return spearman, overlap


# ============================================================
# Main
# ============================================================


def main(device: int = 0, compare_to_ground_truth: bool = True) -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    n_layers = AutoConfig.from_pretrained(MODEL_NAME).num_hidden_layers

    token_id_seqs = build_random_token_id_seqs(
        MODEL_NAME, SEQ_LEN, N_SEQUENCES, SEED_BASE
    )

    gamma, eps, unembed = get_final_norm_and_unembed(MODEL_NAME)

    llm = LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3)
    try:
        scores = compute_induction_scores(
            llm, token_id_seqs, n_layers, gamma, eps, unembed
        )
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    console = Console()
    console.print(_rank_table(scores, "vllm-lens OV-circuit DLA induction scores"))

    if compare_to_ground_truth:
        from pathlib import Path

        ground_truth_path = (
            Path(__file__).parent
            / "tests"
            / "utils"
            / "induction_ground_truth_qwen2.5-0.5b-instruct.pt"
        )
        if ground_truth_path.exists():
            gt = torch.load(ground_truth_path, weights_only=True)
            spearman, overlap = correlate_with_ground_truth(scores, gt["scores"])
            console.print(
                f"\nSpearman correlation vs. attention-pattern ground truth: "
                f"{spearman:.3f}"
            )
            console.print(f"Top-12 head overlap vs. ground truth: {overlap:.1%}")
            console.print(
                "\n[dim]Note: direct logit attribution (what a head writes) and "
                "attention patterns (what a head reads) are related but distinct "
                "signals -- a head can attend correctly without its output being "
                "directly logit-relevant if later layers do more processing "
                "first. A moderate (not near-1.0) correlation here is expected; "
                "see _induction_ground_truth.py's GPT-2-small sanity check, where "
                "this same DLA method independently recovers the exact heads "
                "documented in prior interpretability work, for evidence the "
                "method itself is sound.[/dim]"
            )
        else:
            console.print(
                f"\n[yellow]No ground truth found at {ground_truth_path}; "
                "run `python -m vllm_lens._examples._induction_ground_truth` first."
            )


app = typer.Typer()


@app.command()
def cli(
    device: Annotated[int, typer.Option(help="CUDA device index to use.")] = 0,
    compare_to_ground_truth: Annotated[
        bool,
        typer.Option(help="Correlate against the saved TransformerLens ground truth."),
    ] = True,
) -> None:
    main(device=device, compare_to_ground_truth=compare_to_ground_truth)


if __name__ == "__main__":
    app()
