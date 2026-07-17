"""Repeated-random-token induction test sequences.

Pure token-id utilities with no framework dependency (no vLLM, no
TransformerLens), so the exact same sequences -- byte-for-byte identical
token ids -- can be fed to vLLM (via vllm-lens) and to TransformerLens for
ground-truth comparison without any risk of divergent tokenization.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence


def make_induction_sequence(
    vocab_size: int,
    seq_len: int,
    seed: int,
    exclude_token_ids: Collection[int] = (),
    bos_token_id: int | None = None,
) -> list[int]:
    """Build a repeated-random-token induction test sequence.

    Constructs ``random_ids + random_ids`` (Olsson et al. 2022's induction
    test): ``seq_len`` random token ids, sampled without replacement from
    ``range(vocab_size)`` excluding ``exclude_token_ids`` (e.g. a
    tokenizer's special tokens), followed by an exact repeat of the same
    ids. Sampling *without* replacement avoids accidental duplicate tokens
    within the first half, which would otherwise create a second, unintended
    "earlier occurrence" for some tokens and muddy the induction signal.

    Random (not natural-language) tokens are used deliberately: with
    natural text, correct next-token prediction after a repeated token
    could be explained by syntax or semantic priors as well as by a genuine
    copying/induction circuit, confounding the measurement. With tokens
    drawn independently at random from a large vocabulary, predicting the
    second half correctly has no plausible mechanism other than an
    induction circuit that finds the earlier occurrence and copies forward
    whatever followed it.

    If ``bos_token_id`` is given, it is prepended once at position 0,
    outside the repeated region -- it never recurs, so it never
    contributes an induction target.
    """
    pool = [t for t in range(vocab_size) if t not in exclude_token_ids]
    if len(pool) < seq_len:
        raise ValueError(
            f"vocab_size={vocab_size} minus excluded tokens leaves only "
            f"{len(pool)} candidates, fewer than seq_len={seq_len}"
        )
    rng = random.Random(seed)
    random_ids = rng.sample(pool, seq_len)
    token_ids = random_ids + random_ids

    if bos_token_id is not None:
        token_ids = [bos_token_id, *token_ids]
    return token_ids


def induction_targets(token_ids: Sequence[int]) -> list[int | None]:
    """Generic induction targets for any token sequence.

    A causal LM's hidden state at position ``p`` is used to predict the
    token at position ``p + 1``. An induction head at query position ``p``
    looks for the most recent earlier position ``j < p`` with
    ``token_ids[j] == token_ids[p]``, and copies forward
    ``token_ids[j + 1]`` -- so the induction-correct target for position
    ``p`` is ``token_ids[j + 1]`` if such a ``j`` exists, else ``None``.

    Example: ``[5, 9, 2, 5, 9]`` -> targets ``[None, None, None, 9, 2]``.
    Position 3's token (``5``) last occurred at ``j=0``, so its target is
    ``token_ids[1] == 9``; position 4's token (``9``) last occurred at
    ``j=1``, so its target is ``token_ids[2] == 2``. This is generic (works
    on any sequence, not just the synthetic ones from
    ``make_induction_sequence``), but the signal it measures is only clean
    on sequences built to avoid confounding priors -- see that function's
    docstring.
    """
    last_seen: dict[int, int] = {}
    targets: list[int | None] = []
    for p, tok in enumerate(token_ids):
        j = last_seen.get(tok)
        targets.append(token_ids[j + 1] if j is not None else None)
        last_seen[tok] = p
    return targets
