"""CPU-only unit tests for the J++ lens pieces (``examples/jpp_lens.py``)."""

import torch

import jacobian_lens
from jpp_lens import (
    Pairs,
    assign,
    candidate_ids,
    combine_experts,
    experts_from_sums,
    fit_expert_weights,
    fit_router,
    hit_margins,
    is_correct,
    non_semantic_token_ids,
    readout_position,
)

D = 16


def _clustered_activations(n_per=200, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(k, D, generator=g) * 4
    labels = torch.arange(k).repeat_interleave(n_per)
    # Vary the norm by position so only the *direction* identifies the cluster.
    scale = torch.linspace(1, 10, k * n_per)[:, None]
    x = (centers[labels] + 0.3 * torch.randn(k * n_per, D, generator=g)) * scale
    return x, labels


def test_router_recovers_clusters_and_roundtrips(tmp_path):
    x, labels = _clustered_activations()
    router = fit_router({5: x}, k=3, proj_dim=4)
    who = assign(router, 5, x)
    # Each true cluster maps to exactly one centroid (a permutation of labels).
    assert all(who[labels == c].unique().numel() == 1 for c in range(3))
    assert who.unique().numel() == 3
    # Batched input keeps its leading shape; saved state loads with weights_only.
    assert assign(router, 5, x[:6].reshape(2, 3, D)).shape == (2, 3)
    torch.save(router, tmp_path / "router.pt")
    loaded = torch.load(tmp_path / "router.pt", weights_only=True)
    assert torch.equal(assign(loaded, 5, x), who)


def test_experts_from_sums_fallback_and_combine():
    sums = torch.arange(3 * 2 * 2, dtype=torch.float).reshape(3, 2, 2)
    counts = torch.tensor([4, 1, 100])
    pooled = torch.full((2, 2), -1.0)
    experts, fallback = experts_from_sums(sums, counts, pooled, min_kept=2)
    assert fallback.tolist() == [False, True, False]
    assert torch.equal(experts[0], sums[0] / 4)
    assert torch.equal(experts[1], pooled)
    assert torch.equal(experts[2], sums[2] / 100)
    w = torch.tensor([0.5, 0.0, -1.0])
    assert torch.allclose(combine_experts(experts, w), 0.5 * experts[0] - experts[2])


class _Tok:
    """A toy tokenizer: single-character vocab, ``\\n`` and digits included."""

    vocab = [" ", ".", "\n", "a", "B", "7", "-", "é"]

    def __len__(self):
        return len(self.vocab)

    def batch_decode(self, ids):
        return [self.vocab[i[0]] for i in ids]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def __call__(self, s, add_special_tokens=True):
        ids = [self.vocab.index(c) for c in s if c in self.vocab]
        return type("Enc", (), {"input_ids": ids if len(ids) == len(s) else [0, 0]})()


def test_non_semantic_token_ids_and_padding():
    ids = non_semantic_token_ids(_Tok(), vocab_size=10)
    # whitespace / punctuation / newline excluded, letters (incl. é) and digits kept,
    # LM-head padding rows beyond the tokenizer excluded.
    assert ids == [0, 1, 2, 6, 8, 9]


def test_candidate_ids_readout_position_and_correctness():
    tok = _Tok()
    assert candidate_ids(tok, "a") == [
        3
    ]  # "a", "A"(absent), " a"(two tokens) -> just "a"
    assert candidate_ids(tok, "ab") == []  # no single-token surface
    ids = tok("aB\na7").input_ids
    assert readout_position(ids, tok, "poetry") == 2
    assert readout_position(ids, tok, "multihop") == len(ids) - 1
    assert is_correct(" Atl", "Atlantic") and not is_correct("  ", "x")
    assert not is_correct("Pac", "Atlantic")


def _pairs(n_items, cands, device="cpu"):
    width = max(map(len, cands))
    return Pairs(
        torch.arange(n_items, device=device),
        torch.tensor([c + [c[0]] * (width - len(c)) for c in cands], device=device),
        torch.zeros(n_items, dtype=torch.long, device=device),
        1,
    )


def test_hit_margin_matches_rank_rule():
    torch.manual_seed(0)
    logits = torch.randn(5, 50)
    cands = [[3], [7, 8], [10, 11, 12], [1], [49]]
    pairs = _pairs(5, cands)
    excluded = torch.zeros(50, dtype=torch.bool)
    excluded[20:30] = True
    for k in (1, 3, 10):
        margin, _ = hit_margins(logits, pairs, excluded, k)
        for p, c in enumerate(cands):
            masked = logits[p].masked_fill(excluded, float("-inf"))
            rank = 1 + int((masked > masked[c].max()).sum())
            assert (margin[p] >= 0).item() == (rank <= k)


def test_fit_expert_weights_prefers_the_good_expert():
    torch.manual_seed(0)
    n_items, n_exp, vocab = 12, 4, 100
    cands = [[i] for i in range(n_items)]
    logits = torch.randn(n_items, n_exp, vocab)
    for p in range(n_items):
        logits[p, 1, p] += 20.0  # expert 1 ranks every candidate first ...
        logits[p, [0, 2, 3], 50 + p] += 30.0  # ... the others push a distractor
    res = fit_expert_weights(logits, _pairs(n_items, cands), None, k=3, steps=20)
    assert res["train_pass"] == 1.0
    assert res["single"][1] == 1.0 and max(res["single"][:1] + res["single"][2:]) < 1.0
    w = res["weights"]
    assert w.argmax() == 1 and abs(float(w.norm()) - 1.0) < 1e-5


def test_load_lens_reads_both_formats(tmp_path):
    j = {4: torch.eye(3, dtype=torch.float16)}
    torch.save({"J": j, "source_layers": [4], "d_model": 3}, tmp_path / "ours.pt")
    torch.save(
        {"parameters": {"jacobians": j}, "source_layers": [4], "config": {}},
        tmp_path / "jpp.pt",
    )
    for name in ("ours.pt", "jpp.pt"):
        jac, layers = jacobian_lens.load_lens(str(tmp_path / name))
        assert layers == [4] and jac[4].dtype == torch.float32
