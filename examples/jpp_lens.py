"""J++ lens: Jacobian filtering + readout filtering on top of the J-/R-lens.

Reference: Ayonrinde & Lindsey, "J++ Lens: Jacobian Filtering Enables More
Faithful Workspace Readouts" (https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr);
reference implementation https://github.com/safety-research/jpp_lens (Apache-2.0).

The J-lens averages one Jacobian ``dh_final/dh_l`` over every source position
with equal weight, so a biased minority (positions where the linearisation is
poor) corrupts the mean. The J++ lens makes three changes:

1. **Jacobian filtering** (the main one): k-means partitions each layer's
   activation space (K=8 clusters, on a 64-dim PCA of unit-normalised
   activations); one *expert* Jacobian is averaged per cluster; a signed
   weight per expert, learned on a small dev split of readout items (the only
   supervised step), collapses the experts back into one ``d_model x d_model``
   map — a drop-in for ``J_l`` with no extra inference cost.
2. The LRP backward of the R-lens (``jacobian_lens_fit.py --rules lrp``).
3. **Readout filtering**: drop vocab tokens with no letter or digit (whitespace,
   punctuation) from the ranking.

Pipeline — the middle step is the ordinary fit; the other two run against a
served model in the vllm-lens env (forward passes only)::

    # (1) cluster each source layer's residual activations
    python examples/jpp_lens.py fit-router --layers 8,16,24 --out router.pt

    # (2) fit per-cluster expert Jacobians (prime-rl fit env, see jacobian_lens_fit.py)
    uv run --no-sync torchrun --nproc-per-node=8 examples/jacobian_lens_fit.py \\
        --model Qwen/Qwen3-1.7B --rules lrp --router router.pt --out experts.pt

    # (3) learn the expert weights on the readout items and write the lens
    git clone --depth 1 https://github.com/safety-research/jpp_lens /tmp/jpp
    python examples/jpp_lens.py fit-weights --experts experts.pt \\
        --items-dir /tmp/jpp/data/jlens/evaluations --out jpp-lens.pt

``jpp-lens.pt`` is an ordinary lens file (key ``J``), so ``jacobian_lens.py``
and ``jacobian_lens_chat.py`` read it as-is; pass ``--readout-filter`` there
for change (3). ``experts.pt`` is itself a valid (pooled) lens too.

Top-level imports are torch + stdlib only, so ``jacobian_lens_fit.py`` can
import the router from the fit env.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

MACRO_EVALS = ("multihop", "multilingual", "typo", "association", "poetry")
# poetry reads out at the last newline token (end of the couplet's first
# line); every other eval at the final prompt token.
LAST_NEWLINE_EVALS = {"poetry"}
# Greedy-correctness grading: multihop/multilingual against ``target``, poetry
# against its first intermediate; association/typo pose no question.
CORRECTNESS_TARGET = {
    "multihop": "target",
    "multilingual": "target",
    "poetry": "first_intermediate",
}

# ───────────────────────── routing (expert assignment) ─────────────────────────


def _kmeans(x, k, *, seed=0, iters=100, tol=1e-6):
    """Lloyd's k-means with k-means++ init on ``x [P, D]`` -> centroids ``[K, D]``."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    cents = [x[torch.randint(len(x), (1,), generator=g, device=x.device)].squeeze(0)]
    for _ in range(k - 1):
        d2 = torch.cdist(x, torch.stack(cents)).pow(2).amin(1)
        cents.append(
            x[torch.multinomial(d2.clamp(min=1e-12), 1, generator=g)].squeeze(0)
        )
    cents = torch.stack(cents)
    for _ in range(iters):
        dist = torch.cdist(x, cents)
        who = dist.argmin(1)
        new = torch.stack(
            [
                x[who == c].mean(0)
                if (who == c).any()
                # Empty cluster: re-seed to the point farthest from its centroid.
                else x[dist.gather(1, who[:, None]).argmax()]
                for c in range(k)
            ]
        )
        shift = (new - cents).norm(dim=1).max()
        cents = new
        if shift < tol:
            break
    return cents


def fit_router(acts, *, k=8, proj_dim=64, seed=0, device="cpu"):
    """Cluster each layer's activations ``{layer: [P, d_model]}`` into a router.

    Per layer: unit-normalise (cluster on *direction* — residual norms grow
    with position and would otherwise dominate the PCA), centre, project onto
    the top ``proj_dim`` principal components, k-means++ in that space. All
    state is plain tensors (``weights_only=True`` loadable).
    """
    layers = {}
    for lyr, x in acts.items():
        x = F.normalize(x.to(device).float(), dim=-1)
        mean = x.mean(0)
        xc = x - mean
        _, vecs = torch.linalg.eigh(xc.T @ xc / max(len(x) - 1, 1))
        comps = vecs[:, -proj_dim:].flip(1)  # descending eigenvalue order
        layers[lyr] = {
            "mean": mean.cpu(),
            "components": comps.cpu(),
            "centroids": _kmeans(xc @ comps, k, seed=seed).cpu(),
        }
    return {"k": k, "proj_dim": proj_dim, "seed": seed, "layers": layers}


def assign(router, layer, h):
    """Nearest-centroid cluster of each activation ``h [..., d_model]`` at ``layer``."""
    r = router["layers"][layer]
    dev = h.device
    z = (F.normalize(h.float(), dim=-1) - r["mean"].to(dev)) @ r["components"].to(dev)
    flat = torch.cdist(z.reshape(-1, z.shape[-1]), r["centroids"].to(dev)).argmin(1)
    return flat.reshape(z.shape[:-1])


def experts_from_sums(expert_sum, counts, pooled, *, min_kept=50):
    """Per-cluster mean Jacobians ``[K, d, d]`` from gradient-row sums and
    position counts. A cluster that kept fewer than ``min_kept`` positions
    falls back to the pooled Jacobian. Returns ``(experts, fallback_mask)``."""
    fallback = counts < min_kept
    experts = expert_sum / counts.clamp(min=1).to(expert_sum)[:, None, None]
    experts[fallback] = pooled.to(experts)
    return experts, fallback


def combine_experts(experts, weights):
    """``sum_e w_e J_e`` — the J++ transport map."""
    return torch.einsum("e,eij->ij", weights.to(experts), experts)


# ───────────────────────────── readout filtering ─────────────────────────────


def non_semantic_token_ids(tokenizer, vocab_size):
    """Readout Filtering: vocab ids whose decoded string has no letter or digit
    (whitespace, punctuation), plus LM-head padding rows beyond the tokenizer."""
    n = min(vocab_size, len(tokenizer))
    strings = tokenizer.batch_decode([[i] for i in range(n)])
    ids = [i for i, s in enumerate(strings) if not any(c.isalnum() for c in s)]
    return ids + list(range(n, vocab_size))


# ──────────────────────────── readout eval items ────────────────────────────


def load_items(items_dir, slugs=MACRO_EVALS):
    """``lens-eval-<slug>.json`` items (anthropics/jacobian-lens data; shipped in
    jpp_lens under ``data/jlens/evaluations``)."""
    items = []
    for slug in slugs:
        raw = json.loads((Path(items_dir) / f"lens-eval-{slug}.json").read_text())
        for it in raw["items"]:
            items.append(
                {
                    "eval": slug,
                    "name": it["name"],
                    "prompt": it["prompt"],
                    "target": it.get("target"),
                    "intermediates": list(it["intermediates"]),
                }
            )
    return items


def candidate_ids(tokenizer, word):
    """Single-token vocab ids of a concept's surface forms (with / without a
    leading space; original, capitalised and lower case), so a lens is never
    penalised for the case or spacing of a readout. Empty when no surface is a
    single token (the intermediate is then unscorable)."""
    forms = {word, word.capitalize(), word.lower()}
    ids = set()
    for s in (p + f for f in forms for p in ("", " ")):
        t = tokenizer(s, add_special_tokens=False).input_ids
        if len(t) == 1:
            ids.add(int(t[0]))
    return sorted(ids)


def readout_position(ids, tokenizer, slug):
    if slug in LAST_NEWLINE_EVALS:
        nl = [i for i, t in enumerate(ids) if "\n" in tokenizer.decode([t])]
        if not nl:
            raise ValueError("last-newline readout: prompt has no newline token")
        return nl[-1]
    return len(ids) - 1


def correctness_target(item):
    src = CORRECTNESS_TARGET.get(item["eval"])
    if src == "target":
        return item["target"]
    return item["intermediates"][0] if src else None


def is_correct(next_token, target):
    """Greedy-correctness rule: the next token is the target or a leading
    subword of it (" Atl" matches "Atlantic")."""
    s = next_token.lstrip()
    return bool(s) and target.strip().startswith(s)


# ───────────────────── rank objective + expert-weight search ─────────────────────


@dataclass
class Pairs:
    """The (item, intermediate) pairs the eval scores."""

    item: torch.Tensor  # [Q] row of the item in the logits
    cands: torch.Tensor  # [Q, C] candidate ids, padded by repeating the first
    group: torch.Tensor  # [Q] eval index, for the macro mean
    n_groups: int


def build_pairs(recs, idx, device):
    slugs = sorted({recs[i]["eval"] for i in idx})
    item, cands, group = [], [], []
    for row, i in enumerate(idx):
        for ids in recs[i]["cands"].values():
            item.append(row)
            cands.append(ids)
            group.append(slugs.index(recs[i]["eval"]))
    if not cands:
        raise ValueError("no scoreable (item, intermediate) pairs")
    width = max(map(len, cands))
    cands = [c + [c[0]] * (width - len(c)) for c in cands]
    return Pairs(
        torch.tensor(item, device=device),
        torch.tensor(cands, device=device),
        torch.tensor(group, device=device),
        len(slugs),
    )


def hit_margins(logits, pairs, excluded, k):
    """Per pair: ``best candidate logit - k-th largest logit`` (>= 0 exactly
    when the eval's rank is <= k; excluded ids at -inf, candidates never
    excluded) and the item's logit spread (the surrogate's scale)."""
    spread = logits.std(1).detach()[pairs.item]
    if excluded is not None:
        logits = logits.masked_fill(excluded[None], float("-inf"))
    lq = logits[pairs.item]
    best = lq.gather(1, pairs.cands).amax(1)
    kth = lq.topk(k, dim=1).values[:, -1]
    return best - kth, spread


def macro_mean(v, pairs):
    """Mean of ``v [Q]`` within each eval, then the unweighted mean over evals."""
    sums = torch.zeros(pairs.n_groups, device=v.device, dtype=v.dtype)
    cnt = torch.zeros_like(sums)
    sums.index_add_(0, pairs.group, v)
    cnt.index_add_(0, pairs.group, torch.ones_like(v))
    return (sums / cnt).mean()


def fit_expert_weights(
    logits_pev, pairs, excluded, *, k=10, steps=200, lr=0.05, temperature=0.1, top_n=3
):
    """Signed unit-L2 expert weights maximising the layer's macro pass@k.

    ``logits_pev [P, E, V]`` are the per-expert logits ``W_U diag(g) J_e x_p``;
    RMSNorm's ``rsqrt`` factor is a positive per-item scalar that ranks
    ignore, so ``sum_e w_e L_pe`` ranks exactly as the combined lens would, and
    the objective is linear in ``w`` with no further model passes. The exact
    objective is piecewise constant, so Adam runs on a sigmoid surrogate of
    the hit margin (in units of the logit spread) from three starts — uniform,
    the best single expert, the ``top_n`` best — and the iterate with the
    highest exact macro pass wins. Returns a dict with ``weights``,
    ``train_pass``, ``single`` (each expert's stand-alone pass@k), ``init``, ``step``.
    """
    n_exp = logits_pev.shape[1]

    def exact(margin):
        return float(macro_mean((margin >= 0).float(), pairs))

    single = [
        exact(hit_margins(logits_pev[:, e], pairs, excluded, k)[0])
        for e in range(n_exp)
    ]
    order = sorted(range(n_exp), key=lambda e: (-single[e], e))
    starts = {
        "uniform": torch.ones(n_exp),
        "best": F.one_hot(torch.tensor(order[0]), n_exp).float(),
        f"top{top_n}": F.one_hot(torch.tensor(order[:top_n]), n_exp).float().sum(0),
    }
    best = None
    for init, w0 in starts.items():
        w = (w0 / w0.norm()).to(logits_pev.device).requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for step in range(steps + 1):
            u = w / w.norm()
            margin, spread = hit_margins(
                torch.einsum("e,pev->pv", u, logits_pev), pairs, excluded, k
            )
            soft = macro_mean(torch.sigmoid(margin / (temperature * spread)), pairs)
            if step % 5 == 0 or step == steps:
                cand = (
                    exact(margin.detach()),
                    float(soft.detach()),
                    u.detach().cpu(),
                    init,
                    step,
                )
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if step == steps:
                break
            opt.zero_grad()
            (-soft).backward()
            opt.step()
    assert best is not None
    return {
        "weights": best[2],
        "train_pass": best[0],
        "single": single,
        "init": best[3],
        "step": best[4],
    }


# ─────────────────────────────────── CLI ───────────────────────────────────


def load_unembed(model_name):
    """``(W_U [V, d], g [d])``: the LM head and final-norm weights, read straight
    from the checkpoint's safetensors (no GB-sized server round trip).
    ``(x * g) @ W_U.T`` ranks exactly as the model's RMSNorm + head (the norm's
    ``rsqrt`` is a positive per-row scalar). Standard RMSNorm only (a Gemma-style
    ``1 + g`` gain would need adjusting here, as in ``jacobian_lens.py``)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors import safe_open

    local = Path(model_name)

    def fetch(fname):
        if local.is_dir():
            if not (local / fname).exists():
                raise FileNotFoundError(local / fname)
            return str(local / fname)
        return hf_hub_download(model_name, fname)  # honours HF_HUB_OFFLINE / the cache

    try:
        index = json.loads(Path(fetch("model.safetensors.index.json")).read_text())
        where = index["weight_map"]
    except (FileNotFoundError, EntryNotFoundError):  # single-shard checkpoint
        with safe_open(fetch("model.safetensors"), "pt") as sf:
            where = dict.fromkeys(sf.keys(), "model.safetensors")

    def pick(pred):
        name = next((n for n in sorted(where) if pred(n)), None)
        if name is None:
            raise ValueError(f"no tensor matching the predicate in {model_name}")
        with safe_open(fetch(where[name]), "pt") as sf:
            return sf.get_tensor(name).float()

    norm = pick(lambda n: n.endswith(".norm.weight") and ".layers." not in n)
    if any(n.endswith("lm_head.weight") for n in where):
        head = pick(lambda n: n.endswith("lm_head.weight"))
    else:  # tied embeddings
        head = pick(lambda n: n.endswith("embed_tokens.weight"))
    return head, norm


def _web_prompts(path, n, cache):
    """``n`` web-text prompts: a JSONL file of strings, or FineWeb streamed once
    and cached beside the output (the fitter's corpus and cache format)."""
    src = Path(path) if path else Path(cache)
    if not src.exists():
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )
        texts = []
        for row in ds:
            if len(row["text"]) > 500:
                texts.append(row["text"])
                if len(texts) >= n:
                    break
        src.write_text("".join(json.dumps(t) + "\n" for t in texts))
    with open(src) as f:
        return [json.loads(line) for line in f if line.strip()][:n]


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_fit_router(args):
    from transformers import AutoTokenizer

    from vllm_lens.client import VLLMLensClient

    client = VLLMLensClient(args.base_url)
    tok = AutoTokenizer.from_pretrained(client.model)
    layers = [int(x) for x in args.layers.split(",")]
    prompts = _web_prompts(
        args.prompts_file, args.n_prompts, f"{args.out}.prompts.jsonl"
    )
    acts: dict[int, list[torch.Tensor]] = {lyr: [] for lyr in layers}
    for i, text in enumerate(prompts):
        ids = tok(text, truncation=True, max_length=args.max_seq_len).input_ids
        if len(ids) <= args.skip_first + 1:
            continue
        # Token ids as the prompt: exactly the fitter's truncation, no re-tokenization.
        out = client.generate(ids, max_tokens=1, capture_layers=layers)
        res = out.activations["residual_stream"][
            :, args.skip_first : -1
        ]  # the fit's kept positions
        for j, lyr in enumerate(layers):
            acts[lyr].append(res[j].to(torch.float16))
        if (i + 1) % 50 == 0:
            print(f"  captured {i + 1}/{len(prompts)} prompts", flush=True)
    pooled = {lyr: torch.cat(v) for lyr, v in acts.items()}
    print(
        f"clustering {len(next(iter(pooled.values())))} positions x {len(layers)} layers (K={args.k})"
    )
    router = fit_router(
        pooled, k=args.k, proj_dim=args.proj_dim, seed=args.seed, device=_device()
    )
    router["provenance"] = {
        "model": client.model,
        "n_prompts": len(prompts),
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
    }
    torch.save(router, args.out)
    print(
        f"router -> {args.out} (layers {layers}, K={args.k}, proj_dim={args.proj_dim})"
    )


def cmd_fit_weights(args):
    from transformers import AutoTokenizer

    from vllm_lens.client import VLLMLensClient

    ckpt = torch.load(args.experts, map_location="cpu", weights_only=True)
    if "experts" not in ckpt:
        raise ValueError(
            f"{args.experts} has no experts; fit it with jacobian_lens_fit.py --router"
        )
    experts = {int(lyr): v for lyr, v in ckpt["experts"].items()}
    pooled = {int(lyr): v for lyr, v in ckpt["J"].items()}
    layers = sorted(experts)
    dev = _device()

    client = VLLMLensClient(args.base_url)
    tok = AutoTokenizer.from_pretrained(client.model)

    # One forward per item: the greedy next token (correctness filter) and the
    # residuals at the readout position. Items with no single-token candidate
    # surface, or that the model gets wrong, are dropped.
    recs = []
    for it in load_items(args.items_dir):
        cands = {w: c for w in it["intermediates"] if (c := candidate_ids(tok, w))}
        if not cands:
            continue
        ids = tok(it["prompt"]).input_ids
        pos = readout_position(ids, tok, it["eval"])
        out = client.generate(ids, max_tokens=1, capture_layers=layers)
        tgt = correctness_target(it)
        if tgt is not None and not is_correct(out.text, tgt):
            continue
        recs.append(
            {
                **it,
                "cands": cands,
                "x": out.activations["residual_stream"][:, pos].float(),
            }
        )
    print(f"{len(recs)} scoreable items the model gets right")

    # Stratified fit / held-out split: a fixed fraction of each eval's items.
    rng = random.Random(args.seed)
    fit_keys = set()
    for slug in sorted({r["eval"] for r in recs}):
        keys = [(r["eval"], r["name"]) for r in recs if r["eval"] == slug]
        rng.shuffle(keys)
        fit_keys |= set(keys[: max(1, int(args.fit_fraction * len(keys)))])
    fit_idx = [i for i, r in enumerate(recs) if (r["eval"], r["name"]) in fit_keys]
    held_idx = [i for i in range(len(recs)) if i not in set(fit_idx)]

    head, g = load_unembed(client.model)
    head, g = head.to(dev), g.to(dev)
    vocab = head.shape[0]
    excluded = None
    if not args.no_readout_filter:
        excluded = torch.zeros(vocab, dtype=torch.bool)
        excluded[non_semantic_token_ids(tok, vocab)] = True
        for r in recs:  # a scored candidate is never excluded
            for c in r["cands"].values():
                excluded[c] = False
        excluded = excluded.to(dev)

    def unembed(x):  # [..., d] -> [..., V], rank-equivalent to norm + head
        return (x.to(dev) * g) @ head.T

    x_all = torch.stack([r["x"] for r in recs])  # [P, L, d]
    pairs_fit = build_pairs(recs, fit_idx, dev)
    pairs_held = build_pairs(recs, held_idx, dev) if held_idx else None
    weights, combined, held_margins = {}, {}, {"pooled": [], "jpp": []}
    for li, lyr in enumerate(layers):
        exp = experts[lyr].to(dev).float()
        transports = torch.einsum("pj,eij->pei", x_all[fit_idx, li].to(dev), exp)
        res = fit_expert_weights(
            unembed(transports), pairs_fit, excluded, k=args.k, steps=args.steps
        )
        weights[lyr] = res["weights"]
        combined[lyr] = combine_experts(exp, res["weights"].to(dev)).cpu()
        print(
            f"layer {lyr}: single-expert pass@{args.k} {[round(s, 3) for s in res['single']]} "
            f"-> learned {res['train_pass']:.3f} ({res['init']}, step {res['step']}); "
            f"weights {[round(w, 3) for w in res['weights'].tolist()]}"
        )
        if pairs_held is not None:
            for name, j in (("pooled", pooled[lyr]), ("jpp", combined[lyr])):
                logits = unembed(x_all[held_idx, li].to(dev) @ j.to(dev).float().T)
                held_margins[name].append(
                    hit_margins(logits, pairs_held, excluded, args.k)[0]
                )
    if pairs_held is not None:
        for name, ms in held_margins.items():
            hit = (torch.stack(ms).amax(0) >= 0).float()  # min rank over layers <= k
            print(
                f"held-out pass@{args.k} ({len(held_idx)} items, min over layers): "
                f"{name} {float(macro_mean(hit, pairs_held)):.3f}"
            )

    torch.save(
        {
            # fp32: fp16 rounding of the combined map flips near-tie ranks.
            "J": combined,
            "source_layers": layers,
            "d_model": int(ckpt["d_model"]),
            "expert_weights": weights,
            "provenance": {
                **ckpt.get("provenance", {}),
                "jpp": {
                    "items_dir": args.items_dir,
                    "fit_items": sorted(fit_keys),
                    "k": args.k,
                    "steps": args.steps,
                    "seed": args.seed,
                    "readout_filter": not args.no_readout_filter,
                },
            },
        },
        args.out,
    )
    print(f"J++ lens -> {args.out} ({len(layers)} layers)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser(
        "fit-router", help="cluster each layer's activations on a served model"
    )
    r.add_argument("--base-url", default="http://localhost:8000")
    r.add_argument(
        "--layers", required=True, help="comma-separated source layers to cluster"
    )
    r.add_argument("--n-prompts", type=int, default=1000)
    r.add_argument(
        "--prompts-file",
        default=None,
        help="JSONL of prompt strings (default: FineWeb)",
    )
    r.add_argument("--max-seq-len", type=int, default=128)
    r.add_argument(
        "--skip-first", type=int, default=16, help="match the fitter's --skip-first"
    )
    r.add_argument("--k", type=int, default=8, help="clusters (experts) per layer")
    r.add_argument(
        "--proj-dim", type=int, default=64, help="PCA dims the k-means runs in"
    )
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="router.pt")

    w = sub.add_parser(
        "fit-weights", help="learn the expert weights and write the J++ lens"
    )
    w.add_argument("--base-url", default="http://localhost:8000")
    w.add_argument(
        "--experts",
        required=True,
        help="experts .pt from jacobian_lens_fit.py --router",
    )
    w.add_argument(
        "--items-dir", required=True, help="dir of lens-eval-<slug>.json readout items"
    )
    w.add_argument(
        "--fit-fraction", type=float, default=0.2, help="share of each eval fitted on"
    )
    w.add_argument(
        "--k", type=int, default=10, help="pass@k the weights are fitted for"
    )
    w.add_argument(
        "--steps", type=int, default=200, help="Adam steps per starting point"
    )
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--no-readout-filter", action="store_true")
    w.add_argument("--out", default="jpp-lens.pt")

    args = ap.parse_args()
    {"fit-router": cmd_fit_router, "fit-weights": cmd_fit_weights}[args.cmd](args)


if __name__ == "__main__":
    main()
