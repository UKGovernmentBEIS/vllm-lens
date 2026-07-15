"""Make a synthetic Jacobian lens (random J_l) matching a model's shape.

For *parallel-equivalence* testing only: the lens quality is irrelevant, we just
need the same J_l applied identically across TP/PP/EP configs. Skips the
expensive backward-pass fit on large MoE models. Deterministic (fixed seed) so
the same file is reused for every config.

    python examples/_make_synthetic_lens.py --model deepseek-ai/DeepSeek-V2-Lite \
        --out /tmp/ds2lite-synth-lens.pt
"""

from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated layers only (keeps the file small for large models; "
        "default = all source layers)",
    )
    args = ap.parse_args()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    d = cfg.hidden_size
    n = cfg.num_hidden_layers
    if args.layers:
        source_layers = [int(x) for x in args.layers.split(",")]
    else:
        source_layers = list(range(n - 1))

    g = torch.Generator().manual_seed(0)
    eye = torch.eye(d)
    jac = {}
    for lyr in source_layers:
        # I + small noise: well-conditioned, non-trivial transport.
        jac[lyr] = (eye + args.noise * torch.randn(d, d, generator=g)).to(torch.float16)

    torch.save({"J": jac, "source_layers": source_layers, "d_model": d}, args.out)
    print(f"saved synthetic lens -> {args.out} ({len(jac)} layers, d_model={d})")


if __name__ == "__main__":
    main()
