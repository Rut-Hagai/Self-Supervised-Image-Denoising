"""How much does the reported PSNR depend on the noise draw?

Re-evaluates the same trained weights under several noise seeds and reports
the spread. A gap between methods smaller than this spread is not meaningful.

    python -m experiments.eval_noise_spread --threads 2
"""
import argparse
import json
import os

import numpy as np
import torch

from src.common import load_images, load_net
from src.train import evaluate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="*", default=["supervised", "noise2void"])
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[101, 202, 303, 404, 505])
    ap.add_argument("--noise", default="gauss25")
    ap.add_argument("--n-feature", type=int, default=16)
    ap.add_argument("--test-dir", default="data/Kodak")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    images = load_images(os.path.join(HERE, args.test_dir))
    out = {"settings": vars(args), "per_method": {}}
    for name in args.methods:
        path = os.path.join(HERE, "weights", f"{name}_{args.noise}.pth")
        if not os.path.exists(path):
            print(f"{name}: no checkpoint at {path}")
            continue
        net = load_net(path, n_feature=args.n_feature)
        vals = []
        for seed in args.seeds:
            p = evaluate(net, images, args.noise, seed=seed)
            vals.append(p)
            print(f"  {name:14s} seed {seed:4d}  {p:.4f} dB", flush=True)
        a = np.array(vals)
        out["per_method"][name] = {"values": [round(v, 4) for v in vals],
                                   "mean": round(float(a.mean()), 4),
                                   "spread": round(float(a.max() - a.min()), 4),
                                   "sd": round(float(a.std(ddof=1)), 4)}
        print(f"{name:14s} mean {a.mean():.4f}   spread {a.max() - a.min():.4f}"
              f"   sd {a.std(ddof=1):.4f}\n")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "eval_noise_spread.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    main()
