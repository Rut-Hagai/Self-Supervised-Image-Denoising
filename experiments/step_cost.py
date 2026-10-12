"""Seconds per training step for each method, with the settings of the main
experiment and no evaluation in between. (The totals in the training logs
include evaluation and pauses, so they cannot be used for this.)

    python -m experiments.step_cost --steps 200
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.common import build_net
from src.data import CleanPatches, torch_noise
from src.strategies import build

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["supervised", "noise2noise", "noise2void", "neighbor2neighbor",
         "blind2unblind"]


def measure(method, args):
    torch.manual_seed(args.seed)
    net = build_net(args.n_feature)
    strategy = build(method)
    noise = torch_noise(args.noise)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    data = CleanPatches(os.path.join(HERE, args.train_dir), patch=args.patch,
                        length=(args.steps + args.warmup) * args.batch,
                        seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch, num_workers=0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()
    t0 = None
    for step, clean in enumerate(loader, start=1):
        if step == args.warmup + 1:  # start timing after the warm-up steps
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss = strategy.step(net, clean, noise, gen, 0.5)
        loss.backward()
        opt.step()
    return (time.time() - t0) / args.steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--n-feature", type=int, default=16)
    ap.add_argument("--noise", default="gauss25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-dir", default="data/BSD500_train")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    path = os.path.join(HERE, "results", "step_cost.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out.setdefault("settings", vars(args))
    out.setdefault("seconds_per_step", {})
    for m in (args.only or ORDER):
        out["seconds_per_step"][m] = round(measure(m, args), 4)
        print(f"{m:20s} {out['seconds_per_step'][m]:.4f} s/step", flush=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(out, open(path, "w"), indent=2)

    base = out["seconds_per_step"].get("supervised")
    if base:
        print()
        for m, v in out["seconds_per_step"].items():
            print(f"{m:20s} x{v / base:.2f}")
    return out


if __name__ == "__main__":
    main()
