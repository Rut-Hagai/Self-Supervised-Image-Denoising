"""Experiment A: does a noisy target train as well as a clean one?

Two runs that differ only in the target: the clean image (supervised) or a
second, independent noisy image (Noise2Noise). Same initial weights, crops,
noise draws, optimiser and seed.

    python -m experiments.exp_a_noise2noise_theorem --steps 3000
"""
import argparse
import copy
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.common import build_net, count_params, load_images
from src.data import CleanPatches, torch_noise
from src.strategies import build
from src.train import evaluate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_arm(method, init_state, args, test_images):
    """One training run, starting from the given initial weights."""
    net = build_net(args.n_feature)
    net.load_state_dict(copy.deepcopy(init_state))
    strategy = build(method)
    noise = torch_noise(args.noise)

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    data = CleanPatches(os.path.join(HERE, args.train_dir), patch=args.patch,
                        length=args.steps * args.batch, seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch, num_workers=0)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    curve, t0 = [], time.time()
    net.train()
    for step, clean in enumerate(loader, start=1):
        opt.zero_grad(set_to_none=True)
        loss = strategy.step(net, clean, noise, gen, step / args.steps)
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            p = evaluate(net, test_images, args.noise)
            curve.append({"step": step, "psnr": round(p, 4)})
            print(f"  [{method:11s}] step {step:5d}  loss {loss.item():.6f}"
                  f"  PSNR {p:.3f} dB  ({time.time() - t0:.0f}s)", flush=True)
            net.train()
    return net, curve, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--n-feature", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", default="gauss25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--train-dir", default="data/BSD500_train")
    ap.add_argument("--test-dir", default="data/Kodak")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    test_images = load_images(os.path.join(HERE, args.test_dir))
    torch.manual_seed(args.seed)
    init = build_net(args.n_feature).state_dict()
    print(f"both arms start from the same {count_params(build_net(args.n_feature)):,}"
          f" parameters, {len(test_images)} test images, noise {args.noise}")

    out = {"settings": vars(args), "arms": {}}
    for method in ("supervised", "noise2noise"):
        _, curve, secs = run_arm(method, init, args, test_images)
        out["arms"][method] = {"curve": curve, "seconds": round(secs, 1),
                               "final_psnr": curve[-1]["psnr"]}

    a = out["arms"]["supervised"]["final_psnr"]
    b = out["arms"]["noise2noise"]["final_psnr"]
    out["gap_db"] = round(a - b, 4)
    print(f"\nsupervised  {a:.3f} dB")
    print(f"Noise2Noise {b:.3f} dB")
    print(f"gap         {a - b:+.3f} dB")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "exp_a.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    plot(out)
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
    for method, style in (("supervised", dict(color="#1f4e79", lw=2.0)),
                          ("noise2noise", dict(color="#c0504d", lw=2.0,
                                               ls="--"))):
        c = out["arms"][method]["curve"]
        ax.plot([p["step"] for p in c], [p["psnr"] for p in c],
                label=method, **style)
    ax.set_xlabel("training step")
    ax.set_ylabel("PSNR on the test set (dB)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(HERE, "results", "exp_a.png")
    fig.savefig(path)
    print(f"figure written to {path}")


if __name__ == "__main__":
    main()
