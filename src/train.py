"""One training loop for every method. Only the strategy changes.

    python -m src.train --method neighbor2neighbor --steps 2000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import build_net, count_params, denoise, load_images
from .data import CleanPatches, torch_noise
from .metrics import psnr, to_uint8
from .noise import NoiseModel
from .strategies import REGISTRY, build


def evaluate(net, images, noise_style, seed=101, device="cpu"):
    """Mean PSNR over a test set, using one fixed noise draw per image."""
    model = NoiseModel(noise_style)
    rng = np.random.default_rng(seed)
    out = []
    for im in images:
        noisy = model(im, rng)
        pred = denoise(net, noisy, device)
        out.append(psnr(to_uint8(im), to_uint8(pred)))
    return float(np.mean(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--noise", default="gauss25")
    ap.add_argument("--train-dir", default="data/BSD500_train")
    ap.add_argument("--test-dir", default="data/Kodak")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--n-feature", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="save a resumable checkpoint every N steps")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint if one exists")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    strategy = build(args.method)
    net = build_net(args.n_feature, device)
    noise = torch_noise(args.noise)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    data = CleanPatches(args.train_dir, patch=args.patch,
                        length=args.steps * args.batch, seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch, num_workers=0)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    test_images = load_images(args.test_dir) if args.test_dir else []
    print(f"method={args.method}  noise={args.noise}  device={device}  "
          f"params={count_params(net):,}  steps={args.steps}")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.method}_{args.noise}"
    ckpt_path = os.path.join(args.out, f"{tag}.ckpt")
    log, t0 = [], time.time()

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].to(gen.get_state().device)
                      if hasattr(ck["gen"], "to") else ck["gen"])
        start_step, log = ck["step"], ck["log"]
        print(f"  resumed from step {start_step}", flush=True)

    net.train()
    for step, clean in enumerate(loader, start=1):
        if step <= start_step:  # skip to where the checkpoint left off
            continue
        clean = clean.to(device)
        progress = step / args.steps
        opt.zero_grad(set_to_none=True)
        loss = strategy.step(net, clean, noise, gen, progress)
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  loss {loss.item():.6f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if args.eval_every and step % args.eval_every == 0 and test_images:
            p = evaluate(net, test_images, args.noise, device=device)
            log.append({"step": step, "psnr": p})
            print(f"  >> PSNR on test set: {p:.3f} dB", flush=True)
            net.train()
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "gen": gen.get_state(),
                        "step": step, "log": log}, ckpt_path)

    torch.save(net.state_dict(), os.path.join(args.out, f"{tag}.pth"))
    result = {"method": args.method, "noise": args.noise, "steps": args.steps,
              "params": count_params(net), "seconds": round(time.time() - t0, 1),
              "log": log}
    if test_images:
        result["final_psnr"] = evaluate(net, test_images, args.noise, device=device)
        print(f"final PSNR on {args.test_dir}: {result['final_psnr']:.3f} dB")
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    main()
