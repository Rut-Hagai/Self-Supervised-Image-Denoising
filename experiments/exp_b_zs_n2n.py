"""ZS-N2N on single images, against classical filters that need no training.

A new network is trained on each test image and discarded afterwards, with
the paper's settings. Not reported in the seminar paper.

    python -m experiments.exp_b_zs_n2n --images 5
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, estimate_sigma
from skimage.filters import gaussian

from src.common import load_images, save_image
from src.metrics import psnr, ssim, to_uint8
from src.noise import NoiseModel
from src.zs_n2n import denoise_image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classical(noisy, sigma_true):
    """Four filters with no training. BM3D is given the true noise level."""
    import bm3d
    sigma = float(np.mean(estimate_sigma(noisy, channel_axis=2)))
    out = {}
    t = time.time()
    out["gaussian blur"] = (gaussian(noisy, sigma=1.0, channel_axis=2),
                            time.time() - t)
    t = time.time()
    out["total variation"] = (denoise_tv_chambolle(noisy, weight=0.1,
                                                   channel_axis=2),
                              time.time() - t)
    t = time.time()
    out["non-local means"] = (denoise_nl_means(noisy, h=1.15 * sigma,
                                               sigma=sigma, fast_mode=True,
                                               patch_size=5, patch_distance=6,
                                               channel_axis=2),
                              time.time() - t)
    t = time.time()
    out["BM3D"] = (bm3d.bm3d_rgb(noisy, sigma_true), time.time() - t)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=int, default=5)
    ap.add_argument("--crop", type=int, default=256,
                    help="centre crop, as the paper evaluates")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--noise", default="gauss25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-dir", default="data/Kodak")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    images = load_images(os.path.join(HERE, args.test_dir))[:args.images]
    model = NoiseModel(args.noise)

    # results are cached per image, so an interrupted run can be resumed
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    cache_path = os.path.join(HERE, "results", "exp_b_per_image.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    for i, full in enumerate(images):
        H, W, _ = full.shape
        c = args.crop
        x = full[H // 2 - c // 2:H // 2 + c // 2, W // 2 - c // 2:W // 2 + c // 2]
        y = model(x, np.random.default_rng(args.seed + i))
        row = cache.get(str(i), {})

        row["noisy input"] = (psnr(to_uint8(x), to_uint8(y)),
                              ssim(to_uint8(x), to_uint8(y)), 0.0)
        for name, (est, secs) in classical(y, model.sigma).items():
            row[name] = (psnr(to_uint8(x), to_uint8(est)),
                         ssim(to_uint8(x), to_uint8(est)), secs)

        if "ZS-N2N" in row:
            print(f"image {i + 1}/{len(images)}: ZS-N2N already measured,"
                  f" {row['ZS-N2N'][0]:.2f} dB", flush=True)
        else:
            est, info = denoise_image(y, iterations=args.iterations,
                                      clean_hwc=x, seed=args.seed)
            row["ZS-N2N"] = (info["psnr"], info["ssim"], info["seconds"])
            print(f"image {i + 1}/{len(images)}: noisy {info['psnr_noisy']:.2f}"
                  f" dB -> ZS-N2N {info['psnr']:.2f} dB ({info['seconds']:.0f}s,"
                  f" {info['params']:,} parameters)", flush=True)
            if args.save_images:
                d = os.path.join(HERE, "results", "exp_b")
                os.makedirs(d, exist_ok=True)
                save_image(x, os.path.join(d, f"{i:02d}_clean.png"))
                save_image(y, os.path.join(d, f"{i:02d}_noisy.png"))
                save_image(est, os.path.join(d, f"{i:02d}_zsn2n.png"))

        cache[str(i)] = row
        json.dump(cache, open(cache_path, "w"), indent=2)

    rows = {}
    for i in range(len(images)):
        for name, v in cache[str(i)].items():
            rows.setdefault(name, []).append(v)

    print(f"\nmean over {len(images)} images, {args.noise}, "
          f"{args.crop}x{args.crop} centre crops")
    print(f"{'method':18s} {'PSNR':>8s} {'SSIM':>8s} {'seconds':>9s}")
    summary = {}
    for name, vals in rows.items():
        p = float(np.mean([v[0] for v in vals]))
        s = float(np.mean([v[1] for v in vals]))
        t = float(np.mean([v[2] for v in vals]))
        summary[name] = {"psnr": round(p, 3), "ssim": round(s, 4),
                         "seconds": round(t, 2)}
        print(f"{name:18s} {p:8.2f} {s:8.4f} {t:9.1f}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "exp_b.json"), "w") as fh:
        json.dump({"settings": vars(args), "summary": summary}, fh, indent=2)
    return summary


if __name__ == "__main__":
    main()
