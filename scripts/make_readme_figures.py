"""The two figures in README.md, written to docs/.

    python -m scripts.make_readme_figures
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs")

ORDER = ["supervised", "noise2noise", "noise2void", "neighbor2neighbor",
         "blind2unblind"]
LABEL = {"supervised": "supervised", "noise2noise": "Noise2Noise",
         "noise2void": "Noise2Void", "neighbor2neighbor": "Neighbor2Neighbor",
         "blind2unblind": "Blind2Unblind"}
# "supervised" alone could be read as a sixth surveyed method
PANEL_LABEL = dict(LABEL, supervised="supervised (reference, not surveyed)",
                   zs_n2n="ZS-N2N")
# Okabe-Ito palette, safe for colour-blind readers
COLOR = {"supervised": "#0072B2", "noise2noise": "#D55E00",
         "noise2void": "#009E73", "neighbor2neighbor": "#E69F00",
         "blind2unblind": "#CC79A7", "zs_n2n": "#56B4E9"}


def curves(path):
    """PSNR against training step, from results/exp_d.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.load(open(os.path.join(HERE, "results", "exp_d.json")))["curves"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ends = []
    for m in ORDER:
        xs = [p["step"] for p in data[m]]
        ys = [p["psnr"] for p in data[m]]
        ax.plot(xs, ys, lw=2, color=COLOR[m], zorder=3)
        ends.append((m, xs[-1], ys[-1]))

    # labels at the line ends, pushed apart where two lines finish close
    ends.sort(key=lambda t: t[2])
    last = None
    for m, x, y in ends:
        yy = y if last is None or y - last > 0.38 else last + 0.38
        last = yy
        ax.annotate(LABEL[m], (x, yy), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10.5, color=COLOR[m],
                    annotation_clip=False)

    best = max(data["blind2unblind"], key=lambda p: p["psnr"])
    ax.plot([best["step"]], [best["psnr"]], marker="o", ms=7, mfc="white",
            mec=COLOR["blind2unblind"], mew=2, zorder=4)
    ax.annotate(f"Blind2Unblind peaks here,\nat step {best['step']:,}",
                (best["step"], best["psnr"]), xytext=(26, -92),
                textcoords="offset points", fontsize=9.5, linespacing=1.4,
                color=COLOR["blind2unblind"],
                arrowprops=dict(arrowstyle="-", lw=1,
                                color=COLOR["blind2unblind"],
                                shrinkA=2, shrinkB=6))

    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("PSNR on Kodak (dB)", fontsize=11)
    ax.grid(True, color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(labelsize=9.5, color="#999999")
    ax.set_xlim(0, 21000)
    ax.set_xticks(range(0, 20001, 4000))
    ax.set_xticklabels([f"{t:,}" for t in range(0, 20001, 4000)])
    fig.subplots_adjust(left=0.085, right=0.76, top=0.96, bottom=0.13)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def panels(path, image=4, crop=256, noise="gauss25", seed=0, n_feature=16,
           zs_iterations=2000):
    """One noisy crop through every method. ZS-N2N has no checkpoint, so it is
    trained here on the crop, which takes a few minutes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.common import load_images, load_net, denoise
    from src.metrics import psnr, to_uint8
    from src.noise import NoiseModel
    from src.zs_n2n import denoise_image

    images = load_images(os.path.join(HERE, "data", "Kodak"))
    im = images[image % len(images)]
    h, w, _ = im.shape
    x = im[h // 2 - crop // 2:h // 2 + crop // 2,
           w // 2 - crop // 2:w // 2 + crop // 2]
    y = NoiseModel(noise)(x, np.random.default_rng(seed))

    cells = [("clean reference", x, None),
             ("noisy input", y, psnr(to_uint8(x), to_uint8(y)))]
    for m in ORDER:
        p = os.path.join(HERE, "weights", f"{m}_{noise}.pth")
        if not os.path.exists(p):
            print("missing", p)
            continue
        est = denoise(load_net(p, n_feature=n_feature), y)
        cells.append((PANEL_LABEL[m], est, psnr(to_uint8(x), to_uint8(est))))

    print(f"  training ZS-N2N on this crop, {zs_iterations} iterations")
    est, info = denoise_image(y, iterations=zs_iterations, clean_hwc=x, seed=seed)
    print(f"  {info['params']:,} parameters, {info['seconds']:.0f} s")
    cells.append((PANEL_LABEL["zs_n2n"], est, psnr(to_uint8(x), to_uint8(est))))

    cols, rows = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.45 * rows), dpi=95)
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img, db) in zip(axes.ravel(), cells):
        ax.imshow(np.clip(img, 0, 1))
        key = {v: k for k, v in PANEL_LABEL.items()}.get(title)
        colour = COLOR.get(key, "#333333")
        ax.set_title(title, fontsize=10 if key == "supervised" else 11,
                     color=colour, pad=6)
        if db is not None:
            ax.text(0.5, -0.055, f"{db:.2f} dB", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10.5, color="#555555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.075,
                        wspace=0.05, hspace=0.26)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    torch.set_num_threads(2)
    curves(os.path.join(OUT, "training_curves.png"))
    panels(os.path.join(OUT, "before_after.png"))
