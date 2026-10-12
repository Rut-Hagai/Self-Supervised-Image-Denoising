"""How much correlation between neighbouring pixels does demosaicing put into
the noise? Not reported in the seminar paper.

  1. build an RGGB sensor reading from a clean colour image
  2. add independent Gaussian noise to every photosite
  3. demosaic the noisy and the clean readings and subtract

The difference is the noise after demosaicing; the correlation is measured in
the green channel.

    python -m experiments.exp_c_demosaic_correlation
"""
import argparse
import json
import os

import numpy as np
from scipy.signal import convolve2d
from skimage import data

from src.hebrew import draw_rtl, font as hebrew_font, measure

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# bilinear demosaicing: green is on half the sites, red and blue on a quarter
K_GREEN = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], float) / 4.0
K_RED_BLUE = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 4.0


def bayer_masks(h, w):
    """RGGB: red on even rows and columns, blue on odd, green on the rest."""
    R = np.zeros((h, w), bool)
    G = np.zeros((h, w), bool)
    B = np.zeros((h, w), bool)
    R[0::2, 0::2] = True
    B[1::2, 1::2] = True
    G[0::2, 1::2] = True
    G[1::2, 0::2] = True
    return R, G, B


def demosaic_bilinear(mosaic, R, G, B):
    out = np.empty(mosaic.shape + (3,), float)
    out[..., 0] = convolve2d(mosaic * R, K_RED_BLUE, mode="same", boundary="symm")
    out[..., 1] = convolve2d(mosaic * G, K_GREEN, mode="same", boundary="symm")
    out[..., 2] = convolve2d(mosaic * B, K_RED_BLUE, mode="same", boundary="symm")
    return out


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def neighbour_corr(n):
    return {"horizontal": corr(n[:, :-1], n[:, 1:]),
            "vertical": corr(n[:-1, :], n[1:, :]),
            "diagonal": corr(n[:-1, :-1], n[1:, 1:])}


def training_pair_corr(n):
    """Correlation within the training pairs of Neighbor2Neighbor (two
    adjacent pixels of a cell) and ZS-N2N (the two diagonal means)."""
    a, b = n[0::2, 0::2], n[0::2, 1::2]
    c, d = n[1::2, 0::2], n[1::2, 1::2]
    return {"Neighbor2Neighbor pair": corr(a, b),
            "ZS-N2N pair": corr((b + c) / 2, (a + d) / 2)}


def run(sigma, seed=0):
    rng = np.random.default_rng(seed)
    x = data.astronaut().astype(float)
    h, w, _ = x.shape
    R, G, B = bayer_masks(h, w)
    mosaic = x[..., 0] * R + x[..., 1] * G + x[..., 2] * B
    n_raw = rng.normal(0.0, sigma, size=(h, w))
    n_dem = (demosaic_bilinear(mosaic + n_raw, R, G, B)
             - demosaic_bilinear(mosaic, R, G, B))[..., 1]
    return n_raw, n_dem


def figure(n_raw, n_dem, path, sigma, k=6000, seed=1):
    """Scatter of each pixel's noise against its right neighbour's, before and
    after demosaicing, with the mean per bin drawn on top (flat means
    independent)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    W, H = 860, 430
    rng = np.random.default_rng(seed)
    lim = 3.4 * sigma
    edges = np.linspace(-lim, lim, 17)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    axes = [fig.add_axes([0.095 + 0.455 * j, 0.215, 0.375, 0.675])
            for j in (1, 0)]  # right panel first: Hebrew reads right to left

    for ax, n in ((axes[0], n_raw), (axes[1], n_dem)):
        a, b = n[:, :-1].ravel(), n[:, 1:].ravel()
        i = rng.choice(a.size, k, replace=False)
        ax.scatter(a[i], b[i], s=1.2, c="#444444", alpha=0.18, linewidths=0,
                   zorder=2)
        which = np.digitize(a, edges) - 1
        means = [b[which == j].mean() if np.any(which == j) else np.nan
                 for j in range(len(centres))]
        ax.plot(centres, means, color="#D55E00", lw=2.0, marker="o", ms=5,
                zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#999999", lw=0.7, zorder=1)
        ax.axvline(0, color="#999999", lw=0.7, zorder=1)
        ax.tick_params(labelsize=9)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    fig.savefig(path, facecolor="white")
    plt.close(fig)

    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)

    def rtl(text, x, y, size=15, rotate=False):
        f = hebrew_font(size)
        w, h = measure(d, text, f)
        if rotate:
            lab = Image.new("RGBA", (int(w) + 8, int(h) + 10), (255, 255, 255, 0))
            draw_rtl(ImageDraw.Draw(lab), text, 4, 5, f, anchor="left")
            lab = lab.rotate(90, expand=True)
            img.paste(lab, (int(x), int(y - lab.height / 2)), lab)
            return
        draw_rtl(d, text, x, y - h / 2, f)

    def r(n):
        v = corr(n[:, :-1], n[:, 1:])
        return 0.0 if abs(v) < 0.005 else v

    titles = [f"לפני שחזור הפסיפס, מתאם {r(n_raw):.2f}",
              f"אחרי שחזור הפסיפס, מתאם {r(n_dem):.2f}"]
    for box, title in zip(boxes, titles):
        cx = (box.x0 + box.x1) / 2 * W
        rtl(title, cx, (1 - box.y1) * H - 14)
        rtl("הרעש בפיקסל", cx, (1 - box.y0) * H + 40)
    rtl("הרעש בשכן מימין", 10, H / 2 - 20, rotate=True)
    img.save(path)
    print(f"figure written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[5.0, 10.0, 25.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = {"settings": vars(args), "per_sigma": {}}
    print(f"{'sigma':>6s} | {'before: h':>10s} {'v':>7s} {'d':>7s}"
          f" | {'after: h':>9s} {'v':>7s} {'d':>7s}")
    for s in args.sigmas:
        n_raw, n_dem = run(s, args.seed)
        before, after = neighbour_corr(n_raw), neighbour_corr(n_dem)
        out["per_sigma"][str(s)] = {"before": before, "after": after}
        print(f"{s:6.0f} | {before['horizontal']:10.3f} {before['vertical']:7.3f}"
              f" {before['diagonal']:7.3f} | {after['horizontal']:9.3f}"
              f" {after['vertical']:7.3f} {after['diagonal']:7.3f}")

    sigma = args.sigmas[-1]
    n_raw, n_dem = run(sigma, args.seed)
    pb, pa = training_pair_corr(n_raw), training_pair_corr(n_dem)
    out["training_pairs"] = {"sigma": sigma, "before": pb, "after": pa}
    print(f"\nthe pairs the methods train on, sigma={sigma:.0f}")
    for name in pb:
        print(f"  {name:24s} {pb[name]:+.3f}  ->  {pa[name]:+.3f}")
    out["noise_std"] = {"before": float(n_raw.std()), "after": float(n_dem.std())}
    print(f"\nnoise std {n_raw.std():.2f} -> {n_dem.std():.2f}")
    print("\nThe correlation does not depend on sigma: demosaicing is a linear\n"
          "operation, so scaling the noise scales both sides equally.\n"
          "This is a lower bound. Only one stage of the camera pipeline is\n"
          "simulated, in its simplest variant, and the noise here is uniform\n"
          "Gaussian while a real sensor's noise depends on light intensity.")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "exp_c.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    figure(n_raw, n_dem, os.path.join(HERE, "results", "exp_c.png"), sigma)
    return out


if __name__ == "__main__":
    main()
