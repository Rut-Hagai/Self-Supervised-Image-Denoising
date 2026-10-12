"""The comparison table and figure for the four dataset-based methods and the
supervised reference (experiment B in the paper).

Reads the logs written by src/train.py; trains nothing. Run the trainings
first with run_all.bat.

    python -m experiments.exp_d_compare_methods
"""
import argparse
import json
import os

from src.hebrew import draw_rtl, font as hebrew_font, measure

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORDER = ["supervised", "noise2noise", "noise2void", "neighbor2neighbor",
         "blind2unblind"]

# Okabe-Ito palette, safe for colour-blind readers
COLORS = {"supervised": "#0072B2", "noise2noise": "#D55E00",
          "noise2void": "#009E73", "neighbor2neighbor": "#E69F00",
          "blind2unblind": "#CC79A7"}


def load(noise="gauss25"):
    out = {}
    for m in ORDER:
        p = os.path.join(HERE, "results", f"{m}_{noise}.json")
        if os.path.exists(p):
            out[m] = json.load(open(p))
    return out


def table(runs):
    sup = runs["supervised"]["final_psnr"]
    rows = []
    print(f"{'method':20s} {'final dB':>9s} {'gap':>7s} {'best dB':>8s}"
          f" {'at step':>8s} {'hours':>7s}")
    for m in ORDER:
        if m not in runs:
            continue
        d = runs[m]
        best = max(d["log"], key=lambda p: p["psnr"])
        rows.append({"method": m, "final": round(d["final_psnr"], 3),
                     "gap": round(sup - d["final_psnr"], 3),
                     "best": round(best["psnr"], 3), "best_step": best["step"],
                     "hours": round(d["seconds"] / 3600, 2)})
        print(f"{m:20s} {d['final_psnr']:9.3f} {sup - d['final_psnr']:7.3f}"
              f" {best['psnr']:8.3f} {best['step']:8d} {d['seconds'] / 3600:7.2f}")
    return rows


LABELS = {"supervised": "אימון מפוקח", "noise2noise": "Noise2Noise",
          "noise2void": "Noise2Void", "neighbor2neighbor": "Neighbor2Neighbor",
          "blind2unblind": "Blind2Unblind"}


def figure(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    W, H = 860, 470
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.085, 0.155, 0.645, 0.805])

    for m in ORDER:
        if m not in runs:
            continue
        c = runs[m]["log"]
        ax.plot([p["step"] for p in c], [p["psnr"] for p in c], lw=2.0,
                color=COLORS[m], marker="o", ms=4, zorder=3)

    # labels at the line ends; matplotlib cannot draw Hebrew, so those are
    # positioned here and drawn with PIL below
    nudge = {"supervised": 0.22, "noise2noise": -0.26}
    hebrew = []
    for m in ORDER:
        if m not in runs:
            continue
        last = runs[m]["log"][-1]
        xy = (last["step"], last["psnr"] + nudge.get(m, 0.0))
        if LABELS[m].isascii():
            ax.annotate(LABELS[m], xy, xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=10, color=COLORS[m])
        else:
            hebrew.append((m, xy))

    # Blind2Unblind's lambda starts rising at 80% of 20,000 steps
    lam_xy = None
    if "blind2unblind" in runs:
        ax.axvline(16000, color="#999999", lw=1.0, ls=":", zorder=1)
        lam_xy = (16000, ax.get_ylim()[0])

    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.tick_params(labelsize=9)

    fig.canvas.draw()

    def px(xy):
        x, y = ax.transData.transform(xy)
        return x, H - y
    hebrew_px = [(m, px(xy), COLORS[m]) for m, xy in hebrew]
    lam_px = px(lam_xy) if lam_xy else None
    fig.savefig(path, facecolor="white")
    plt.close(fig)

    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)

    def rtl(text, x, y, size=16, fill=(0, 0, 0), anchor="center", rotate=False):
        f = hebrew_font(size)
        w, h = measure(d, text, f)
        if rotate:
            lab = Image.new("RGBA", (int(w) + 8, int(h) + 10), (255, 255, 255, 0))
            draw_rtl(ImageDraw.Draw(lab), text, 4, 5, f, fill=fill, anchor="left")
            lab = lab.rotate(90, expand=True)
            img.paste(lab, (int(x), int(y - lab.height / 2)), lab)
            return
        left = {"center": x - w / 2, "left": x, "right": x - w}[anchor]
        draw_rtl(d, text, left, y - h / 2, f, fill=fill, anchor="left")

    rtl("צעד אימון", 0.085 * W + 0.645 * W / 2, H - 20)
    rtl("PSNR ביחידות dB", 8, 0.155 * H + 0.805 * H / 2, rotate=True)
    for m, (x, y), color in hebrew_px:
        rtl(LABELS[m], x + 9, y, anchor="left",
            fill=tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)))
    if lam_px:
        rtl("עליית λ", lam_px[0] - 6, lam_px[1] - 16, size=14,
            anchor="right", fill=(102, 102, 102))
    img.save(path)
    print(f"figure written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", default="gauss25")
    args = ap.parse_args(argv)

    runs = load(args.noise)
    missing = [m for m in ORDER if m not in runs]
    if missing:
        print("missing runs:", ", ".join(missing))
    if "supervised" not in runs:
        print("the supervised run is the reference; it has to exist")
        return
    rows = table(runs)
    out = os.path.join(HERE, "results", "exp_d.json")
    json.dump({"rows": rows,
               "curves": {m: runs[m]["log"] for m in runs}},
              open(out, "w"), indent=2)
    figure(runs, os.path.join(HERE, "results", "exp_d.png"))


if __name__ == "__main__":
    main()
