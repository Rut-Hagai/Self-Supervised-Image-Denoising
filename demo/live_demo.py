"""Live demonstration for the seminar talk.

  1  the Neighbor2Neighbor checkpoint released by its authors
  2  every model trained in this project, on the same noisy image
  3  ZS-N2N trained from scratch on the image, with a live window

    python -m demo.live_demo --image 4 --part 1
    python -m demo.live_demo --image 4 --part 2
    python -m demo.live_demo --image 4 --part 3

Without --part all three run in order. Each part writes its figure to results/.
"""
import argparse
import os
import time

import numpy as np
import torch

from src.common import count_params, denoise, load_images, load_net
from src.hebrew import draw_rtl, font
from src.metrics import psnr, ssim, to_uint8
from src.noise import NoiseModel
from src.zs_n2n import denoise_image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = "weights/model_gauss25_b4e100r02.pth"
OURS = ["supervised", "noise2noise", "noise2void", "neighbor2neighbor",
        "blind2unblind"]

LABEL = {"supervised": "אימון מפוקח", "noise2noise": "Noise2Noise",
         "noise2void": "Noise2Void", "neighbor2neighbor": "Neighbor2Neighbor",
         "blind2unblind": "Blind2Unblind"}


def show(panels, path, suptitle=None, cols=None):
    """panels: list of (title, image). Writes the figure and opens it.

    Matplotlib cannot draw Hebrew, so the titles are added afterwards with PIL.
    """
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    cols = cols or min(len(panels), 4)
    rows = int(np.ceil(len(panels) / cols))
    W, H = int(3.4 * cols * 120), int(3.7 * rows * 120)
    fig, axes = plt.subplots(rows, cols, figsize=(W / 120, H / 120), dpi=120,
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(np.clip(im, 0, 1))
    top = 0.90 if suptitle else 0.95
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=0.01,
                        wspace=0.04, hspace=0.12)

    fig.canvas.draw()
    boxes = [ax.get_window_extent() for ax in axes.ravel()[:len(panels)]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)

    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)

    def rtl(text, cx, top, size=17):
        draw_rtl(d, text, cx, top, font(size))

    for (title, _), box in zip(panels, boxes):
        rtl(title, (box.x0 + box.x1) / 2, H - box.y1 - 24)
    if suptitle:
        rtl(suptitle, W / 2, 10, size=19)
    img.save(path)
    print(f"  figure written to {path}")
    try:
        Image.open(path).show()
    except Exception:
        pass


class LiveWindow:
    """A window showing the noisy image and the current ZS-N2N output, redrawn
    as training runs. Each frame is composed with PIL for the Hebrew titles.
    If no window can be opened, training carries on without it.
    """

    def __init__(self, noisy, clean=None, every=100):
        self.noisy, self.clean, self.every = noisy, clean, every
        self.ok, self.im = False, None
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            self.plt = plt
            frame = self._frame(0, None, noisy * 0 + 0.5)
            h, w, _ = frame.shape
            self.fig, ax = plt.subplots(figsize=(w / 110, h / 110), dpi=110)
            self.fig.patch.set_facecolor("white")
            ax.axis("off")
            self.im = ax.imshow(frame)
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.fig.canvas.manager.set_window_title("ZS-N2N")
            plt.show(block=False)
            plt.pause(0.001)
            self.ok = True
        except Exception as exc:
            print(f"  (no live window: {exc})")

    def _frame(self, step, cur, est, panel=360):
        from PIL import Image, ImageDraw

        def tile(a):
            img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
            return img.resize((panel, panel), Image.NEAREST)

        pad, top, bottom = 18, 34, 30
        canvas = Image.new("RGB", (panel * 2 + pad * 3, panel + top + bottom),
                           (255, 255, 255))
        d = ImageDraw.Draw(canvas)

        def rtl(text, cx, y, size=16):
            draw_rtl(d, text, cx, y, font(size))

        right_x = pad * 2 + panel
        canvas.paste(tile(self.noisy), (right_x, top))
        canvas.paste(tile(est), (pad, top))
        rtl("התמונה הרועשת", right_x + panel / 2, 8)
        rtl("הפלט של הרשת ברגע זה", pad + panel / 2, 8)
        line = f"צעד {step}"
        if cur is not None:
            line += f"   PSNR {cur:.2f} dB"
        rtl(line, canvas.width / 2, top + panel + 7, size=17)
        return np.asarray(canvas)

    def update(self, step, cur, est):
        if not self.ok or est is None or step % self.every:
            return
        try:
            self.im.set_data(self._frame(step, cur, est))
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
        except Exception:
            self.ok = False

    def close(self):
        if self.ok:
            try:
                self.plt.ioff()
            except Exception:
                pass


def db(clean, im):
    """', 28.43 dB', or '' when there is no clean image (--already-noisy)."""
    return "" if clean is None else f", {psnr(to_uint8(clean), to_uint8(im)):.2f} dB"


def crop_centre(im, size):
    if size <= 0:
        return im
    h, w, _ = im.shape
    return im[h // 2 - size // 2:h // 2 + size // 2,
              w // 2 - size // 2:w // 2 + size // 2]


def part1(x, y, args):
    print("\nPART 1  the model released by the authors of Neighbor2Neighbor")
    path = os.path.join(HERE, PUBLISHED)
    if not os.path.exists(path):
        print(f"  missing {path} - run scripts/get_data.py first")
        return
    net = load_net(path)
    print(f"  {count_params(net):,} parameters, trained by the authors on ImageNet")
    t = time.time()
    est = denoise(net, y)
    print(f"  denoised in {time.time() - t:.1f}s on this machine")
    print(f"  noisy {db(x, y) or ' (no clean reference)'}")
    print(f"  output{db(x, est) or ' (no clean reference)'}")
    panels = [(f"התמונה הרועשת{db(x, y)}", y),
              (f"Neighbor2Neighbor, המשקלים של המחברים{db(x, est)}", est)]
    if x is not None:
        panels.insert(0, ("התמונה הנקייה", x))
    show(panels, os.path.join(HERE, "results", "demo_part1.png"), cols=3)


def part2(x, y, args):
    print("\nPART 2  the models trained in this project, under one protocol")
    panels = [(f"התמונה הרועשת{db(x, y)}", y)]
    if x is not None:
        panels.insert(0, ("התמונה הנקייה", x))
    found = 0
    for name in OURS:
        path = os.path.join(HERE, "weights", f"{name}_{args.noise}.pth")
        if not os.path.exists(path):
            print(f"  {name:20s} no checkpoint yet ({path})")
            continue
        net = load_net(path, n_feature=args.n_feature)
        est = denoise(net, y)
        if x is None:
            print(f"  {name:20s} done")
        else:
            print(f"  {name:20s} {psnr(to_uint8(x), to_uint8(est)):6.2f} dB"
                  f"   SSIM {ssim(to_uint8(x), to_uint8(est)):.4f}")
        panels.append((f"{LABEL[name]}{db(x, est)}", est))
        found += 1
    if found:
        show(panels, os.path.join(HERE, "results", "demo_part2.png"),
             suptitle="אותה רשת, אותם נתונים ואותו לוח אימון. "
                      "רק זוג האימון ופונקציית ההפסד שונים")


def part3(x, y, args):
    print("\nPART 3  ZS-N2N trained live on this one image, from scratch")
    t0 = time.time()

    window = None if args.no_window else LiveWindow(y, x)

    def on_step(step, cur, running):
        if window is not None:
            window.update(step, cur, running)
        if step % 200 == 0:
            msg = f"  step {step:5d}  {time.time() - t0:5.1f}s"
            if cur:
                msg += f"   PSNR {cur:.2f} dB"
            print(msg, flush=True)

    est, info = denoise_image(y, iterations=args.iterations, clean_hwc=x,
                              seed=args.seed, on_step=on_step)
    if window is not None:
        window.close()
    print(f"  {info['params']:,} parameters, {info['seconds']:.0f}s total")
    if x is not None:
        print(f"  noisy {info['psnr_noisy']:.2f} dB -> {info['psnr']:.2f} dB")
    panels = [(f"התמונה הרועשת{db(x, y)}", y),
              (f"ZS-N2N, אומן ממש עכשיו{db(x, est)}", est)]
    if x is not None:
        panels.insert(0, ("התמונה הנקייה", x))
    show(panels, os.path.join(HERE, "results", "demo_part3.png"), cols=3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=int, default=4, help="index into the test set")
    ap.add_argument("--image-path", default=None,
                    help="any image file, instead of the test set")
    ap.add_argument("--already-noisy", action="store_true",
                    help="the file is a real noisy photograph; nothing is "
                         "added to it and no PSNR is reported")
    ap.add_argument("--crop", type=int, default=256, help="0 for the full image")
    ap.add_argument("--noise", default="gauss25")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--n-feature", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-dir", default="data/Kodak")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--part", type=int, choices=[1, 2, 3])
    ap.add_argument("--no-window", action="store_true",
                    help="part 3: do not open the live window, print only")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.image_path:
        from PIL import Image as PILImage
        im = np.asarray(PILImage.open(args.image_path).convert("RGB"),
                        dtype=np.float32) / 255.0
        print(f"image: {args.image_path}, {im.shape[1]} x {im.shape[0]} pixels")
    else:
        images = load_images(os.path.join(HERE, args.test_dir))
        im = images[args.image % len(images)]
    im = crop_centre(im, args.crop)

    if args.already_noisy:
        x, y = None, im
        print("treated as already noisy; no clean reference, so no PSNR")
    else:
        x = im
        y = NoiseModel(args.noise)(x, np.random.default_rng(args.seed))

    parts = [args.part] if args.part else [1, 2, 3]
    for p in parts:
        {1: part1, 2: part2, 3: part3}[p](x, y, args)


if __name__ == "__main__":
    main()
