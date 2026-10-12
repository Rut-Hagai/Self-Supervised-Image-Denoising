"""Figure 2.4: the same image at five PSNR levels.

    python -m figures.fig_2_4
"""
import os

from . import sources as S
from .style import OUT, grid

TARGETS = (20, 25, 30, 35, 40)


def make(path=None):
    x = S.camera()
    panels = []
    for t in TARGETS:
        # PSNR = 20 log10(255 / sigma), solved for sigma
        sigma = 255.0 / (10 ** (t / 20.0))
        y = S.gauss(x, sigma, S.rng(10 + t))
        panels.append((f"{S.psnr(x, y):.1f} dB", y))
    return grid(panels, path or os.path.join(OUT, "fig_2_4.png"),
                suptitle="סולם ה-PSNR כפי שהוא נראה בעין",
                cols=5, panel=215)


if __name__ == "__main__":
    make()
