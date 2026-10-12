"""Figure 1.1: the clean image, the noise, and their sum.

    python -m figures.fig_1_1
"""
import os

from . import sources as S
from .style import OUT, grid


def make(path=None, sigma=25):
    x = S.camera()
    y = S.gauss(x, sigma, S.rng(1))
    n = y - x
    # noise around mid grey, stretched so it is visible
    shown = 0.5 + n * 3.0
    return grid(
        [("התמונה הנקייה x\n(אינה בידינו בפועל)", x),
         ("רכיב הרעש n\n(תוחלת אפס, בלתי תלוי בין פיקסלים)", shown),
         (f"התמונה הנמדדת y = x + n\nPSNR {S.psnr(x, y):.2f} dB", y)],
        path or os.path.join(OUT, "fig_1_1.png"),
        cols=3, panel=330, title_lines=2)


if __name__ == "__main__":
    make()
