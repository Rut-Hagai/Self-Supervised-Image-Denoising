"""Figure 3.1: averaging independent noisy shots of the same image.

    python -m figures.fig_3_1
"""
import os

import numpy as np

from . import sources as S
from .style import OUT, grid

SIGMA = 40
COUNTS = (1, 2, 4, 16, 64)


def make(path=None):
    x = S.camera()
    gen = S.rng(3)
    shots = [S.gauss(x, SIGMA, gen) for _ in range(max(COUNTS))]

    panels = [("התמונה הנקייה", x)]
    for n in COUNTS:
        avg = np.mean(shots[:n], axis=0)
        db = S.psnr(x, avg)
        label = ("צילום בודד, ללא מיצוע" if n == 1
                 else f"ממוצע של {n} צילומים")
        panels.append((f"{label}\n{db:.2f} dB", avg))
    out = grid(panels, path or os.path.join(OUT, "fig_3_1.png"),
               suptitle="מיצוע צילומים רועשים בלתי תלויים: "
                        "הרעש מתבטל והסיגנל נשמר",
               cols=3, panel=290, title_lines=2)
    print(f"  single shot {S.psnr(x, shots[0]):.2f} dB, "
          f"average of {max(COUNTS)} {S.psnr(x, np.mean(shots, axis=0)):.2f} dB")
    return out


if __name__ == "__main__":
    make()
