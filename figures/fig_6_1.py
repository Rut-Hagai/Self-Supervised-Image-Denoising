"""Figure 6.1: the neighbour sub-sampler of Neighbor2Neighbor.

    python -m figures.fig_6_1
"""
import os

import numpy as np

from . import sources as S
from .style import OUT, grid, zoom, lines, box, RED

N = 24         # pixels across the patch shown
CELL = (4, 3)  # the 2 x 2 cell outlined in red


def sub_sample(a, gen):
    """g1 and g2: two adjacent pixels from every 2 by 2 cell."""
    h, w = a.shape
    g1 = np.zeros((h // 2, w // 2))
    g2 = np.zeros((h // 2, w // 2))
    for i in range(h // 2):
        for j in range(w // 2):
            # horizontally or vertically adjacent, never diagonally
            if gen.integers(0, 2):
                r = int(gen.integers(0, 2))
                p, q = (2 * i + r, 2 * j), (2 * i + r, 2 * j + 1)
            else:
                c = int(gen.integers(0, 2))
                p, q = (2 * i, 2 * j + c), (2 * i + 1, 2 * j + c)
            if gen.integers(0, 2):
                p, q = q, p
            g1[i, j], g2[i, j] = a[p], a[q]
    return g1, g2


def make(path=None, sigma=25):
    x = S.camera()
    y = S.gauss(x, sigma, S.rng(7))
    patch = y[72:72 + N, 16:16 + N]
    g1, g2 = sub_sample(patch, S.rng(8))

    k = 14
    whole = lines(zoom(patch, k), step=2 * k)
    ci, cj = CELL
    box(whole, cj * 2 * k, ci * 2 * k, (cj + 1) * 2 * k, (ci + 1) * 2 * k,
        colour=RED, width=3)

    return grid(
        [("התמונה הרועשת y, מחולקת לתאים של 2 על 2", whole),
         ("תת-התמונה g1: פיקסל אחד מכל תא", zoom(g1, 2 * k)),
         ("תת-התמונה g2: שכנו שבאותו תא", zoom(g2, 2 * k))],
        path or os.path.join(OUT, "fig_6_1.png"),
        suptitle="הדוגם השכן: זוג אימון שנבנה מתוך תמונה רועשת אחת",
        cols=3, panel=300)


if __name__ == "__main__":
    make()
