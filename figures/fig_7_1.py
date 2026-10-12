"""Figure 7.1: the global mask mapper of Blind2Unblind, shown on the
replacement values without a network.

    python -m figures.fig_7_1
"""
import os

import numpy as np
from PIL import ImageDraw
from scipy.ndimage import convolve

from . import sources as S
from .style import OUT, grid, zoom, lines, RED

N = 24


def _replacement(a):
    """The neighbourhood mean that is written in place of a masked pixel."""
    k = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
    return convolve(a, k / k.sum(), mode="reflect")


def _masked(a, pos, rep):
    """One of the four versions: position `pos` of every cell replaced."""
    out = a.copy()
    di, dj = divmod(pos, 2)
    out[di::2, dj::2] = rep[di::2, dj::2]
    return out


def make(path=None, sigma=25):
    x = S.camera()
    y = S.gauss(x, sigma, S.rng(12))
    patch = y[72:72 + N, 16:16 + N]
    rep = _replacement(patch)

    k = 14
    whole = lines(zoom(patch, k), step=2 * k)

    def marked(pos):
        img = zoom(_masked(patch, pos, rep), k)
        d = ImageDraw.Draw(img)
        di, dj = divmod(pos, 2)
        for i in range(di, N, 2):
            for j in range(dj, N, 2):
                d.rectangle([j * k, i * k, (j + 1) * k - 1, (i + 1) * k - 1],
                            outline=RED, width=2)
        return img

    # each pixel from the version in which it was masked
    joined = np.zeros_like(patch)
    for pos in range(4):
        di, dj = divmod(pos, 2)
        joined[di::2, dj::2] = _masked(patch, pos, rep)[di::2, dj::2]

    return grid(
        [("התמונה הרועשת y, בתאים של 2 על 2", whole),
         ("אחת מארבע הגרסאות הממוסכות", marked(0)),
         ("גרסה ממוסכת אחרת", marked(3)),
         ("ההרכבה: מכל גרסה רק מה שמוסך בה", zoom(joined, k))],
        path or os.path.join(OUT, "fig_7_1.png"),
        suptitle="ממפה המסכות הגלובלי: ארבע הגרסאות יחד מכסות כל פיקסל "
                 "בדיוק פעם אחת",
        cols=4, panel=250)


if __name__ == "__main__":
    make()
