"""Figure 5.1: the masking scheme of Noise2Void.

Left, a receptive field with its centre visible; centre, the same field with
the centre replaced by a neighbour's value; right, a 64 x 64 training window
with its 64 masked positions chosen by stratified sampling.

    python -m figures.fig_5_1
"""
import os

import numpy as np
from PIL import ImageDraw

from . import sources as S
from .style import OUT, grid, values, zoom, box, RED

FIELD = 9    # pixels across the receptive field drawn
WINDOW = 64  # training window size in the paper
MASKED = 64  # masked positions per window


def _field(a, cell, centre_value=None):
    """The receptive field as numbers, with the centre pixel outlined."""
    a = a.copy()
    if centre_value is not None:
        a[FIELD // 2, FIELD // 2] = centre_value
    img = values(a, cell=cell)
    c = FIELD // 2
    return box(img, c * cell, c * cell, (c + 1) * cell - 1, (c + 1) * cell - 1,
               colour=RED, width=4)


def make(path=None, sigma=25):
    x = S.camera()
    y = S.gauss(x, sigma, S.rng(5))
    gen = S.rng(6)

    i, j = 88, 20
    patch = y[i:i + FIELD, j:j + FIELD]
    # a random neighbour within 2 pixels, as in the paper
    di, dj = gen.integers(-2, 3), gen.integers(-2, 3)
    while di == 0 and dj == 0:
        di, dj = gen.integers(-2, 3), gen.integers(-2, 3)
    neighbour = y[i + FIELD // 2 + di, j + FIELD // 2 + dj]

    cell = 42
    visible = _field(patch, cell)
    masked = _field(patch, cell, centre_value=neighbour)

    win = y[56:56 + WINDOW, 8:8 + WINDOW]
    k = 6
    img = zoom(win, k)
    d = ImageDraw.Draw(img)
    blocks = int(np.sqrt(MASKED))
    step = WINDOW // blocks
    for bi in range(blocks):
        for bj in range(blocks):
            pi = bi * step + int(gen.integers(0, step))
            pj = bj * step + int(gen.integers(0, step))
            cx, cy = (pj + 0.5) * k, (pi + 0.5) * k
            d.ellipse([cx - k, cy - k, cx + k, cy + k], outline=RED, width=2)

    return grid(
        [("רשת רגילה: הפיקסל המרכזי נראה לרשת", visible),
         ("נקודה עיוורת: ערכו הוחלף בערך של שכן", masked),
         (f"חלון אימון {WINDOW} על {WINDOW}: "
          f"{MASKED} המיקומים המסווים שבו", img)],
        path or os.path.join(OUT, "fig_5_1.png"),
        suptitle="סכמת המיסוך של Noise2Void: ההסתרה נעשית בנתונים "
                 "ולא בארכיטקטורה",
        cols=3, panel=300)


if __name__ == "__main__":
    make()
