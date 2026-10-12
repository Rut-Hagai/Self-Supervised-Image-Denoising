"""Figure 6.2: epsilon, the signal difference between the two neighbours,
measured on the clean image and on the noisy one.

    python -m figures.fig_6_2
"""
import os

import numpy as np

from . import sources as S
from .fig_6_1 import sub_sample
from .style import OUT, grid, zoom, colormap

N = 80


def make(path=None, sigma=25):
    x = S.camera()
    y = S.gauss(x, sigma, S.rng(9))
    i, j = 36, 8
    xc, yc = x[i:i + N, j:j + N], y[i:i + N, j:j + N]

    # the same pairs for both maps
    seed = 11
    a1, a2 = sub_sample(xc, S.rng(seed))
    b1, b2 = sub_sample(yc, S.rng(seed))
    eps_clean = np.abs(a1 - a2)
    eps_noisy = np.abs(b1 - b2)

    return grid(
        [("התמונה הנקייה x", zoom(xc, 4)),
         ("ε, הפרש הסיגנל הנקי בין השכנים", zoom(colormap(eps_clean), 8)),
         ("אותו הפרש, נמדד בתמונה הרועשת", zoom(colormap(eps_noisy), 8))],
        path or os.path.join(OUT, "fig_6_2.png"),
        suptitle="בהיר פירושו הפרש גדול. בתמונה הנקייה ε נדלק בקצוות בלבד, "
                 "ובתמונה הרועשת הוא גדול כמעט בכל מקום",
        cols=3, panel=300)


if __name__ == "__main__":
    make()
