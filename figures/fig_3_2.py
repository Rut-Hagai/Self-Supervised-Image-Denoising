"""Figure 3.2: one convolution with five different 3 x 3 kernels.

    python -m figures.fig_3_2
"""
import os

import numpy as np
from scipy.ndimage import convolve

from . import sources as S
from .style import OUT, grid

KERNELS = [
    ("זהות", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float), False),
    ("מיצוע (טשטוש)", np.ones((3, 3)) / 9.0, False),
    ("קצוות אנכיים", np.array([[-1, 0, 1]] * 3, float), True),
    ("קצוות אופקיים", np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float), True),
    ("חידוד", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float), False),
]


def make(path=None):
    x = S.camera()
    panels = []
    for name, k, centred in KERNELS:
        out = convolve(x, k, mode="reflect")
        # edge kernels sum to zero, so their output is shown around mid grey
        panels.append((name, out + 0.5 if centred else np.clip(out, 0, 1)))
    return grid(panels, path or os.path.join(OUT, "fig_3_2.png"),
                suptitle="קונבולוציה אחת היא טבלה אחת בגודל 3 על 3, "
                         "המוסעת על פני התמונה",
                cols=5, panel=215)


if __name__ == "__main__":
    make()
