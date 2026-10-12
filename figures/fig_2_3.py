"""Figure 2.3: Poisson noise at five light levels.

    python -m figures.fig_2_3
"""
import os

from . import sources as S
from .style import OUT, grid

PEAKS = (4, 16, 64, 256, 4096)


def make(path=None):
    x = S.camera()
    gen = S.rng(2)
    panels = []
    for p in PEAKS:
        y = S.poisson(x, p, gen)
        panels.append((f"לכל היותר {p:,} פוטונים\n{S.psnr(x, y):.1f} dB", y))
    return grid(panels, path or os.path.join(OUT, "fig_2_3.png"),
                suptitle="רעש פואסון: פחות פוטונים פירושו יותר רעש יחסי, "
                         "והאזורים הכהים רועשים תמיד יותר",
                cols=5, panel=215, title_lines=2)


if __name__ == "__main__":
    make()
