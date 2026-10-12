"""Figure 3.3: local mean against non-local means on a crop with an edge.

    python -m figures.fig_3_3
"""
import os

from scipy.ndimage import uniform_filter

from . import sources as S
from .style import OUT, grid, zoom

SIGMA = 25
CROP = (64, 128, 64)  # row, column, size: one strong edge with flat sides


def make(path=None):
    x_full = S.camera()
    y_full = S.gauss(x_full, SIGMA, S.rng(4))

    from skimage.restoration import denoise_nl_means
    local = uniform_filter(y_full, size=5)
    nlm = denoise_nl_means(y_full, patch_size=5, patch_distance=7,
                           h=0.8 * SIGMA / 255.0, fast_mode=True,
                           sigma=SIGMA / 255.0)

    i, j, n = CROP

    def cut(a):
        return a[i:i + n, j:j + n]

    panels = [("נקייה", zoom(cut(x_full), 6)),
              (f"רועשת\n{S.psnr(x_full, y_full):.2f} dB", zoom(cut(y_full), 6)),
              (f"מיצוע מקומי\n{S.psnr(x_full, local):.2f} dB", zoom(cut(local), 6)),
              (f"מיצוע לא-מקומי\n{S.psnr(x_full, nlm):.2f} dB", zoom(cut(nlm), 6))]
    return grid(panels, path or os.path.join(OUT, "fig_3_3.png"),
                suptitle="מיצוע מקומי מסיר רעש אך הורס את הקצה, "
                         "ואילו מיצוע לא-מקומי משמר אותו",
                cols=4, panel=250, title_lines=2)


if __name__ == "__main__":
    make()
