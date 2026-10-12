"""Figure 2.1: an image and the pixel values of one 8 x 8 window.

    python -m figures.fig_2_1
"""
import os

from . import sources as S
from .style import OUT, grid, to_img, box, values

WIN = (86, 82)  # top-left of a window that crosses an edge


def make(path=None, n=8):
    x = S.camera()
    i, j = WIN

    whole = to_img(x).resize((512, 512))
    k = 512 / x.shape[0]
    box(whole, j * k, i * k, (j + n) * k, (i + n) * k, width=3)

    window = values(x[i:i + n, j:j + n], cell=56)

    return grid(
        [("תמונה בגודל 256 על 256", whole),
         ("החלון המסומן, בגודל 8 על 8,\nוערכי הפיקסלים שבו בטווח 0 עד 255", window)],
        path or os.path.join(OUT, "fig_2_1.png"),
        cols=2, panel=420, title_lines=2)


if __name__ == "__main__":
    make()
