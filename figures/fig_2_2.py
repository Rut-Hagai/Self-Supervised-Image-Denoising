"""Figure 2.2: a colour image and its three channels.

    python -m figures.fig_2_2
"""
import os

import numpy as np

from . import sources as S
from .style import OUT, grid


def make(path=None):
    x = S.astronaut()
    panels = [("תמונת צבע\nשלוש מטריצות זו על גב זו", x)]
    for c, name in enumerate(("האדום", "הירוק", "הכחול")):
        # tinted only to tell the channels apart: white where the channel is
        # full, its own colour where it is empty
        v = x[:, :, c]
        tinted = np.ones_like(x)
        for other in range(3):
            if other != c:
                tinted[:, :, other] = v
        panels.append((f"הערוץ {name}\nמטריצה אחת של מספרים", tinted))
    return grid(panels, path or os.path.join(OUT, "fig_2_2.png"),
                cols=4, panel=250, title_lines=2)


if __name__ == "__main__":
    make()
