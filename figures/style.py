"""Shared drawing for the paper's figures.

Panels are PIL images with Hebrew titles (matplotlib cannot draw Hebrew; see
src/hebrew.py). They are laid out left to right, matching the captions.
"""
import os

import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures_out")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
BLUE = (70, 110, 190)


def font(size):
    from src.hebrew import font as _font
    return _font(size)


def rtl(draw, text, cx, top, size=15, fill=BLACK, anchor="center"):
    """Draw one title line. See src/hebrew.py for how the ordering is done."""
    from src.hebrew import draw_rtl
    return draw_rtl(draw, text, cx, top, font(size), fill=fill,
                    anchor=anchor)[1]


def to_img(a):
    """A float array in [0, 1], grey or colour, as a PIL image."""
    if isinstance(a, Image.Image):
        return a.convert("RGB")
    a = np.clip(np.asarray(a, dtype=np.float64), 0, 1)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def zoom(img, k):
    """Nearest neighbour enlargement, so single pixels stay visible as squares."""
    img = to_img(img)
    return img.resize((img.width * k, img.height * k), Image.NEAREST)


def box(img, x0, y0, x1, y1, colour=RED, width=3):
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1, y1], outline=colour, width=width)
    return img


def lines(img, step, colour=BLUE, width=1):
    """A grid every `step` pixels, for showing the 2 by 2 cells."""
    d = ImageDraw.Draw(img)
    for x in range(0, img.width + 1, step):
        d.line([(x, 0), (x, img.height)], fill=colour, width=width)
    for y in range(0, img.height + 1, step):
        d.line([(0, y), (img.width, y)], fill=colour, width=width)
    return img


def values(a, cell=46, lo=None, hi=None, size=13):
    """A block of pixels drawn as grey squares with their values written on.

    Light squares get dark text and dark squares light text, so the numbers
    stay readable at both ends of the range.
    """
    a = np.asarray(a)
    v = a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    h, w = v.shape
    img = Image.new("RGB", (w * cell, h * cell), WHITE)
    d = ImageDraw.Draw(img)
    f = font(size)
    for i in range(h):
        for j in range(w):
            g = int(v[i, j])
            d.rectangle([j * cell, i * cell, (j + 1) * cell - 1, (i + 1) * cell - 1],
                        fill=(g, g, g))
            t = str(g)
            bb = d.textbbox((0, 0), t, font=f)
            d.text((j * cell + (cell - bb[2] + bb[0]) / 2 - bb[0],
                    i * cell + (cell - bb[3] + bb[1]) / 2 - bb[1]), t,
                   font=f, fill=(20, 20, 20) if g > 127 else (235, 235, 235))
    return img


def colormap(a, name="magma"):
    """A scalar map rendered in colour, for the two epsilon maps of figure 6.2."""
    import matplotlib
    try:
        cmap = matplotlib.colormaps[name]       # Matplotlib 3.6 and newer
    except Exception:                           # older releases
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name)
    a = np.asarray(a, dtype=np.float64)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    return Image.fromarray((cmap(a)[:, :, :3] * 255 + 0.5).astype(np.uint8))


def grid(panels, path, suptitle=None, cols=None, panel=340, gap=18,
         title_size=15, sup_size=17, title_lines=1):
    """Lay panels out in a grid, title each one, and write the file.

    panels : list of (title, image). A title may contain a newline.
    """
    cols = cols or min(len(panels), 5)
    rows = int(np.ceil(len(panels) / cols))
    imgs = []
    for t, im in panels:
        im = to_img(im)
        imgs.append((t, im.resize((panel, round(im.height * panel / im.width)),
                                  Image.LANCZOS if im.width > panel else Image.NEAREST)))
    cell_h = max(im.height for _, im in imgs)
    title_h = 6 + title_lines * (title_size + 6)
    sup_h = (sup_size + 16) if suptitle else 0
    W = gap + cols * (panel + gap)
    H = sup_h + gap + rows * (title_h + cell_h + gap)
    canvas = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(canvas)
    if suptitle:
        rtl(d, suptitle, W / 2, 8, size=sup_size)
    for n, (t, im) in enumerate(imgs):
        r, c = divmod(n, cols)
        x = gap + c * (panel + gap)
        y = sup_h + gap + r * (title_h + cell_h + gap)
        for i, line in enumerate(t.split("\n")):
            rtl(d, line, x + panel / 2, y + i * (title_size + 5), size=title_size)
        canvas.paste(im, (x, y + title_h))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)
    print("wrote", path, canvas.size)
    return canvas
