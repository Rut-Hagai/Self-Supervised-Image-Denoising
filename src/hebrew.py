"""Drawing Hebrew labels with PIL, with or without Raqm.

Matplotlib cannot lay out right-to-left text, so labels are drawn with PIL.
PIL does it only through the Raqm layout engine, which not every Pillow build
has; without it a Hebrew string comes out reversed. When Raqm is missing,
`shape` reorders the string itself: reverse it, then restore the order of any
Latin or numeric run inside it, so that "פי 3.7" does not become "פי 7.3".
"""
import os
import string

_FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "C:/Windows/Fonts/arial.ttf",
               "C:/Windows/Fonts/segoeui.ttf",
               "/Library/Fonts/Arial.ttf",
               "/System/Library/Fonts/Supplemental/Arial.ttf")


def font(size):
    """A TrueType font with Hebrew glyphs. Falls back to matplotlib's DejaVu."""
    from PIL import ImageFont
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    import matplotlib
    return ImageFont.truetype(
        os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data",
                     "fonts", "ttf", "DejaVuSans.ttf"), size)


_LTR = set(string.ascii_letters + string.digits + "εσλγηθ")
_NEUTRAL = set(" \t+-=*/.,:;_()[]{}<>|&%#@!?'\"")
_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{",
           "<": ">", ">": "<"}

_RAQM = None


def has_raqm():
    global _RAQM
    if _RAQM is None:
        try:
            from PIL import features
            _RAQM = bool(features.check("raqm"))
        except Exception:
            _RAQM = False
    return _RAQM


def has_hebrew(text):
    return any("֐" <= c <= "׿" for c in text)


def shape(text):
    """Logical to visual order for a right-to-left line, without Raqm."""
    out = list(reversed(text))
    n = len(out)
    i = 0
    while i < n:
        if out[i] not in _LTR:
            i += 1
            continue
        # a left-to-right run, including neutrals that sit inside it
        j = i + 1
        last = i + 1
        while j < n:
            if out[j] in _LTR:
                j += 1
                last = j
            elif out[j] in _NEUTRAL:
                j += 1
            else:
                break
        out[i:last] = out[i:last][::-1]
        i = last
    return "".join(_MIRROR.get(c, c) for c in out)


def _prepare(draw, text, font):
    """The string to draw and the keyword arguments to draw it with."""
    if not has_hebrew(text):
        return text, dict(font=font)  # e.g. "20.4 dB" stays left to right
    if has_raqm():
        kw = dict(font=font, direction="rtl", language="he")
        try:
            draw.textbbox((0, 0), text, **kw)
            return text, kw
        except Exception:
            pass
    return shape(text), dict(font=font)


def measure(draw, text, font):
    """(width, height) of the line as draw_rtl would draw it."""
    shown, kw = _prepare(draw, text, font)
    bb = draw.textbbox((0, 0), shown, **kw)
    return bb[2] - bb[0], bb[3] - bb[1]


def draw_rtl(draw, text, x, y, font, fill=(0, 0, 0), anchor="center"):
    """Draw one line. x is its centre or its left edge, y its top.
    Returns (width, height)."""
    shown, kw = _prepare(draw, text, font)
    bb = draw.textbbox((0, 0), shown, **kw)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    left = x - w / 2 if anchor == "center" else x
    draw.text((left - bb[0], y - bb[1]), shown, fill=fill, **kw)
    return w, h
