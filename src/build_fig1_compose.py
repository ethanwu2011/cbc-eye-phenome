"""Compose Figure 1 from its six panels, preserving the existing layout.

Panels a to c are BioRender artwork and are not redistributed in this repository,
so this script runs only where those exports are present. Panels d to f are the
Plotly renders produced by the scripts here, with their titles cropped.

Layout, matching the figure as designed:

    a  sieve        b  body map     c  three designs        (top row)
    d  per-condition landscape   |  e  class composition    (bottom left / right top)
                                 |  f  what the eye adds    (bottom right lower)

Panels a to c are the BioRender artwork. Panels d to f are the Plotly renders with
their titles cropped, because the panel letter and the caption carry that text.

Each panel is scaled to its box by width, and the box heights are solved so the
right-hand column (e over f) is exactly as tall as panel d. Nothing is stretched.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import FIGURES

FIG = FIGURES
OUT = f"{FIG}/Fig1_composed_v2.png"

WIDTH = 4000
MARGIN = 70
GUTTER = 80
ROWGAP = 110
LABEL = 78          # vertical space reserved above each row for the panel letters
LABEL_SIZE = 86
INK = (26, 28, 32)

TOP = [("a", "F1a_sieve_fixed.png"), ("b", "F1b_body.png"), ("c", "F1c_designs.png")]
LEFT = ("d", "P_d.png")
RIGHT = [("e", "P_e.png"), ("f", "P_f.png")]


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/SFNS.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def trim(path, thresh=735, pad=10):
    im = Image.open(f"{FIG}/{path}").convert("RGB")
    a = np.asarray(im).astype(int)
    ink = a.sum(axis=2) < thresh
    r = np.where(ink.any(axis=1))[0]
    c = np.where(ink.any(axis=0))[0]
    if not len(r):
        return im
    return im.crop((max(0, c.min() - pad), max(0, r.min() - pad),
                    min(im.width, c.max() + pad), min(im.height, r.max() + pad)))


def main():
    panels = {k: trim(f) for k, f in TOP + [LEFT] + RIGHT}
    ar = {k: im.width / im.height for k, im in panels.items()}

    content = WIDTH - 2 * MARGIN

    # ---- top row: three panels at a common height
    ar_sum = sum(ar[k] for k, _ in TOP)
    h1 = (content - GUTTER * (len(TOP) - 1)) / ar_sum
    top_w = {k: ar[k] * h1 for k, _ in TOP}

    # ---- bottom: solve so (e over f) equals the height of d
    #      h_d = w_L / ar_d        and       h_e + h_f + GUTTER = h_d
    #      with h_e = w_R / ar_e,  h_f = w_R / ar_f
    inv = 1 / ar["e"] + 1 / ar["f"]
    avail = content - GUTTER
    # w_L / ar_d = w_R * inv + GUTTER,  w_L + w_R = avail
    w_R = (avail / ar["d"] - GUTTER) / (inv + 1 / ar["d"])
    w_L = avail - w_R
    h2 = w_L / ar["d"]

    H = MARGIN + LABEL + int(h1) + ROWGAP + LABEL + int(h2) + MARGIN
    canvas = Image.new("RGB", (WIDTH, H), "white")
    dr = ImageDraw.Draw(canvas)
    font = _font(LABEL_SIZE)

    def place(key, x, y, w, h):
        im = panels[key].resize((int(w), int(h)), Image.LANCZOS)
        canvas.paste(im, (int(x), int(y)))
        dr.text((int(x), int(y) - LABEL + 4), key.upper() + ".", fill=INK, font=font)

    y = MARGIN + LABEL
    x = MARGIN
    for k, _ in TOP:
        place(k, x, y, top_w[k], h1)
        x += top_w[k] + GUTTER

    y2 = y + h1 + ROWGAP + LABEL
    place("d", MARGIN, y2, w_L, h2)
    xr = MARGIN + w_L + GUTTER
    he = w_R / ar["e"]
    hf = w_R / ar["f"]
    place("e", xr, y2, w_R, he)
    place("f", xr, y2 + he + GUTTER, w_R, hf)

    canvas.save(OUT)
    print(f"saved {OUT}  {canvas.size}")
    print(f"  top row height {h1:.0f}, bottom {h2:.0f}")
    print(f"  d {w_L:.0f}x{h2:.0f}   e {w_R:.0f}x{he:.0f}   f {w_R:.0f}x{hf:.0f}")
    for k in ("a", "b", "c", "d", "e", "f"):
        print(f"    panel {k}: source {panels[k].size}, aspect {ar[k]:.2f}")


if __name__ == "__main__":
    main()
