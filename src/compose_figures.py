"""Compose Figures 1 and 2 properly.

The previous composer tiled panels into fixed cells, so panels with different aspect
ratios floated inside boxes with uneven whitespace. This one trims each panel to its
ink, scales it to a common row height, and packs rows tightly, so the composite reads
as one figure rather than a contact sheet.

Figure 1  a study design and the circularity sieve (BioRender)
          b the per-disease landscape
          c the organ-level landscape
          d taxonomy by organ system

Figure 2  a non-redundancy   b feature independence   c additivity
          d eye-to-systemic flow map   e per-disease drivers   f what the eye adds
"""
from config import CBCEYE_ROOT
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIG=f"{CBCEYE_ROOT}/final_manuscript/figures"
GAP=54          # gutter between panels
LAB=76          # space above each row for the panel letter
MARGIN=40
LABEL_DX=6

def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p,sz)
            except Exception: pass
    return ImageFont.load_default()

def trim(path, thresh=735, pad=14):
    """crop a panel to its ink so panels of different aspect sit flush"""
    im=Image.open(path).convert("RGB")
    a=np.asarray(im).astype(int)
    ink=a.sum(axis=2) < thresh
    rows=np.where(ink.any(axis=1))[0]; cols=np.where(ink.any(axis=0))[0]
    if len(rows)==0: return im
    t,b=max(0,rows[0]-pad), min(a.shape[0],rows[-1]+pad)
    l,r=max(0,cols[0]-pad), min(a.shape[1],cols[-1]+pad)
    return im.crop((l,t,r,b))

def compose(rows, out, row_h=1180, width=None):
    """rows = list of lists of (label, path). Each row is scaled to row_h."""
    laid=[]
    for row in rows:
        ims=[]
        for lab,p in row:
            if not os.path.exists(p):
                print(f"   MISSING {p}"); continue
            im=trim(p)
            w=int(im.width*row_h/im.height)
            ims.append((lab, im.resize((w,row_h), Image.LANCZOS)))
        laid.append(ims)
    if width is None:
        width=max(sum(im.width for _,im in r)+GAP*(len(r)-1) for r in laid)+2*MARGIN
    # rescale any row that overflows the chosen width
    final=[]
    for ims in laid:
        tot=sum(im.width for _,im in ims)+GAP*(len(ims)-1)
        avail=width-2*MARGIN
        if tot>avail:
            k=avail/tot
            ims=[(lab, im.resize((int(im.width*k),int(im.height*k)), Image.LANCZOS)) for lab,im in ims]
        final.append(ims)
    height=MARGIN+sum(LAB+max(im.height for _,im in r)+GAP for r in final)-GAP+MARGIN
    canvas=Image.new("RGB",(width,height),"white")
    dr=ImageDraw.Draw(canvas); font=_font(62)
    y=MARGIN
    for ims in final:
        rh=max(im.height for _,im in ims)
        # centre the row horizontally
        tot=sum(im.width for _,im in ims)+GAP*(len(ims)-1)
        x=(width-tot)//2
        for lab,im in ims:
            dr.text((x+LABEL_DX,y),lab,fill=(20,20,20),font=font)
            canvas.paste(im,(x, y+LAB+(rh-im.height)//2))
            x+=im.width+GAP
        y+=LAB+rh+GAP
    canvas.save(out)
    print(f"  {os.path.basename(out):<22} {canvas.size[0]}x{canvas.size[1]}")
    return canvas

def main():
    # ---- Figure 1, BioRender sieve as panel a, then the three landscape results
    compose([[("a",f"{FIG}/F1a_sieve.png"), ("b",f"{FIG}/Fig1B_landscape.png")],
             [("c",f"{FIG}/Fig1B_organ_landscape.png"), ("d",f"{FIG}/Fig1C_taxonomy.png")]],
            f"{FIG}/Fig1_composed.png", row_h=1180)
    # ---- Figure 2, three tests on top, mechanism and per-disease beneath
    compose([[("a",f"{FIG}/Fig2A_nonredundancy.png"), ("b",f"{FIG}/Fig2B_eye_to_cbc.png"),
              ("c",f"{FIG}/Fig2C_additivity.png")],
             [("d",f"{FIG}/Fig2D_flowmap.png"), ("e",f"{FIG}/Fig2E_vignettes.png")],
             [("f",f"{FIG}/Fig_eye_gain.png")]],
            f"{FIG}/Fig2_composed.png", row_h=1020)

if __name__=="__main__": main()
