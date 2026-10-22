"""Figure 2d, the eye-to-systemic flow map, as a Sankey.

Replaces the hand-drawn box-and-arrow diagram. Same data, same edge definition
(age- and sex-adjusted log odds ratio between an ophthalmic-sign family or a blood
count family and a systemic-disease family), drawn as flows whose width is the
strength of the association.

Only positive associations above THRESH are drawn. Negative and near-zero edges
carry no flow and would be meaningless as ribbon widths, so they are omitted and
the count of what is shown is printed for the legend.

No layout title: the panel letter and caption live in the figure legend.
"""
import pandas as pd
import plotly.graph_objects as go

from config import CBCEYE_ROOT, RESULTS, FIGURES

HERE = RESULTS
OUT = FIGURES
THRESH = 0.25

EYE = "#2B6E6E"
CBC = "#9C5B5B"
DIS = "#6E7478"
INK = "#1F2328"

SHORT = {
    "Optic-nerve (neuritis, atrophy)": "Optic nerve<br>neuritis, atrophy",
    "Oculomotor / pupil (ptosis, diplopia)": "Oculomotor and pupil<br>ptosis, diplopia",
    "Neuro-visual (field defect, nystagmus)": "Neuro-visual<br>field defect, nystagmus",
    "Retinal vascular occl. (artery/vein)": "Retinal vascular occlusion<br>artery, vein",
    "Retinal microvascular (diabetic/HTN retinopathy)": "Retinal microvascular<br>diabetic, hypertensive",
    "Ocular surface / autonomic (dry eye, tear film)": "Ocular surface<br>dry eye, tear film",
    "Structural / aging (cataract, glaucoma, AMD)": "Structural and aging<br>cataract, glaucoma, AMD",
    "Anemia indices (Hgb, RDW, MCV)": "Anemia indices<br>Hgb, RDW, MCV",
    "Inflammatory (WBC, neutrophils)": "Inflammatory<br>WBC, neutrophils",
    "Platelet / coag (platelets, RDW-SD)": "Platelet and coagulation<br>platelets, RDW-SD",
    "Demyelinating / CNS (MS)": "Demyelinating and CNS<br>multiple sclerosis",
    "Cerebrovascular (stroke, TIA)": "Cerebrovascular<br>stroke, TIA",
    "Peripheral / autonomic NEUROPATHY": "Peripheral and autonomic<br>neuropathy",
    "Neurodegenerative (Parkinson, dementia)": "Neurodegenerative<br>Parkinson, dementia",
    "Cardiometabolic-renal (CKD, HF, diabetes)": "Cardiometabolic and renal<br>CKD, heart failure, diabetes",
    "Hematologic / oncologic (anemia, leukemia)": "Hematologic and oncologic<br>anemia, leukemia",
    "Infection / inflammatory (sepsis, autoimmune)": "Infection and inflammatory<br>sepsis, autoimmune",
    "Hepatic / visceral (cirrhosis)": "Hepatic and visceral<br>cirrhosis",
}


# concise one-line labels for the in-figure panel, where horizontal space is tight.
# The parenthetical examples live in the figure legend instead.
TIGHT = {
    "Optic-nerve (neuritis, atrophy)": "Optic nerve",
    "Oculomotor / pupil (ptosis, diplopia)": "Oculomotor / pupil",
    "Neuro-visual (field defect, nystagmus)": "Neuro-visual",
    "Retinal vascular occl. (artery/vein)": "Retinal vascular occl.",
    "Retinal microvascular (diabetic/HTN retinopathy)": "Retinal microvascular",
    "Ocular surface / autonomic (dry eye, tear film)": "Ocular surface",
    "Structural / aging (cataract, glaucoma, AMD)": "Structural / aging",
    "Anemia indices (Hgb, RDW, MCV)": "Anemia indices",
    "Inflammatory (WBC, neutrophils)": "Inflammatory",
    "Platelet / coag (platelets, RDW-SD)": "Platelet / coagulation",
    "Demyelinating / CNS (MS)": "Demyelinating / CNS",
    "Cerebrovascular (stroke, TIA)": "Cerebrovascular",
    "Peripheral / autonomic NEUROPATHY": "Peripheral neuropathy",
    "Neurodegenerative (Parkinson, dementia)": "Neurodegenerative",
    "Cardiometabolic-renal (CKD, HF, diabetes)": "Cardiometabolic-renal",
    "Hematologic / oncologic (anemia, leukemia)": "Hematologic / oncologic",
    "Infection / inflammatory (sepsis, autoimmune)": "Infection / inflammatory",
    "Hepatic / visceral (cirrhosis)": "Hepatic / visceral",
}


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def main(tight=False, out_name="Fig2D_flowmap.png", width=2050, height=1180,
         fontsize=15, lmargin=320, rmargin=330, bmargin=108, keyy=-0.075):
    d = pd.read_csv(f"{HERE}/flowmap_edge_data.csv")
    keep = d[d.logOR > THRESH].copy().sort_values("logOR", ascending=False)
    GAP = 0.055          # blank band separating the eye block from the blood-count block
    PADF = 0.018         # blank band between adjacent nodes

    # order sources eye-first, then CBC, each by total outgoing strength
    strength = keep.groupby(["source", "modality"]).logOR.sum().reset_index()
    eye_src = strength[strength.modality == "eye"].sort_values("logOR", ascending=False).source.tolist()
    cbc_src = strength[strength.modality == "CBC"].sort_values("logOR", ascending=False).source.tolist()
    sources = eye_src + cbc_src
    targets = (keep.groupby("target").logOR.sum()
               .sort_values(ascending=False).index.tolist())

    nodes = sources + targets
    idx = {n: i for i, n in enumerate(nodes)}
    node_colors = ([EYE] * len(eye_src) + [CBC] * len(cbc_src) + [DIS] * len(targets))

    # ---- explicit vertical layout, so the eye block and the blood-count block
    #      are visually separate and the labels can be drawn outside the diagram
    out = keep.groupby("source").logOR.sum()
    inn = keep.groupby("target").logOR.sum()

    def stack(names, weights, gap_after=None):
        """Return {name: (y_center, height)} packed top to bottom in 0..1."""
        tot = sum(weights[n] for n in names)
        ngap = len(names) - 1 + (1 if gap_after is not None else 0)
        avail = 1.0 - PADF * (len(names) - 1) - (GAP if gap_after is not None else 0.0)
        pos, y = {}, 0.0
        for i, n in enumerate(names):
            h = avail * weights[n] / tot
            pos[n] = (y + h / 2, h)
            y += h + PADF
            if gap_after is not None and i == gap_after:
                y += GAP - PADF
        return pos

    src_pos = stack(sources, out, gap_after=len(eye_src) - 1)
    tgt_pos = stack(targets, inn)
    node_x = [0.001] * len(sources) + [0.999] * len(targets)
    node_y = ([src_pos[n][0] for n in sources] + [tgt_pos[n][0] for n in targets])

    link_color = [rgba(EYE if m == "eye" else CBC, 0.46) for m in keep.modality]

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=10, thickness=22,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=[""] * len(nodes), color=node_colors,
            x=node_x, y=node_y,
            customdata=[SHORT.get(n, n).replace("<br>", ", ") for n in nodes],
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=[idx[s] for s in keep.source],
            target=[idx[t] for t in keep.target],
            value=keep.logOR.tolist(),
            color=link_color,
            hovertemplate="%{source.label} to %{target.label}<br>log OR %{value:.2f}<extra></extra>",
        ),
    ))
    # ---- labels drawn outside the diagram, aligned to each node centre
    LAB = (lambda n: TIGHT.get(n, SHORT.get(n, n))) if tight else (lambda n: SHORT.get(n, n))
    ann = []
    for n in sources:
        ann.append(dict(x=-0.012, y=1 - src_pos[n][0], xref="paper", yref="paper",
                        text=LAB(n), showarrow=False,
                        xanchor="right", yanchor="middle", align="right",
                        font=dict(size=fontsize, color=INK)))
    for n in targets:
        ann.append(dict(x=1.012, y=1 - tgt_pos[n][0], xref="paper", yref="paper",
                        text=LAB(n), showarrow=False,
                        xanchor="left", yanchor="middle", align="left",
                        font=dict(size=fontsize, color=INK)))
    # ---- colour key, bottom left, below the diagram
    ann.append(dict(x=0.0, y=keyy, xref="paper", yref="paper", showarrow=False,
                    xanchor="left", yanchor="top", align="left",
                    text=(f"<span style='color:{EYE}'><b>&#9644; ophthalmic sign family</b></span>"
                          f"&nbsp;&nbsp;&nbsp;"
                          f"<span style='color:{CBC}'><b>&#9644; blood count family</b></span>"),
                    font=dict(size=fontsize)))
    if not tight:   # in the narrow panel this collides with the colour key,
                    # and the same statement is made in the figure legend
        ann.append(dict(x=1.0, y=keyy, xref="paper", yref="paper", showarrow=False,
                    xanchor="right", yanchor="top", align="right",
                    text=(f"ribbon width = age- and sex-adjusted log odds ratio, "
                          f"associations above {THRESH:.2f} shown"),
                    font=dict(size=fontsize-1, color="#5F6672")))

    fig.update_layout(
        font=dict(family="Helvetica Neue, Helvetica, Arial", size=15, color=INK),
        paper_bgcolor="white", plot_bgcolor="white",
        annotations=ann,
        margin=dict(l=lmargin, r=rmargin, t=26, b=bmargin),
        width=width, height=height,
    )
    fig.write_image(f"{OUT}/{out_name}", scale=2)
    fig.write_html(f"{OUT}/{out_name.replace('.png','.html')}")

    n_eye = int((keep.modality == "eye").sum())
    n_cbc = int((keep.modality == "CBC").sum())
    print(f"saved {OUT}/{out_name}")
    print(f"  {len(keep)} edges drawn above log OR {THRESH} "
          f"({n_eye} from the eye, {n_cbc} from the blood count), "
          f"of {len(d)} tested, {(d.logOR > 0).sum()} positive")
    print(f"  {len(sources)} source families, {len(targets)} disease families")
    print(f"  strongest: {keep.iloc[0].source} to {keep.iloc[0].target} at {keep.iloc[0].logOR:.2f}")


if __name__ == "__main__":
    # standalone, wide, with the descriptive two-line labels
    main()
    # panel-shaped for slotting into Figure 2d, concise labels
    # sized so that, scaled into the Figure 2d slot, its type matches the
    # neighbouring panels (about 26 px glyph height in the 4872 px composite)
    main(tight=True, out_name="Fig2D_flowmap_panel.png",
         width=1707, height=1524, fontsize=39, lmargin=480, rmargin=495,
         bmargin=210, keyy=-0.045)
