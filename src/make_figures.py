"""FDR post-processing + figures for the phenome-wide landscape.

Applies Benjamini-Hochberg across diseases in pre-registered families, classifies
each disease (eye-specific / CBC-specific / synergy / comparable) gated on
FDR-significant Delta-AUROC, and renders the eye-vs-CBC scatter + organ-system
taxonomy. Headline stats use circularity_tier == 'clean'.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stats_utils as su

HERE = CBCEYE_ROOT
import circularity as circ
df = pd.read_parquet(f"{HERE}/landscape_full.parquet")
if "error" in df:
    df = df[df.error.isna()].copy()
# recompute tier with corrected logic (single-analyte dominance no longer forces tautological)
df["circularity_tier"] = [circ.tier(d, 0 if pd.isna(md) else md)
                          for d, md in zip(df.description, df.manifest_drop)]
print(f"diseases: {len(df)}  | clean: {(df.circularity_tier=='clean').sum()}  "
      f"tautological: {(df.circularity_tier=='tautological').sum()}  "
      f"expected: {df.circularity_tier.isin(['expected_cbc','expected_manifestation']).sum()}")

# ---- FDR across diseases ----
df["q_signal"] = su.benjamini_hochberg(df.p_best_gt_half.values)          # Family A: AUC>0.5
df["q_both_cbc"] = su.benjamini_hochberg(df.p_both_gt_cbc.values)         # Family B
df["q_both_eye"] = su.benjamini_hochberg(df.p_both_gt_eye.values)
df["q_eye_cbc"] = su.benjamini_hochberg(df.p_eye_vs_cbc_two.values)

def classify(r):
    if r.q_signal >= 0.05:
        return "no signal"
    eye_gt_cbc = r.q_eye_cbc < 0.05 and r.d_eye_vs_cbc > 0
    cbc_gt_eye = r.q_eye_cbc < 0.05 and r.d_eye_vs_cbc < 0
    both_adds_cbc = r.q_both_cbc < 0.05 and r.d_both_vs_cbc > 0.01
    both_adds_eye = r.q_both_eye < 0.05 and r.d_both_vs_eye > 0.01
    if both_adds_cbc and both_adds_eye:
        return "synergy"
    if eye_gt_cbc:
        return "eye-specific"
    if cbc_gt_eye:
        return "CBC-specific"
    return "comparable"

df["klass"] = df.apply(classify, axis=1)
df.to_parquet(f"{HERE}/landscape_full.parquet")  # write back with q + class

clean = df[df.circularity_tier == "clean"]
print("\n=== CLASS (clean diseases, FDR<0.05) ===")
print(clean.klass.value_counts().to_string())

def show(title, sub, by, cols=("description","group","n_cases","auc_eye","auc_cbc","auc_both")):
    print(f"\n=== {title} ===")
    print(sub.sort_values(by, ascending=False).head(12)[list(cols)].to_string(index=False))

show("EYE-SPECIFIC (clean) — eye sees what CBC can't", clean[clean.klass=="eye-specific"], "auc_eye")
show("SYNERGY (clean) — both modalities add", clean[clean.klass=="synergy"], "d_both_vs_cbc",
     cols=("description","group","n_cases","auc_eye","auc_cbc","auc_both","d_both_vs_cbc"))
show("CBC-SPECIFIC (clean)", clean[clean.klass=="CBC-specific"], "auc_cbc")

# ---- scatter ----
groups = sorted(df.group.dropna().unique())
cmap = {g: plt.cm.tab20(i % 20) for i, g in enumerate(groups)}
fig, ax = plt.subplots(figsize=(9.5, 9))
ax.fill_between([0.45,0.95],[0.45,0.95],0.95,alpha=0.04,color="red")
ax.fill_between([0.45,0.95],0.45,[0.45,0.95],alpha=0.04,color="blue")
ax.plot([0.45,0.95],[0.45,0.95],"k--",lw=1,alpha=0.4)
for tier, mk, alpha in [("clean","o",0.85), ("expected_manifestation","^",0.5),
                        ("expected_cbc","s",0.5), ("tautological","x",0.4)]:
    s = df[df.circularity_tier==tier]
    ax.scatter(s.auc_eye, s.auc_cbc, s=18+s.n_cases/60, marker=mk,
               c=[cmap[g] for g in s.group], alpha=alpha, edgecolor="k", lw=0.3,
               label=f"{tier} (n={len(s)})")
# annotate notable clean eye-specific + synergy
notable = pd.concat([clean[clean.klass=="eye-specific"].nlargest(7,"auc_eye"),
                     clean[clean.klass=="synergy"].nlargest(7,"d_both_vs_cbc"),
                     clean[clean.klass=="CBC-specific"].nlargest(4,"auc_cbc")])
for _, r in notable.iterrows():
    ax.annotate(r.description[:26], (r.auc_eye, r.auc_cbc), fontsize=6.5,
                xytext=(4,3), textcoords="offset points")
ax.text(0.47,0.90,"CBC-specific\n(eye blind)",color="#d62728",fontsize=11,weight="bold")
ax.text(0.83,0.47,"eye-specific\n(CBC blind)",color="#1f77b4",fontsize=11,weight="bold")
ax.set_xlabel("AUROC — eye findings only"); ax.set_ylabel("AUROC — CBC only")
ax.set_xlim(0.45,0.95); ax.set_ylim(0.45,0.95); ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=7, framealpha=0.9, title="circularity")
ax.set_title(f"Eye vs CBC across {len(df)} systemic phenotypes (MIMIC-IV, PheCode v1.2)\n"
             f"bubble=#cases; markers=circularity tier")
fig.tight_layout(); fig.savefig(f"{HERE}/landscape_phewide_fig.png", dpi=150)
print(f"\nsaved landscape_phewide_fig.png")

# ---- organ-system taxonomy ----
clean_sig = clean[clean.klass != "no signal"]
tax = clean_sig.groupby("group").klass.value_counts().unstack(fill_value=0)
tax["n"] = tax.sum(axis=1)
print("\n=== organ-system taxonomy (clean, signal) ===")
print(tax.sort_values("n", ascending=False).to_string())
