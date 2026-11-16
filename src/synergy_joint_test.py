"""Stricter version of the synergy gate (Methods, AUROC inference).

The taxonomy in make_figures.py calls a phenotype synergy when the joint model exceeds the CBC
model and the eye model, each at Benjamini-Hochberg q < 0.05 within its own family of tests and
by more than 0.01 AUROC. Here the two contrasts are treated as one claim: for each phenotype the
larger of the two p-values is taken, and Benjamini-Hochberg is applied once to those. Phenotypes
that fail this gate fall through to the eye-versus-CBC rule, exactly as in make_figures.py.

Reads results/landscape_full.csv, prints the class counts under both gates, and writes
results/synergy_joint_test.csv (one row per analyzed phenotype).
"""
import os
import numpy as np, pandas as pd

import stats_utils as su
from config import RESULTS

df = pd.read_csv(os.path.join(RESULTS, "landscape_full.csv"), dtype={"phecode": str})
df["p_joint"] = np.maximum(df.p_both_gt_cbc, df.p_both_gt_eye)
df["q_joint"] = su.benjamini_hochberg(df.p_joint.values)          # same family as the taxonomy: all screened phenotypes


def classify(r):
    if r.klass == "no signal":
        return "no signal"
    if r.q_joint < 0.05 and r.d_both_vs_cbc > 0.01 and r.d_both_vs_eye > 0.01:
        return "synergy"
    if r.q_eye_cbc < 0.05 and r.d_eye_vs_cbc > 0:
        return "eye-specific"
    if r.q_eye_cbc < 0.05 and r.d_eye_vs_cbc < 0:
        return "CBC-specific"
    return "comparable"


df["klass_joint"] = df.apply(classify, axis=1)
a = df[(df.circularity_tier == "clean") & (df.klass != "no signal")]

order = ["CBC-specific", "eye-specific", "synergy", "comparable"]
tab = pd.DataFrame({"two-family gate": a.klass.value_counts(), "joint gate": a.klass_joint.value_counts()}).loc[order]
print(f"analyzed phenotypes: {len(a)}\n{tab.to_string()}")
for col, name in (("klass", "two-family gate"), ("klass_joint", "joint gate")):
    print(f"eye-specific or synergy, {name}: {a[col].isin(['eye-specific', 'synergy']).sum()}")
moved = a[a.klass != a.klass_joint]
print(f"\nreclassified under the joint gate: {len(moved)}")
print(moved[["description", "group", "p_both_gt_cbc", "p_both_gt_eye", "q_joint", "klass", "klass_joint"]]
      .round(4).to_string(index=False))

a[["phecode", "description", "group", "p_both_gt_cbc", "p_both_gt_eye", "p_joint", "q_joint", "klass", "klass_joint"]].to_csv(
    os.path.join(RESULTS, "synergy_joint_test.csv"), index=False)
