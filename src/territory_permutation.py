"""Permutation test for each territory's advantage over the all-phenotype median (Fig. 5a-c).

For each design, the observed statistic is the median eye-minus-CBC advantage across the
territory minus the median across every phenotype in that design. The null draws phenotype
sets of the same size, without replacement, from the same design, 20,000 times. One-sided
p = (count of null draws at least as large + 1) / (20,000 + 1). Territory membership comes
from the keyword sets in build_fig5_territory.py, so the figure and the test cannot drift.
Each cell has its own seeded generator, so a cell's p does not depend on the order run.
"""
import os, re
import numpy as np, pandas as pd

from config import CBCEYE_ROOT

HERE = CBCEYE_ROOT
B, SEED = 20000, 7

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_fig5_territory.py")).read()
NEURO = re.search(r"^NEURO=r'(.+)'$", src, re.M).group(1)
MICROR = re.search(r"^MICROR=r'(.+)'$", src, re.M).group(1)
EXCLUDE = "".join(re.findall(r"r'([^']+)'", src.split("EXCLUDE=(")[1].split(")")[0]))

xs = pd.read_parquet(f"{HERE}/landscape_full.parquet").drop_duplicates("phecode")
xs = xs[(xs.circularity_tier == "clean") & (xs.klass != "no signal")].copy(); xs["adv"] = xs.auc_eye - xs.auc_cbc
pr = pd.read_csv(f"{HERE}/prospective_landscape_B730_canonical.csv", dtype={"phecode": str})
pr = pr[pr.canonical].copy(); pr["adv"] = pr.c_eye - pr.c_cbc
nh = pd.read_csv(f"{HERE}/nhanes_validation_results.csv"); nh["adv"] = nh.eye - nh.cbc

LEGS = [("concurrent", xs, "description", NEURO, MICROR),
        ("index-anchored", pr, "description", NEURO, MICROR),
        ("NHANES", nh, "condition", r"[Ss]troke|memory|confusion|dementia", r"Albumin|CKD|kidney")]


def members(df, col, pat):
    d = df[col].astype(str)
    return df[d.str.contains(pat, case=False, na=False, regex=True)
              & ~d.str.contains(EXCLUDE, case=False, na=False, regex=True)]


def perm(allv, sub):
    rng = np.random.default_rng(SEED)
    obs = np.median(sub) - np.median(allv)
    null = np.array([np.median(rng.choice(allv, len(sub), replace=False)) for _ in range(B)]) - np.median(allv)
    return obs, (int((null >= obs).sum()) + 1) / (B + 1)


rows = []
for name, df, col, pn, pm in LEGS:
    allv = df.adv.dropna().to_numpy(float)
    for terr, pat in (("neurological", pn), ("microvascular / renal", pm)):
        sub = members(df, col, pat).adv.dropna().to_numpy(float)
        obs, p = perm(allv, sub)
        rows.append(dict(design=name, territory=terr, n=len(sub), n_design=len(allv), rel=obs, perm_p=p))
T = pd.DataFrame(rows)
T.to_csv(f"{HERE}/territory_permutation.csv", index=False)
print(T.round(4).to_string(index=False))
