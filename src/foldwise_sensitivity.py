"""Fold-wise preprocessing sensitivity for the index-anchored landscape, all 449 phenotypes.

Identical to the published pipeline except that the eye vocabulary filter (>= 20 carriers)
and median imputation are fitted inside each training fold rather than once on the full
eligible set. Reports how far every C-index moves and whether any class assignment flips.

With few events a fold's Coxnet path can diverge at its smallest penalty (sksurv raises
"weights are too large, consider increasing alpha"). That fold is then refitted with the path
stopped at a larger penalty, and the `fallback` column records where this happened.

    python foldwise_sensitivity.py                  all phenotypes
    python foldwise_sensitivity.py 290,807          refit these and update their rows

Aggregates only, no patient rows printed.
"""
import os, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from config import CBCEYE_ROOT

HERE = CBCEYE_ROOT
B, SEED = 730, 42
RATIOS = (0.1, 0.2, 0.3, 0.5)   # 0.1 is the published setting; larger values stop the path earlier
OUT = f"{HERE}/foldwise_sensitivity_B730.csv"

base = pd.read_parquet(f"{HERE}/temporal_base.parquet")
CBC = [c for c in base.columns if c.startswith("CBC_")]
base["female"] = (base.gender == "F").astype(float)
DEMO = ["anchor_age", "female", "n_codes_le_t0", "n_adm_le_t0"]
base[DEMO] = base[DEMO].apply(pd.to_numeric, errors="coerce")
eye = pd.read_parquet(f"{HERE}/eye_at_t0.parquet")
sub_eye = eye.groupby("subject_id").code.apply(set).to_dict()
onset = pd.read_parquet(f"{HERE}/phecode_onset.parquet")
onset["phecode"] = onset.phecode.astype(str)

elig = base[base.follow_days > B].copy()
EL = list(elig.index); pos = {s: i for i, s in enumerate(EL)}
fu = elig.follow_days.to_numpy(float)
allcodes = sorted({c for s in EL for c in sub_eye.get(s, ())})
cidx = {c: i for i, c in enumerate(allcodes)}
EYEALL = np.zeros((len(EL), len(allcodes)), dtype=np.float32)   # unfiltered, filter per fold
for s in EL:
    for c in sub_eye.get(s, ()):
        EYEALL[pos[s], cidx[c]] = 1
DEM = elig[DEMO].to_numpy(float)
CB = elig[CBC].to_numpy(float)
og = {k: dict(zip(v.subject_id, v.onset_days)) for k, v in onset.groupby("phecode")}

pub = pd.read_csv(f"{HERE}/prospective_landscape_B730_canonical.csv", dtype={"phecode": str})
pub = pub[pub.canonical].set_index("phecode")


def surv(e, t):
    return np.array([(bool(a), float(b)) for a, b in zip(e, t)],
                    dtype=[("event", bool), ("time", float)])


def fit_fold(Xtr, ytr):
    for ratio in RATIOS:
        m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=ratio, n_alphas=12,
                                   max_iter=20000, tol=1e-6)
        try:
            return m.fit(Xtr, ytr), ratio
        except ArithmeticError:
            continue
    raise ArithmeticError("Coxnet diverged at every alpha_min_ratio tried")


def cstat(Xnum, eyeraw, Y):
    """C-index and the folds that needed a shorter path, as {fold: alpha_min_ratio}."""
    r = np.zeros(len(Y)); fb = {}
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(Xnum)):
        med = np.nanmedian(Xnum[tr], axis=0)
        Xtr = np.where(np.isnan(Xnum[tr]), med, Xnum[tr])
        Xte = np.where(np.isnan(Xnum[te]), med, Xnum[te])
        if eyeraw is not None:
            keep = eyeraw[tr].sum(0) >= 20
            Xtr = np.hstack([Xtr, eyeraw[tr][:, keep]])
            Xte = np.hstack([Xte, eyeraw[te][:, keep]])
        sc = StandardScaler().fit(Xtr)
        try:
            m, ratio = fit_fold(sc.transform(Xtr), Y[tr])
        except Exception:
            return np.nan, fb
        if ratio != RATIOS[0]:
            fb[k] = ratio
        r[te] = m.predict(sc.transform(Xte))
    try:
        return concordance_index_censored(Y["event"], Y["time"], r)[0], fb
    except Exception:
        return np.nan, fb


def run(p):
    o = og.get(p)
    if not o:
        return None
    keep = [i for i, s in enumerate(EL) if o.get(s, np.inf) > B]
    if len(keep) < 500:
        return None
    ev = np.array([1 if o.get(EL[i], np.inf) != np.inf else 0 for i in keep])
    tm = np.array([(o[EL[i]] - B) if o.get(EL[i], np.inf) != np.inf else (fu[i] - B)
                   for i in keep], float)
    m = tm > 0
    keep = [k for k, x in zip(keep, m) if x]; ev, tm = ev[m], tm[m]
    Y = surv(ev, tm); ix = np.array(keep)
    fits = dict(c_base=cstat(DEM[ix], None, Y),
                c_eye=cstat(DEM[ix], EYEALL[ix], Y),
                c_cbc=cstat(np.hstack([DEM[ix], CB[ix]]), None, Y),
                c_both=cstat(np.hstack([DEM[ix], CB[ix]]), EYEALL[ix], Y))
    note = "; ".join(f"{m[2:]} fold {k + 1} at alpha_min_ratio {v}"
                     for m, (_, fb) in fits.items() for k, v in fb.items())
    return dict(phecode=p, **{m: c for m, (c, _) in fits.items()}, fallback=note)


def tax(r):
    if np.isnan([r.c_eye, r.c_cbc, r.c_both]).any(): return np.nan
    if r.c_both > max(r.c_eye, r.c_cbc) + 0.015: return "synergy"
    if r.c_eye > r.c_cbc + 0.03: return "eye-specific"
    if r.c_cbc > r.c_eye + 0.03: return "CBC-specific"
    return "comparable"


ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None
todo = ONLY or list(pub.index)
out = Parallel(n_jobs=min(8, len(todo)))(delayed(run)(p) for p in todo)
fw = pd.DataFrame([r for r in out if r]).set_index("phecode")
both = fw.join(pub[["c_base", "c_eye", "c_cbc", "c_both", "klass"]], rsuffix="_pub", how="inner")
if ONLY:
    prev = pd.read_csv(OUT, dtype={"phecode": str}).set_index("phecode")
    both = pd.concat([prev.drop(index=both.index, errors="ignore"), both])
    both = both.loc[[p for p in pub.index if p in both.index]]
print(f"phenotypes compared: {len(both)} of {len(pub)}")
print(f"failed fits (NaN)  : {int(both[['c_base','c_eye','c_cbc','c_both']].isna().any(axis=1).sum())}\n")
for c in ("c_base", "c_eye", "c_cbc", "c_both"):
    d = (both[c] - both[c + "_pub"]).abs()
    print(f"  {c:<7} |delta|  median {d.median():.4f}   95th pct {d.quantile(.95):.4f}   max {d.max():.4f}")
both["fallback"] = both.get("fallback", pd.Series("", index=both.index)).fillna("")
both["klass_fw"] = both.apply(tax, axis=1)
flips = both[both.klass_fw != both.klass]
print(f"\nclass flips: {len(flips)} of {len(both)}")
if len(flips):
    print(flips[["klass", "klass_fw", "c_eye", "c_cbc", "c_both"]].to_string())
print("\npublished classes :", both.klass.value_counts().to_dict())
print("fold-wise classes :", both.klass_fw.value_counts().to_dict())
both = both[["c_base", "c_eye", "c_cbc", "c_both", "c_base_pub", "c_eye_pub", "c_cbc_pub",
             "c_both_pub", "klass", "klass_fw", "fallback"]]
both.to_csv(OUT)
print(f"\nsaved {OUT}")
print(f"fits that needed a shorter path: {int((both.fallback != '').sum())}")
print(both[both.fallback != ""][["fallback"]].to_string())

# the numbers quoted in Methods
d = {c: both[c] - both[c + "_pub"] for c in ("c_base", "c_eye", "c_cbc", "c_both")}
print(f"\nMETHODS  compared {len(both)}, complete {int(both[['c_base','c_eye','c_cbc','c_both']].notna().all(axis=1).sum())}")
print(f"  median |change|  base {d['c_base'].abs().median():.4f}  cbc {d['c_cbc'].abs().median():.4f}"
      f"  eye {d['c_eye'].abs().median():.4f}   mean change eye {d['c_eye'].mean():+.4f}")
gap = both.c_eye_pub - both.c_cbc_pub
syn = both.c_both_pub - np.maximum(both.c_eye_pub, both.c_cbc_pub)
fl = both.klass_fw != both.klass
near = ((gap.abs() - 0.03).abs() < 0.01) | ((syn - 0.015).abs() < 0.01)
print(f"  eye-specific {int((both.klass == 'eye-specific').sum())} -> {int((both.klass_fw == 'eye-specific').sum())};"
      f"  class changed {int(fl.sum())} of {len(both)}, {int((fl & near).sum())} of them within 0.01 of a"
      f" class threshold (published C-indices)")
xs = pd.read_parquet(f"{HERE}/landscape_full.parquet").drop_duplicates("phecode")
xs["phecode"] = xs.phecode.astype(str)
xs = xs[(xs.circularity_tier == "clean") & (xs.klass != "no signal")]
m = (xs.assign(adv=xs.auc_eye - xs.auc_cbc)[["phecode", "adv"]]
     .merge(both.reset_index(), on="phecode"))
from scipy import stats
print(f"  concordance with the concurrent landscape on {len(m)} shared phenotypes: Spearman rho"
      f" {stats.spearmanr(m.adv, m.c_eye - m.c_cbc).statistic:.3f} fold-wise against"
      f" {stats.spearmanr(m.adv, m.c_eye_pub - m.c_cbc_pub).statistic:.3f} published")
