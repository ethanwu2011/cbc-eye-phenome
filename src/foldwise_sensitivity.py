"""Fold-wise preprocessing sensitivity for the index-anchored landscape, all 476 phenotypes.

Identical to the published pipeline except that the eye vocabulary filter (>= 20 carriers)
and median imputation are fitted inside each training fold rather than once on the full
eligible set. Reports how far every C-index moves and whether any class assignment flips.
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

pub = pd.read_csv(f"{HERE}/prospective_landscape_B730.csv", dtype={"phecode": str})
pub = pub[pub.clean].set_index("phecode")


def surv(e, t):
    return np.array([(bool(a), float(b)) for a, b in zip(e, t)],
                    dtype=[("event", bool), ("time", float)])


def cstat(Xnum, eyeraw, Y):
    r = np.zeros(len(Y))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xnum):
        med = np.nanmedian(Xnum[tr], axis=0)
        Xtr = np.where(np.isnan(Xnum[tr]), med, Xnum[tr])
        Xte = np.where(np.isnan(Xnum[te]), med, Xnum[te])
        if eyeraw is not None:
            keep = eyeraw[tr].sum(0) >= 20
            Xtr = np.hstack([Xtr, eyeraw[tr][:, keep]])
            Xte = np.hstack([Xte, eyeraw[te][:, keep]])
        sc = StandardScaler().fit(Xtr)
        m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.1, n_alphas=12,
                                   max_iter=20000, tol=1e-6)
        try:
            m.fit(sc.transform(Xtr), Y[tr]); r[te] = m.predict(sc.transform(Xte))
        except Exception:
            return np.nan
    try:
        return concordance_index_censored(Y["event"], Y["time"], r)[0]
    except Exception:
        return np.nan


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
    return dict(phecode=p,
                c_base=cstat(DEM[ix], None, Y),
                c_eye=cstat(DEM[ix], EYEALL[ix], Y),
                c_cbc=cstat(np.hstack([DEM[ix], CB[ix]]), None, Y),
                c_both=cstat(np.hstack([DEM[ix], CB[ix]]), EYEALL[ix], Y))


def tax(r):
    if r.c_both > max(r.c_eye, r.c_cbc) + 0.015: return "synergy"
    if r.c_eye > r.c_cbc + 0.03: return "eye-specific"
    if r.c_cbc > r.c_eye + 0.03: return "CBC-specific"
    return "comparable"


out = Parallel(n_jobs=8)(delayed(run)(p) for p in pub.index)
fw = pd.DataFrame([r for r in out if r]).set_index("phecode")
both = fw.join(pub[["c_base", "c_eye", "c_cbc", "c_both", "klass"]], rsuffix="_pub", how="inner")
print(f"phenotypes compared: {len(both)} of {len(pub)}")
print(f"failed fits (NaN)  : {int(both[['c_base','c_eye','c_cbc','c_both']].isna().any(axis=1).sum())}\n")
for c in ("c_base", "c_eye", "c_cbc", "c_both"):
    d = (both[c] - both[c + "_pub"]).abs()
    print(f"  {c:<7} |delta|  median {d.median():.4f}   95th pct {d.quantile(.95):.4f}   max {d.max():.4f}")
both["klass_fw"] = both.apply(tax, axis=1)
flips = both[both.klass_fw != both.klass]
print(f"\nclass flips: {len(flips)} of {len(both)}")
if len(flips):
    print(flips[["klass", "klass_fw", "c_eye", "c_cbc", "c_both"]].to_string())
print("\npublished classes :", both.klass.value_counts().to_dict())
print("fold-wise classes :", both.klass_fw.value_counts().to_dict())
both.to_csv(f"{HERE}/foldwise_sensitivity_B730.csv")
print(f"\nsaved {HERE}/foldwise_sensitivity_B730.csv")
