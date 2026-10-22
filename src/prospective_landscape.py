"""THE PROSPECTIVE LANDSCAPE — the cross-sectional Figure 1, redone in TIME.

The paper's novelty is not "the eye predicts microvascular disease" (partly known). It is the
EYE-vs-CBC COMPARISON: which of two routine, near-free data streams carries which organ axis.
This runs that comparison phenome-wide and PROSPECTIVELY.

Design: time-to-event with censoring. Predictors at T0 only. Blanking B = 365 d, so every
event occurs at least a year after the ophthalmic encounter; events then accrue over all
remaining follow-up (to 14.5 y). Regularized Cox (Coxnet) for speed at phenome scale,
parallelized over phenotypes; the headline hits are re-confirmed with gradient-boosted
survival separately.

Per phenotype: C(base) / C(eye) / C(CBC) / C(both), and the same 4-way taxonomy as the
cross-sectional landscape, so the two figures can be read side by side.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

import sys
HERE=CBCEYE_ROOT; SEED=42
B=int(sys.argv[1]) if len(sys.argv)>1 else 365
MIN_EV=40
TAG="" if B==365 else f"_B{B}"
base=pd.read_parquet(f"{HERE}/temporal_base.parquet")
eye=pd.read_parquet(f"{HERE}/eye_at_t0.parquet")
onset=pd.read_parquet(f"{HERE}/phecode_onset.parquet"); onset["phecode"]=onset.phecode.astype(str)
meta=pd.read_parquet(f"{HERE}/phecode_long.parquet",columns=["phecode","description","group"]).drop_duplicates("phecode")
meta["phecode"]=meta.phecode.astype(str)
DESC=dict(zip(meta.phecode,meta.description)); GRP=dict(zip(meta.phecode,meta.group))
CBC=[c for c in base.columns if c.startswith("CBC_")]
base["female"]=(base.gender=="F").astype(float)
DEMO=["anchor_age","female","n_codes_le_t0","n_adm_le_t0"]
base[DEMO]=base[DEMO].apply(pd.to_numeric,errors="coerce")
sub_eye=eye.groupby("subject_id").code.apply(set).to_dict()

elig=base[base.follow_days>B].copy()
EL=list(elig.index); pos={s:i for i,s in enumerate(EL)}
fu=elig.follow_days.to_numpy(float)
# eye matrix built ONCE on the eligible set (label-blind frequency filter)
allcodes=sorted({c for s in EL for c in sub_eye.get(s,())})
cidx_={c:i for i,c in enumerate(allcodes)}
EYE=np.zeros((len(EL),len(allcodes)),dtype=np.float32)
for s in EL:
    for c in sub_eye.get(s,()):
        EYE[pos[s],cidx_[c]]=1
EYE=EYE[:,EYE.sum(0)>=20]
DEM=elig[DEMO].to_numpy(float)
CB=elig[CBC].to_numpy(float)
DEM=np.where(np.isnan(DEM),np.nanmedian(DEM,0),DEM); CB=np.where(np.isnan(CB),np.nanmedian(CB,0),CB)
print(f"eligible {len(EL):,} | eye features {EYE.shape[1]} | CBC {CB.shape[1]}",flush=True)
og={k:dict(zip(v.subject_id,v.onset_days)) for k,v in onset.groupby("phecode")}

def surv(e,t): return np.array([(bool(a),float(b)) for a,b in zip(e,t)],dtype=[("event",bool),("time",float)])
def cstat(X,Y):
    r=np.zeros(len(Y))
    for tr,te in KFold(5,shuffle=True,random_state=SEED).split(X):
        sc=StandardScaler().fit(X[tr])
        m=CoxnetSurvivalAnalysis(l1_ratio=0.9,alpha_min_ratio=0.1,n_alphas=12,max_iter=20000,tol=1e-6)
        try:
            m.fit(sc.transform(X[tr]),Y[tr]); r[te]=m.predict(sc.transform(X[te]))
        except Exception: return np.nan
    try: return concordance_index_censored(Y["event"],Y["time"],r)[0]
    except Exception: return np.nan

def run(p):
    o=og.get(p)
    if not o: return None
    keep=[i for i,s in enumerate(EL) if o.get(s,np.inf)>B]
    if len(keep)<500: return None
    ev=np.array([1 if o.get(EL[i],np.inf)!=np.inf else 0 for i in keep])
    tm=np.array([(o[EL[i]]-B) if o.get(EL[i],np.inf)!=np.inf else (fu[i]-B) for i in keep],float)
    m=tm>0; keep=[k for k,x in zip(keep,m) if x]; ev,tm=ev[m],tm[m]
    if ev.sum()<MIN_EV or (len(ev)-ev.sum())<100: return None
    Y=surv(ev,tm); ix=np.array(keep)
    cb=cstat(DEM[ix],Y); ce=cstat(np.hstack([DEM[ix],EYE[ix]]),Y)
    cc=cstat(np.hstack([DEM[ix],CB[ix]]),Y); co=cstat(np.hstack([DEM[ix],EYE[ix],CB[ix]]),Y)
    if any(np.isnan(x) for x in (cb,ce,cc,co)): return None
    return dict(phecode=p,description=DESC.get(p),group=GRP.get(p),n=len(ev),events=int(ev.sum()),
        c_base=round(cb,3),c_eye=round(ce,3),c_cbc=round(cc,3),c_both=round(co,3),
        eye_gain=round(ce-cb,3),cbc_gain=round(cc-cb,3),both_gain=round(co-cb,3),
        eye_minus_cbc=round(ce-cc,3),eye_over_cbc=round(co-cc,3))

cands=[p for p in og if len(og[p])>=60]
print(f"screening {len(cands)} phenotypes...",flush=True)
out=Parallel(n_jobs=6,verbose=1)(delayed(run)(p) for p in cands)
t=pd.DataFrame([r for r in out if r])
def tax(r):
    if r.c_both>max(r.c_eye,r.c_cbc)+0.015: return "synergy"
    if r.c_eye>r.c_cbc+0.03: return "eye-specific"
    if r.c_cbc>r.c_eye+0.03: return "CBC-specific"
    return "comparable"
t["klass"]=t.apply(tax,axis=1)
t=t.sort_values("both_gain",ascending=False)
t.to_csv(f"{HERE}/prospective_landscape{TAG}.csv",index=False)
print(f"\n{'='*112}\nPROSPECTIVE LANDSCAPE — {len(t)} phenotypes, every event >=1 year after the ophthalmic encounter\n{'='*112}")
print("  taxonomy:",t.klass.value_counts().to_dict())
print(f"  eye leads (C_eye > C_cbc + .03): {int((t.c_eye>t.c_cbc+0.03).sum())}   CBC leads: {int((t.c_cbc>t.c_eye+0.03).sum())}")
print(f"  median eye gain {t.eye_gain.median():+.3f} | median CBC gain {t.cbc_gain.median():+.3f} | median both gain {t.both_gain.median():+.3f}")
print(f"\n  --- by organ system (median C gain over baseline) ---")
print(f"  {'organ':<26}{'n':>4}{'eye':>9}{'CBC':>9}{'eye-CBC':>10}   leads")
for g,x in sorted(t.groupby("group"),key=lambda kv:-(kv[1].eye_gain.median()-kv[1].cbc_gain.median())):
    if len(x)<3: continue
    e,c=x.eye_gain.median(),x.cbc_gain.median()
    print(f"  {str(g)[:25]:<26}{len(x):>4}{e:>+9.3f}{c:>+9.3f}{e-c:>+10.3f}   {'EYE' if e>c+0.01 else ('CBC' if c>e+0.01 else 'both')}")
print(f"\n  --- top 20 where the EYE leads prospectively ---")
for _,r in t[t.c_eye>t.c_cbc+0.03].sort_values("eye_minus_cbc",ascending=False).head(20).iterrows():
    print(f"  {str(r.description)[:44]:<46}{str(r.group)[:16]:<18} ev={r.events:>4} eye {r.c_eye:.3f} vs cbc {r.c_cbc:.3f} (+{r.eye_minus_cbc:.3f}) both {r.c_both:.3f}")
print("\nsaved prospective_landscape.csv")
