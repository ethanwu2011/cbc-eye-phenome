"""Per-phenotype INFERENCE for the prospective landscape.

The taxonomy in Fig 3A currently rests on arbitrary cutoffs (eye-vs-CBC +/-0.03, synergy
+0.015) with no uncertainty attached. This computes, for every phenotype, a bootstrap
95% CI and p-value for:
    delta_eye_cbc = C(demo+eye) - C(demo+CBC)      does one modality genuinely lead?
    delta_synergy = C(demo+eye+CBC) - max(single)  does combining genuinely help?

Method: fit the same 5-fold Coxnet models, keep the pooled out-of-fold risk scores, then
resample PATIENTS (not phenotypes) 400x and recompute both concordances on each replicate,
so the two models are compared on identical resamples and their correlation is preserved.
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
MIN_EV=40; NBOOT=400
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
elig=base[base.follow_days>B].copy(); EL=list(elig.index); pos={s:i for i,s in enumerate(EL)}
fu=elig.follow_days.to_numpy(float)
allc=sorted({c for s in EL for c in sub_eye.get(s,())}); ci_={c:i for i,c in enumerate(allc)}
EYE=np.zeros((len(EL),len(allc)),dtype=np.float32)
for s in EL:
    for c in sub_eye.get(s,()): EYE[pos[s],ci_[c]]=1
EYE=EYE[:,EYE.sum(0)>=20]
DEM=elig[DEMO].to_numpy(float); CB=elig[CBC].to_numpy(float)
DEM=np.where(np.isnan(DEM),np.nanmedian(DEM,0),DEM); CB=np.where(np.isnan(CB),np.nanmedian(CB,0),CB)
og={k:dict(zip(v.subject_id,v.onset_days)) for k,v in onset.groupby("phecode")}
print(f"eligible {len(EL):,} | eye feats {EYE.shape[1]}",flush=True)

def surv(e,t): return np.array([(bool(a),float(b)) for a,b in zip(e,t)],dtype=[("event",bool),("time",float)])
def oof_scores(X,Y):
    r=np.zeros(len(Y))
    for tr,te in KFold(5,shuffle=True,random_state=SEED).split(X):
        sc=StandardScaler().fit(X[tr])
        m=CoxnetSurvivalAnalysis(l1_ratio=0.9,alpha_min_ratio=0.1,n_alphas=12,max_iter=20000,tol=1e-6)
        try: m.fit(sc.transform(X[tr]),Y[tr]); r[te]=m.predict(sc.transform(X[te]))
        except Exception: return None
    return r

def run(p):
    o=og.get(p)
    if not o: return None
    keep=[i for i,s in enumerate(EL) if o.get(s,np.inf)>B]
    if len(keep)<500: return None
    ev=np.array([1 if o.get(EL[i],np.inf)!=np.inf else 0 for i in keep])
    tm=np.array([(o[EL[i]]-B) if o.get(EL[i],np.inf)!=np.inf else (fu[i]-B) for i in keep],float)
    k=tm>0; keep=[a for a,x in zip(keep,k) if x]; ev,tm=ev[k],tm[k]
    if ev.sum()<MIN_EV or (len(ev)-ev.sum())<100: return None
    Y=surv(ev,tm); ix=np.array(keep)
    se=oof_scores(np.hstack([DEM[ix],EYE[ix]]),Y)
    sc_=oof_scores(np.hstack([DEM[ix],CB[ix]]),Y)
    sb=oof_scores(np.hstack([DEM[ix],EYE[ix],CB[ix]]),Y)
    if se is None or sc_ is None or sb is None: return None
    C=lambda s,m=None: concordance_index_censored(Y["event"][m] if m is not None else Y["event"],
                                                  Y["time"][m] if m is not None else Y["time"],
                                                  s[m] if m is not None else s)[0]
    try: ce,cc,cb=C(se),C(sc_),C(sb)
    except Exception: return None
    rng=np.random.default_rng(SEED); n=len(Y)
    de=np.empty(NBOOT); ds=np.empty(NBOOT)
    for b in range(NBOOT):
        m=rng.integers(0,n,n)
        if Y["event"][m].sum()<8: de[b]=ds[b]=np.nan; continue
        try:
            a,c2,d=C(se,m),C(sc_,m),C(sb,m)
            de[b]=a-c2; ds[b]=d-max(a,c2)
        except Exception: de[b]=ds[b]=np.nan
    de,ds=de[~np.isnan(de)],ds[~np.isnan(ds)]
    if len(de)<100: return None
    return dict(phecode=p,description=DESC.get(p),group=GRP.get(p),n=len(Y),events=int(ev.sum()),
        c_eye=round(ce,3),c_cbc=round(cc,3),c_both=round(cb,3),
        d_eye_cbc=round(ce-cc,3),d_lo=round(np.percentile(de,2.5),3),d_hi=round(np.percentile(de,97.5),3),
        p_eye_cbc=round(2*min((de<0).mean(),(de>0).mean()),4),
        d_syn=round(cb-max(ce,cc),3),s_lo=round(np.percentile(ds,2.5),3),s_hi=round(np.percentile(ds,97.5),3),
        p_syn=round((ds<=0).mean(),4))

cands=[p for p in og if len(og[p])>=60]
print(f"computing bootstrap inference for {len(cands)} phenotypes ({NBOOT} resamples each)...",flush=True)
out=Parallel(n_jobs=6,verbose=1)(delayed(run)(p) for p in cands)
t=pd.DataFrame([r for r in out if r])
import stats_utils as su
t["q_eye_cbc"]=su.benjamini_hochberg(t.p_eye_cbc.fillna(1).to_numpy())
t["q_syn"]=su.benjamini_hochberg(t.p_syn.fillna(1).to_numpy())
t["sig_lead"]=np.where((t.q_eye_cbc<0.05)&(t.d_eye_cbc>0),"eye",
              np.where((t.q_eye_cbc<0.05)&(t.d_eye_cbc<0),"CBC","ns"))
t["sig_syn"]=(t.q_syn<0.05)&(t.d_syn>0)
t.to_csv(f"{HERE}/prospective_significance{TAG}.csv",index=False)
print(f"\n{'='*96}\nPER-PHENOTYPE INFERENCE ({len(t)} phenotypes, {NBOOT}-resample patient bootstrap, BH-FDR)\n{'='*96}")
print("  which modality significantly LEADS:",t.sig_lead.value_counts().to_dict())
print(f"  significant SYNERGY (both > best single, q<0.05): {int(t.sig_syn.sum())}/{len(t)}")
print(f"\n  top significantly EYE-led:")
for _,r in t[t.sig_lead=='eye'].sort_values('d_eye_cbc',ascending=False).head(12).iterrows():
    print(f"    {str(r.description)[:42]:<44}{str(r.group)[:15]:<17} ev={r.events:>4} Δ={r.d_eye_cbc:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}] q={r.q_eye_cbc:.3g}")
print("\nsaved prospective_significance.csv")
