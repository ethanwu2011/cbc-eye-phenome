"""SCREENING / LEAD-TIME analysis — the point of the paper if it is a screening paper.

Current §2.5 uses a 90-day blanking window and a 1-year horizon, so "precedes" means
"precedes by >=3 months". That is a weak screening claim. Here we sweep the BLANKING
window (the required lead time) out to 2 YEARS and the horizon out to 5 years, using
only patients with complete observed follow-up to the horizon.

For each (blank B, horizon H):
   eligible = follow_days >= H  AND  no onset of the phenotype at or before B
   case     = first onset in (B, H]
   control  = no onset at or before H
Features at T0 only: base = age/sex/utilisation ; +eye ; +CBC ; +both.
Pooled 5-fold OOF AUROC, in-fold eye vocabulary (leakage control), DeLong both-vs-base,
BH-FDR within each cell.

If the eye+CBC increment survives B=730, the ophthalmic finding preceded the coded
diagnosis by more than two years -> a genuine screening lead time, and one that
reverse causation / ocular manifestation cannot explain.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import stats_utils as su

HERE=CBCEYE_ROOT; SEED=42; MIN_CASES=60
base=pd.read_parquet(f"{HERE}/temporal_base.parquet")
eye=pd.read_parquet(f"{HERE}/eye_at_t0.parquet")
onset=pd.read_parquet(f"{HERE}/phecode_onset.parquet")
pheinfo=None
try:
    import phecode_map as pm; _,_,pheinfo=pm.load_maps()
except Exception: pass

CBC=[c for c in base.columns if c.startswith("CBC_")]
base["female"]=(base.gender=="F").astype(float)
DEMO=["anchor_age","female","n_codes_le_t0","n_adm_le_t0"]
base[DEMO]=base[DEMO].apply(pd.to_numeric,errors="coerce")
sub_eye=eye.groupby("subject_id").code.apply(set).to_dict()
onset_g={k:v for k,v in onset.groupby("phecode")}
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)

def oof(X,y):
    s=np.zeros(len(y))
    for tr,te in cv.split(X,y):
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med)
        Xi=np.where(np.isnan(X),med,X)
        m=HistGradientBoostingClassifier(max_iter=200,early_stopping=True,random_state=SEED)
        m.fit(Xi[tr],y[tr]); s[te]=m.predict_proba(Xi[te])[:,1]
    return s

def eye_matrix(subs,y):
    """in-fold vocabulary: codes selected on TRAIN rows only (label-blind frequency)."""
    codes=sorted({c for s in subs for c in sub_eye.get(s,())})
    idx={c:i for i,c in enumerate(codes)}
    M=np.zeros((len(subs),len(codes)),dtype=np.float32)
    for i,s in enumerate(subs):
        for c in sub_eye.get(s,()):
            M[i,idx[c]]=1
    keep=M.sum(0)>=20                     # frequency filter, no labels used
    return M[:,keep]

CELLS=[(90,365,"≥3 mo lead · 1 yr"),(365,730,"≥1 yr lead · 2 yr"),
       (365,1095,"≥1 yr lead · 3 yr"),(730,1825,"≥2 yr lead · 5 yr")]
rows=[]
for B,H,label in CELLS:
    elig_all=base[base.follow_days>=H]
    print(f"\n=== {label}  (blank {B}d, horizon {H}d) — {len(elig_all):,} patients with complete follow-up ===",flush=True)
    cell=[]
    for phe,g in onset_g.items():
        o=dict(zip(g.subject_id,g.onset_days))
        subs=[s for s in elig_all.index if o.get(s,np.inf)>B]      # drop prevalent / too-soon
        if len(subs)<200: continue
        y=np.array([1 if B<o.get(s,np.inf)<=H else 0 for s in subs],float)
        n_case=int(y.sum())
        if n_case<MIN_CASES or (len(y)-n_case)<MIN_CASES: continue
        bb=base.loc[subs]
        Xd=bb[DEMO].to_numpy(float); Xc=bb[CBC].to_numpy(float); Xe=eye_matrix(subs,y)
        sc={"base":oof(Xd,y),"eye":oof(np.hstack([Xd,Xe]),y),
            "cbc":oof(np.hstack([Xd,Xc]),y),"both":oof(np.hstack([Xd,Xe,Xc]),y)}
        D=su.delong_analyze(y,sc); au=D["auc"]
        def gd(a,b):
            """signed delta (a-b) and p(a>b); delong_analyze only stores one key order."""
            k=D["delta"].get(f"{a}-{b}")
            if k: return k["delta"],k["p_gt"]
            k=D["delta"].get(f"{b}-{a}")
            if k: return -k["delta"],1.0-k["p_gt"]
            return au[a]-au[b],np.nan
        _,p_both_base=gd("both","base")
        d={"p_gt":p_both_base}
        cell.append(dict(cell=label,blank=B,horizon=H,phecode=phe,n=len(subs),cases=n_case,
            base=round(au["base"],3),eye=round(au["eye"],3),cbc=round(au["cbc"],3),both=round(au["both"],3),
            gain=round(au["both"]-au["base"],3),eye_over_cbc=round(au["both"]-au["cbc"],3),
            p=d.get("p_gt",np.nan)))
    if not cell: print("  (no phenotype met the case floor)",flush=True); continue
    c=pd.DataFrame(cell)
    c["q"]=su.benjamini_hochberg(c.p.fillna(1).to_numpy())
    c=c.sort_values("gain",ascending=False); rows.append(c)
    sig=c[(c.q<0.05)&(c.gain>0)]
    print(f"  phenotypes modelled {len(c)} | FDR-significant gain over baseline: {len(sig)}",flush=True)
    print(f"  median gain {c.gain.median():+.3f} | best {c.gain.max():+.3f}",flush=True)
    for _,r in c.head(8).iterrows():
        nm=(pheinfo.get(r.phecode,{}) or {}).get("description",r.phecode) if isinstance(pheinfo,dict) else r.phecode
        print(f"    {str(nm)[:44]:<46} n={r.n:>5} cases={r.cases:>4} base {r.base:.3f} -> both {r.both:.3f} (Δ{r.gain:+.3f}, q={r.q:.3g})",flush=True)

if rows:
    out=pd.concat(rows,ignore_index=True)
    if isinstance(pheinfo,dict):
        out["description"]=[ (pheinfo.get(p,{}) or {}).get("description",p) for p in out.phecode ]
    out.to_csv(f"{HERE}/temporal_leadtime.csv",index=False)
    print("\n=== LEAD-TIME SUMMARY (does the signal survive longer required lead?) ===")
    for lab,g in out.groupby("cell",sort=False):
        s=g[(g.q<0.05)&(g.gain>0)]
        print(f"  {lab:<22} modelled {len(g):>3} | FDR-sig {len(s):>3} | median gain {g.gain.median():+.3f} | median eye-over-CBC {g.eye_over_cbc.median():+.3f}")
    print("\nsaved temporal_leadtime.csv")
