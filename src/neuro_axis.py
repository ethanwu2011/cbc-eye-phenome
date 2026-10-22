"""The NEURO AXIS, done properly.

The first pass discarded the neurological phenotypes with an arbitrary 60-incident-case
floor, which threw away polyneuropathy (55/45/69/56), late effects of cerebrovascular
disease (58/62/82/59), dementias (51/...) and stroke (23/21/30/38) — i.e. most of the axis.

Fixes:
 (1) COMPOSITE clinical endpoints (standard practice: pool the components of one syndrome),
     which is both clinically sensible and where the power actually is.
 (2) Case floor lowered to 30 with DeLong 95% CIs reported, so small strata are shown
     honestly rather than silently dropped.
 (3) Every endpoint is run with the FULL ophthalmic vocabulary AND with the neuro-ophthalmic
     SEQUELA CODES ABLATED — because for the neuro axis specifically, the question is whether
     any antecedent signal survives removal of post-event ocular manifestations.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import stats_utils as su

HERE=CBCEYE_ROOT; SEED=42; MIN_CASES=30
base=pd.read_parquet(f"{HERE}/temporal_base.parquet")
eye=pd.read_parquet(f"{HERE}/eye_at_t0.parquet")
onset=pd.read_parquet(f"{HERE}/phecode_onset.parquet"); onset["phecode"]=onset.phecode.astype(str)
meta=pd.read_parquet(f"{HERE}/phecode_long.parquet",columns=["phecode","description"]).drop_duplicates("phecode")
meta["phecode"]=meta.phecode.astype(str); D=dict(zip(meta.phecode,meta.description))

CBC=[c for c in base.columns if c.startswith("CBC_")]
base["female"]=(base.gender=="F").astype(float)
DEMO=["anchor_age","female","n_codes_le_t0","n_adm_le_t0"]
base[DEMO]=base[DEMO].apply(pd.to_numeric,errors="coerce")
sub_eye=eye.groupby("subject_id").code.apply(set).to_dict()

# ---- neuro-ophthalmic sequela codes (same list as the manifestation ablation) ----
def is_seq(c):
    c=str(c).upper()
    return any(c.startswith(p) for p in ("3684","3682","3773","3771","3770","3795","36812","3623",
                                         "H534","H532","H46","H472","H471","H55","H5312","H34"))
SEQ={c for s in sub_eye.values() for c in s if is_seq(c)}
print(f"sequela codes in vocabulary: {len(SEQ)}",flush=True)

def pick(pat):
    return [p for p,d in D.items() if pd.notna(d) and pd.Series([str(d)]).str.contains(pat,case=False,regex=True).iloc[0]]
COMPOSITES={
 "Cerebrovascular disease (any)": pick(r"stroke|cerebral (artery )?occlusion|cerebral infarct|cerebrovascular|transient cerebral|cerebral ischemi|carotid"),
 "Dementia / cognitive decline (any)": pick(r"dementia|alzheimer|memory loss|cognitive"),
 "Peripheral neuropathy (any)": pick(r"neuropathy|polyneuropathy"),
 "Seizure disorder (any)": pick(r"epilep|convuls"),
 "Parkinsonism / movement (any)": pick(r"parkinson|movement disorder|tremor|dyskinesia"),
}
SINGLES={"Ischemic stroke (cerebral artery occlusion w/ infarction)":pick(r"cerebral artery occlusion, with cerebral inf"),
         "Polyneuropathy in diabetes":pick(r"polyneuropathy in diabetes"),
         "Dementias":pick(r"^dementias$"),
         "Late effects of cerebrovascular disease":pick(r"late effects of cerebrovascular"),
         "Transient cerebral ischemia":pick(r"transient cerebral ischemia")}
ALL={**{k:v for k,v in COMPOSITES.items()},**SINGLES}
og={k:dict(zip(v.subject_id,v.onset_days)) for k,v in onset.groupby("phecode")}
def onset_map(codes):
    m={}
    for p in codes:
        for s,v in og.get(p,{}).items():
            if v<m.get(s,np.inf): m[s]=v
    return m

cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
def oof(X,y):
    s=np.zeros(len(y))
    for tr,te in cv.split(X,y):
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med); Xi=np.where(np.isnan(X),med,X)
        m=HistGradientBoostingClassifier(max_iter=200,early_stopping=True,random_state=SEED)
        m.fit(Xi[tr],y[tr]); s[te]=m.predict_proba(Xi[te])[:,1]
    return s
def eyemat(subs,ablate):
    codes=sorted({c for s in subs for c in sub_eye.get(s,()) if not (ablate and c in SEQ)})
    idx={c:i for i,c in enumerate(codes)}
    M=np.zeros((len(subs),len(codes)),dtype=np.float32)
    for i,s in enumerate(subs):
        for c in sub_eye.get(s,()):
            if c in idx: M[i,idx[c]]=1
    return M[:,M.sum(0)>=15]

CELLS=[(90,365,"≥3mo·1yr"),(365,730,"≥1yr·2yr"),(365,1095,"≥1yr·3yr"),(730,1825,"≥2yr·5yr")]
rows=[]
for name,codes in ALL.items():
    o=onset_map(codes)
    if not o: continue
    for B,H,lab in CELLS:
        elig=base[base.follow_days>=H]
        subs=[s for s in elig.index if o.get(s,np.inf)>B]
        y=np.array([1 if B<o.get(s,np.inf)<=H else 0 for s in subs],float)
        nc=int(y.sum())
        if nc<MIN_CASES or len(y)-nc<MIN_CASES: continue
        bb=base.loc[subs]; Xd=bb[DEMO].to_numpy(float); Xc=bb[CBC].to_numpy(float)
        Xe=eyemat(subs,False); Xa=eyemat(subs,True)
        sc={"base":oof(Xd,y),"eye":oof(np.hstack([Xd,Xe]),y),"cbc":oof(np.hstack([Xd,Xc]),y),
            "both":oof(np.hstack([Xd,Xe,Xc]),y),"both_abl":oof(np.hstack([Xd,Xa,Xc]),y)}
        R=su.delong_analyze(y,sc); au=R["auc"]; ci=R["ci"]
        def gd(a,b):
            k=R["delta"].get(f"{a}-{b}")
            if k: return k["delta"],k["p_gt"]
            k=R["delta"].get(f"{b}-{a}")
            if k: return -k["delta"],1-k["p_gt"]
            return au[a]-au[b],np.nan
        g,pg=gd("both","base"); eo,peo=gd("both","cbc"); ga,pga=gd("both_abl","base")
        rows.append(dict(endpoint=name,cell=lab,n=len(subs),cases=nc,
            base=round(au["base"],3),eye=round(au["eye"],3),cbc=round(au["cbc"],3),
            both=round(au["both"],3),both_ci=f"[{ci['both'][0]:.3f},{ci['both'][1]:.3f}]",
            both_abl=round(au["both_abl"],3),
            gain=round(g,3),p_gain=round(pg,4),eye_over_cbc=round(eo,3),p_eye=round(peo,4),
            gain_ablated=round(ga,3),p_gain_abl=round(pga,4)))
        print(f"  {name[:40]:<42}{lab:<11} cases={nc:>4} base {au['base']:.3f} eye {au['eye']:.3f} cbc {au['cbc']:.3f} both {au['both']:.3f} (Δ{g:+.3f},p={pg:.3g}) | ablated both {au['both_abl']:.3f} (Δ{ga:+.3f},p={pga:.3g})",flush=True)

t=pd.DataFrame(rows)
if len(t):
    t["q"]=su.benjamini_hochberg(t.p_gain.fillna(1).to_numpy())
    t["q_abl"]=su.benjamini_hochberg(t.p_gain_abl.fillna(1).to_numpy())
    t.to_csv(f"{HERE}/neuro_axis.csv",index=False)
    print("\n"+"="*100); print("NEURO AXIS — FDR-significant (eye+CBC over baseline)"); print("="*100)
    s=t[(t.q<0.05)&(t.gain>0)].sort_values("gain",ascending=False)
    for _,r in s.iterrows():
        surv="SURVIVES ablation" if (r.q_abl<0.05 and r.gain_ablated>0) else "lost after ablation"
        print(f"  {r.endpoint[:44]:<46}{r.cell:<11} cases={int(r.cases):>4} base {r.base:.3f} -> both {r.both:.3f} {r.both_ci} (Δ{r.gain:+.3f}, q={r.q:.2g})  [{surv}]")
    print(f"\n  {len(s)}/{len(t)} endpoint-cells FDR-significant; of those, {int(((s.q_abl<0.05)&(s.gain_ablated>0)).sum())} survive sequela ablation")
    print("saved neuro_axis.csv")
else: print("no endpoint met the floor")
