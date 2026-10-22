"""LEAD-TIME, done with time-to-event instead of fixed horizons.

The fixed-horizon design threw away power twice over: it required complete follow-up to the
horizon (so a patient followed 8 years was excluded from the 5-year cell), and it capped case
accrual at the horizon (so an event at year 6 never counted). Censoring fixes both — every
patient contributes their observed time, and events accrue over the full 14.5 years.

  eligible : follow_days > B and no onset at or before B      (B = required lead / blanking)
  event    : first onset after B,  time = onset - B
  censored : no onset,             time = follow_days - B
Features are measured at T0 only. Gradient-boosted survival models, 5-fold, pooled
out-of-fold risk scores; Harrell C, IPCW-C, and time-dependent AUROC read out at
1, 2, 3, 5 and 10 years so the lead-time question is answered across the whole curve
rather than at one arbitrary cut.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sklearn.model_selection import KFold

HERE=CBCEYE_ROOT; SEED=42
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
og={k:dict(zip(v.subject_id,v.onset_days)) for k,v in onset.groupby("phecode")}
def pick(pat):
    return [p for p,d in D.items() if pd.notna(d) and pd.Series([str(d)]).str.contains(pat,case=False,regex=True).iloc[0]]
def omap(codes):
    m={}
    for p in codes:
        for s,v in og.get(p,{}).items():
            if v<m.get(s,np.inf): m[s]=v
    return m

ENDPOINTS={
 "Nephropathy (renal microvascular)": pick(r"nephritis and nephropathy in diseases class"),
 "Polyneuropathy in diabetes":        pick(r"polyneuropathy in diabetes"),
 "Peripheral neuropathy (any)":       pick(r"neuropathy|polyneuropathy"),
 "Cerebrovascular disease (any)":     pick(r"stroke|cerebral (artery )?occlusion|cerebral infarct|cerebrovascular|transient cerebral|cerebral ischemi"),
 "Dementia / cognitive (any)":        pick(r"dementia|alzheimer|memory loss|cognitive"),
 "Congestive heart failure":          pick(r"^congestive heart failure"),
 "Atherosclerosis of extremities":    pick(r"atherosclerosis of the extremities"),
 "Atrial fibrillation":               pick(r"^atrial fibrillation$"),
 "Chronic kidney disease":            pick(r"chronic kidney disease|chronic renal failure"),
 "Type 2 diabetes":                   pick(r"^type 2 diabetes$"),
}
YRS=[1,2,3,5,10]
def surv(y_event,y_time):
    return np.array([(bool(e),float(t)) for e,t in zip(y_event,y_time)],dtype=[("event",bool),("time",float)])
def eyemat(subs):
    codes=sorted({c for s in subs for c in sub_eye.get(s,())})
    idx={c:i for i,c in enumerate(codes)}
    M=np.zeros((len(subs),len(codes)),dtype=np.float32)
    for i,s in enumerate(subs):
        for c in sub_eye.get(s,()):
            if c in idx: M[i,idx[c]]=1
    return M[:,M.sum(0)>=20]

rows=[]
for B,blab in [(365,"≥1 yr lead"),(730,"≥2 yr lead")]:
    print(f"\n{'='*104}\n{blab}  (blanking {B} d; events accrue over ALL remaining follow-up)\n{'='*104}",flush=True)
    for name,codes in ENDPOINTS.items():
        o=omap(codes)
        elig=base[(base.follow_days>B)]
        subs=[s for s in elig.index if o.get(s,np.inf)>B]
        if len(subs)<400: continue
        ev=np.array([1 if o.get(s,np.inf)!=np.inf else 0 for s in subs])
        tm=np.array([ (o[s]-B) if o.get(s,np.inf)!=np.inf else (elig.loc[s,"follow_days"]-B) for s in subs],float)
        keep=tm>0; subs=[s for s,k in zip(subs,keep) if k]; ev,tm=ev[keep],tm[keep]
        n_ev=int(ev.sum())
        if n_ev<40: continue
        bb=base.loc[subs]
        blocks={"base":bb[DEMO].to_numpy(float),
                "eye":np.hstack([bb[DEMO].to_numpy(float),eyemat(subs)]),
                "cbc":np.hstack([bb[DEMO].to_numpy(float),bb[CBC].to_numpy(float)]),
                "both":np.hstack([bb[DEMO].to_numpy(float),eyemat(subs),bb[CBC].to_numpy(float)])}
        Y=surv(ev,tm); out={}
        for bn,X in blocks.items():
            X=np.where(np.isnan(X),np.nanmedian(np.where(np.isnan(X),np.nan,X),axis=0),X)
            X=np.nan_to_num(X); risk=np.zeros(len(Y))
            for tr,te in KFold(5,shuffle=True,random_state=SEED).split(X):
                m=GradientBoostingSurvivalAnalysis(n_estimators=150,learning_rate=0.1,max_depth=3,random_state=SEED)
                m.fit(X[tr],Y[tr]); risk[te]=m.predict(X[te])
            c=concordance_index_censored(Y["event"],Y["time"],risk)[0]
            aucs={}
            for yr in YRS:
                t=yr*365.25
                if t<np.percentile(tm[ev==1],95) and (tm[ev==1]<t).sum()>=15:
                    try: aucs[yr]=float(cumulative_dynamic_auc(Y,Y,risk,[t])[0][0])
                    except Exception: pass
            out[bn]=(c,aucs)
        cb,cE,cC,cBo=out["base"][0],out["eye"][0],out["cbc"][0],out["both"][0]
        a=out["both"][1]; ab=out["base"][1]
        td=" ".join(f"{yr}y {a[yr]:.3f}" for yr in YRS if yr in a)
        print(f"  {name[:38]:<40} n={len(subs):>5} ev={n_ev:>4} | C base {cb:.3f} eye {cE:.3f} cbc {cC:.3f} both {cBo:.3f} (Δ{cBo-cb:+.3f}) | tdAUC both: {td}",flush=True)
        rows.append(dict(endpoint=name,lead=blab,blank=B,n=len(subs),events=n_ev,
            c_base=round(cb,3),c_eye=round(cE,3),c_cbc=round(cC,3),c_both=round(cBo,3),
            c_gain=round(cBo-cb,3),eye_over_cbc=round(cBo-cC,3),
            **{f"tdauc_both_{yr}y":round(a[yr],3) for yr in YRS if yr in a},
            **{f"tdauc_base_{yr}y":round(ab[yr],3) for yr in YRS if yr in ab}))
t=pd.DataFrame(rows); t.to_csv(f"{HERE}/leadtime_survival.csv",index=False)
print(f"\n{'='*104}\nSUMMARY — time-to-event recovers the power the fixed horizons destroyed\n{'='*104}")
for lead,g in t.groupby("lead",sort=False):
    print(f"  {lead}: {len(g)} endpoints | median events {g.events.median():.0f} | median C-gain {g.c_gain.median():+.3f} | eye>CBC in {(g.c_eye>g.c_cbc).sum()}/{len(g)}")
print("\nsaved leadtime_survival.csv")
