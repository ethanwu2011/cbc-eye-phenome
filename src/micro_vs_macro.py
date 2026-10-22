"""THE CENTRAL TEST: is the retina a MICROvascular sentinel?

Hypothesis, stated so it can fail: ophthalmic data at a single encounter forecast future
MICROvascular end-organ damage (kidney, peripheral nerve, autonomic nerve, skin/limb
microcirculation) but NOT MACROvascular events (stroke, MI, large-vessel disease).

That is a falsifiable, mechanistically motivated prediction — the retina is capillary-bed
tissue — and it is tested here with matched designs on both sides, so the contrast is not
an artifact of one side being better powered.

Time-to-event with censoring, blanking B = 1 yr (predictors at T0 only), events accruing over
all remaining follow-up. Reports Harrell C for base / eye / CBC / both, plus the eye's margin
over the CBC, plus a sequela-ablated fit.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold

HERE=CBCEYE_ROOT; SEED=42; B=365
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
def is_seq(c):
    c=str(c).upper()
    return any(c.startswith(p) for p in ("3684","3682","3773","3771","3770","3795","36812","3623",
                                         "H534","H532","H46","H472","H471","H55","H5312","H34"))
SEQ={c for s in sub_eye.values() for c in s if is_seq(c)}
def pick(pat):
    return [p for p,d in D.items() if pd.notna(d) and pd.Series([str(d)]).str.contains(pat,case=False,regex=True).iloc[0]]
def omap(codes):
    m={}
    for p in codes:
        for s,v in og.get(p,{}).items():
            if v<m.get(s,np.inf): m[s]=v
    return m

MICRO={
 "Nephropathy (renal microvascular)": pick(r"nephritis and nephropathy in diseases class"),
 "Diabetic polyneuropathy":           pick(r"polyneuropathy in diabetes"),
 "Peripheral neuropathy (any)":       pick(r"neuropathy|polyneuropathy"),
 "Chronic kidney disease":            pick(r"chronic kidney disease|chronic renal failure"),
 "End stage renal disease":           pick(r"end stage renal|esrd|renal dialysis"),
 "Chronic ulcer of skin / leg / foot": pick(r"chronic ulcer"),
 "Peripheral angiopathy (microvasc.)": pick(r"peripheral angiopathy"),
 "Retinal vascular occlusion":        pick(r"retinal vascular occlusion"),
 "Diabetic ophthalmic complication":  pick(r"diabetic retinopathy|ophthalmic manifestations"),
}
MACRO={
 "Cerebrovascular disease (any)":     pick(r"stroke|cerebral (artery )?occlusion|cerebral infarct|cerebrovascular|transient cerebral|cerebral ischemi"),
 "Myocardial infarction":             pick(r"^myocardial infarction"),
 "Coronary atherosclerosis":          pick(r"coronary atherosclerosis"),
 "Atherosclerosis of extremities":    pick(r"atherosclerosis of the extremities"),
 "Congestive heart failure":          pick(r"^congestive heart failure"),
 "Atrial fibrillation":               pick(r"^atrial fibrillation$"),
 "Aortic aneurysm":                   pick(r"aortic aneurysm"),
}
def surv(e,t): return np.array([(bool(a),float(b)) for a,b in zip(e,t)],dtype=[("event",bool),("time",float)])
def eyemat(subs,ablate=False):
    codes=sorted({c for s in subs for c in sub_eye.get(s,()) if not (ablate and c in SEQ)})
    idx={c:i for i,c in enumerate(codes)}
    M=np.zeros((len(subs),len(codes)),dtype=np.float32)
    for i,s in enumerate(subs):
        for c in sub_eye.get(s,()):
            if c in idx: M[i,idx[c]]=1
    return M[:,M.sum(0)>=20]
def cidx(X,Y):
    X=np.nan_to_num(np.where(np.isnan(X),np.nanmedian(X,axis=0),X))
    r=np.zeros(len(Y))
    for tr,te in KFold(5,shuffle=True,random_state=SEED).split(X):
        m=GradientBoostingSurvivalAnalysis(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=SEED)
        m.fit(X[tr],Y[tr]); r[te]=m.predict(X[te])
    return concordance_index_censored(Y["event"],Y["time"],r)[0]

rows=[]
for axis,PANEL in [("MICROvascular",MICRO),("MACROvascular",MACRO)]:
    print(f"\n{'='*116}\n{axis}  (blank {B} d, time-to-event, predictors at T0)\n{'='*116}",flush=True)
    for name,codes in PANEL.items():
        o=omap(codes); elig=base[base.follow_days>B]
        subs=[s for s in elig.index if o.get(s,np.inf)>B]
        if len(subs)<400: continue
        ev=np.array([1 if o.get(s,np.inf)!=np.inf else 0 for s in subs])
        tm=np.array([(o[s]-B) if o.get(s,np.inf)!=np.inf else (elig.loc[s,"follow_days"]-B) for s in subs],float)
        k=tm>0; subs=[s for s,x in zip(subs,k) if x]; ev,tm=ev[k],tm[k]
        if ev.sum()<40:
            print(f"  {name[:42]:<44} only {int(ev.sum())} events — skipped",flush=True); continue
        bb=base.loc[subs]; Y=surv(ev,tm); Xd=bb[DEMO].to_numpy(float); Xc=bb[CBC].to_numpy(float)
        Xe=eyemat(subs); Xa=eyemat(subs,True)
        cb=cidx(Xd,Y); ce=cidx(np.hstack([Xd,Xe]),Y); cc=cidx(np.hstack([Xd,Xc]),Y)
        cbo=cidx(np.hstack([Xd,Xe,Xc]),Y); ca=cidx(np.hstack([Xd,Xa,Xc]),Y)
        print(f"  {name[:42]:<44} n={len(subs):>5} ev={int(ev.sum()):>4} | base {cb:.3f} eye {ce:.3f} cbc {cc:.3f} both {cbo:.3f} (Δ{cbo-cb:+.3f}) eye-cbc {ce-cc:+.3f} | ablated {ca:.3f}",flush=True)
        rows.append(dict(axis=axis,endpoint=name,n=len(subs),events=int(ev.sum()),
            c_base=round(cb,3),c_eye=round(ce,3),c_cbc=round(cc,3),c_both=round(cbo,3),
            c_gain=round(cbo-cb,3),eye_minus_cbc=round(ce-cc,3),eye_gain=round(ce-cb,3),c_ablated=round(ca,3)))
t=pd.DataFrame(rows); t.to_csv(f"{HERE}/micro_vs_macro.csv",index=False)
print(f"\n{'='*116}\nVERDICT — does the retina behave as a MICROvascular sentinel?\n{'='*116}")
for axis,g in t.groupby("axis",sort=False):
    print(f"  {axis:<16} endpoints {len(g):>2} | median events {g.events.median():>5.0f} | median eye gain over base {g.eye_gain.median():+.3f} | median eye-minus-CBC {g.eye_minus_cbc.median():+.3f}")
mi,ma=t[t.axis=="MICROvascular"],t[t.axis=="MACROvascular"]
print(f"\n  eye beats CBC:  MICRO {int((mi.eye_minus_cbc>0.02).sum())}/{len(mi)}   MACRO {int((ma.eye_minus_cbc>0.02).sum())}/{len(ma)}")
print(f"  eye gain >0.05 over baseline:  MICRO {int((mi.eye_gain>0.05).sum())}/{len(mi)}   MACRO {int((ma.eye_gain>0.05).sum())}/{len(ma)}")
print("\nsaved micro_vs_macro.csv")
