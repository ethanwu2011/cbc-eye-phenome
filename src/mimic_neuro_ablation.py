"""Reverse-causation / manifestation safeguard for the neuro headline (reviewer's #1 ask).
For ischemic stroke and MS in MIMIC, re-estimate the eye-only and eye-over-CBC AUROC after
ABLATING downstream neuro-ophthalmic SEQUELA codes (visual-field defects/homonymous hemianopia,
diplopia, optic neuritis/atrophy, papilledema, nystagmus, transient visual loss, retinal-artery
occlusion). If the eye signal survives ablation, it is not merely post-event ocular manifestation.
Runs in full cohort and in non-diabetics; DeLong AUC CIs + paired tests. Reuses landscape prep."""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import stats_utils as su
exec(open(f"{CBCEYE_ROOT}/mimic_landscape.py").read().split("rng=np.random.default_rng(42)")[0])
mb=meta[["ageband","gender","race"]]
diabetics={s for s in members if s in sub_all and has(sub_all[s],["250"],["E08","E09","E10","E11","E13"])}
nondiab=[s for s in members if s not in diabetics]

def is_sequela(c,v):
    c=str(c)
    if v=="9":  return any(c.startswith(p) for p in ("3684","3682","3773","3771","3770","3795","36812","3623"))
    if v=="10": return any(c.startswith(p) for p in ("H534","H532","H46","H472","H471","H55","H5312","H34"))
    return False
# eye-code -> is it a sequela? (eye codes are dotless; need version — store per-code version from sub_all)
codever={}
for s,lst in sub_all.items():
    for c,v in lst: codever.setdefault(c,v)
SEQ={c for c,v in codever.items() if is_sequela(c,v)}
print(f"sequela eye codes flagged: {len(SEQ)}",flush=True)

cv=StratifiedKFold(5,shuffle=True,random_state=42)
def oof(X,y):
    s=np.zeros(len(y))
    for tr,te in cv.split(X,y):
        c=HistGradientBoostingClassifier(max_iter=150,random_state=42); c.fit(X[tr],y[tr]); s[te]=c.predict_proba(X[te])[:,1]
    return s

def build(disease_codes,pool_members,seed=42,ablate=False):
    d9,d10=disease_codes
    rng=np.random.default_rng(seed)
    cases=[s for s in pool_members if s in sub_all and has(sub_all[s],d9,d10)]
    if len(cases)<80: return None
    cs=set(cases); pool=defaultdict(list)
    for s in pool_members:
        if s not in cs: pool[(mb.at[s,"ageband"],mb.at[s,"gender"],mb.at[s,"race"])].append(s)
    for k in pool: rng.shuffle(pool[k])
    ctrl=[]
    for s in cases:
        k=(mb.at[s,"ageband"],mb.at[s,"gender"],mb.at[s,"race"])
        if pool[k]: ctrl.append(pool[k].pop())
    cases=cases[:len(ctrl)]; subs=cases+ctrl; y=np.array([1]*len(cases)+[0]*len(ctrl))
    cnt=defaultdict(int)
    for s in subs:
        for c in sub_eye.get(s,()):
            if ablate and c in SEQ: continue
            cnt[c]+=1
    vocab=[c for c,n in cnt.items() if n>=20]; vidx={c:i for i,c in enumerate(vocab)}
    Xe=np.zeros((len(subs),len(vocab)))
    for i,s in enumerate(subs):
        for c in sub_eye.get(s,()):
            if c in vidx: Xe[i,vidx[c]]=1
    Xc=np.vstack([cbc_arr[cbc_idx[s]] if s in cbc_idx else cbc_med.values for s in subs])
    return dict(y=y,Xe=Xe,Xc=Xc,nvocab=len(vocab),ncase=len(cases))

DIS={"Ischemic stroke":(["434","436"],["I63"]),"Multiple sclerosis":(["340"],["G35"])}
rows=[]
for name,codes in DIS.items():
    for coh,pool in [("full",members),("nondiab",nondiab)]:
        full=build(codes,pool,ablate=False); abl=build(codes,pool,ablate=True)
        if full is None or abl is None: continue
        y=full["y"]
        sc={"eye_full":oof(full["Xe"],y),"eye_abl":oof(abl["Xe"],y),"cbc":oof(full["Xc"],y),
            "both_full":oof(np.hstack([full["Xe"],full["Xc"]]),y),
            "both_abl":oof(np.hstack([abl["Xe"],full["Xc"]]),y)}
        D=su.delong_analyze(y,sc)
        au=D["auc"]; ci=D["ci"]
        def gd(a,b):  # signed delta (a-b), p(a>b), se — order-agnostic
            d=D["delta"].get(f"{a}-{b}")
            if d: return d["delta"],d["p_gt"],d["se"]
            d=D["delta"].get(f"{b}-{a}")
            if d: return -d["delta"],1-d["p_gt"],d["se"]
            return au[a]-au[b],np.nan,np.nan
        addf_d,addf_p,addf_se=gd("both_full","cbc")     # eye-adds (full vocab)
        adda_d,adda_p,adda_se=gd("both_abl","cbc")      # eye-adds (sequela ablated)
        abl_d,abl_p,abl_se=gd("eye_full","eye_abl")     # eye-only drop from ablation
        rows.append(dict(disease=name,cohort=coh,ncase=full["ncase"],
            eye_full=round(au["eye_full"],3),eye_full_ci=f"[{ci['eye_full'][0]:.3f},{ci['eye_full'][1]:.3f}]",
            eye_abl=round(au["eye_abl"],3),eye_abl_ci=f"[{ci['eye_abl'][0]:.3f},{ci['eye_abl'][1]:.3f}]",
            eye_drop_from_ablation=round(abl_d,3),eye_drop_p=round(abl_p,4),
            cbc=round(au["cbc"],3),
            eyeadds_full=round(addf_d,3),eyeadds_full_p=round(addf_p,4),eyeadds_full_se=round(addf_se,4),
            eyeadds_abl=round(adda_d,3),eyeadds_abl_p=round(adda_p,4),eyeadds_abl_se=round(adda_se,4)))
        print(f"{name:<18}{coh:<9} eyeOnly {au['eye_full']:.3f}->abl {au['eye_abl']:.3f} (drop {abl_d:+.3f},p{abl_p:.3f}) | eye-adds full {addf_d:+.3f}(p{addf_p:.3f}) abl {adda_d:+.3f}(p{adda_p:.3f})",flush=True)
r=pd.DataFrame(rows); r.to_csv(f"{CBCEYE_ROOT}/mimic_neuro_ablation.csv",index=False)
print("\nsaved mimic_neuro_ablation.csv",flush=True)
