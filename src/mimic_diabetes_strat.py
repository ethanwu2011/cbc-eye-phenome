"""MIMIC diabetes-independence test. Mirrors mimic_landscape.py's 1:1 matched case-control
design, but recomputes eye/CBC/both AUROC (a) in the FULL eye cohort and (b) in NON-DIABETICS
only (ICD 250 / E08-E13 excluded from BOTH cases and the matched control pool). Question:
does the eye still discriminate neuro disease in non-diabetics, and does its apparent
cardiometabolic signal collapse once diabetic-retinopathy patients are removed (as in NHANES)?
Reuses the landscape prep WITHOUT running its disease loop."""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
exec(open(f"{CBCEYE_ROOT}/mimic_landscape.py").read().split("rng=np.random.default_rng(42)")[0])
# now available: members, meta, sub_eye, sub_all, cbc_arr, cbc_idx, cbc_med, D, has, MIN_CASES
cv=StratifiedKFold(5,shuffle=True,random_state=42)
def auc(X,y): return cross_val_score(HistGradientBoostingClassifier(max_iter=150,random_state=42),X,y,cv=cv,scoring="roc_auc").mean()
mb=meta[["ageband","gender","race"]]
MIN_ND=80

diabetics={s for s in members if s in sub_all and has(sub_all[s],["250"],["E08","E09","E10","E11","E13"])}
nondiab=[s for s in members if s not in diabetics]
print(f"diabetics in eye cohort: {len(diabetics):,} / {len(members):,}  | non-diabetic {len(nondiab):,}",flush=True)

def run(d9,d10,pool_members,seed=42):
    rng=np.random.default_rng(seed)
    cases=[s for s in pool_members if s in sub_all and has(sub_all[s],d9,d10)]
    if len(cases)<MIN_ND: return None
    cs=set(cases); pool=defaultdict(list)
    for s in pool_members:
        if s not in cs: pool[(mb.at[s,"ageband"],mb.at[s,"gender"],mb.at[s,"race"])].append(s)
    for k in pool: rng.shuffle(pool[k])
    ctrl=[]
    for s in cases:
        k=(mb.at[s,"ageband"],mb.at[s,"gender"],mb.at[s,"race"])
        if pool[k]: ctrl.append(pool[k].pop())
    cases=cases[:len(ctrl)]; subs=cases+ctrl; y=np.array([1]*len(cases)+[0]*len(ctrl))
    if len(cases)<MIN_ND: return None
    cnt=defaultdict(int)
    for s in subs:
        for c in sub_eye.get(s,()): cnt[c]+=1
    vocab=[c for c,n in cnt.items() if n>=20]; vidx={c:i for i,c in enumerate(vocab)}
    Xe=np.zeros((len(subs),len(vocab)))
    for i,s in enumerate(subs):
        for c in sub_eye.get(s,()):
            if c in vidx: Xe[i,vidx[c]]=1
    Xc=np.vstack([cbc_arr[cbc_idx[s]] if s in cbc_idx else cbc_med.values for s in subs])
    a_e=auc(Xe,y); a_c=auc(Xc,y); a_b=auc(np.hstack([Xe,Xc]),y)
    return dict(n=len(cases),eye=round(a_e,3),cbc=round(a_c,3),both=round(a_b,3),eye_adds=round(a_b-a_c,3))

PANEL=[k for k in ["Ischemic stroke","TIA","Multiple sclerosis","Dementia/Alzheimer","Parkinson disease",
 "Peripheral neuropathy","Epilepsy","Heart failure","Ischemic heart disease","Essential hypertension",
 "Chronic kidney disease","Atrial fibrillation","Peripheral arterial disease","Anemia","COPD",
 "Major depression","Cirrhosis","Sepsis"] if k in D]
NEURO={"Ischemic stroke","TIA","Multiple sclerosis","Dementia/Alzheimer","Parkinson disease","Peripheral neuropathy","Epilepsy","Major depression"}
rows=[]
print(f"\n{'disease':<26}{'sys':>6} | {'n_full':>6}{'eyeF':>6}{'addF':>6} | {'n_nd':>6}{'eyeND':>7}{'addND':>7}",flush=True)
for name in PANEL:
    sys,d9,d10=D[name]
    full=run(d9,d10,members); nd=run(d9,d10,nondiab)
    if full is None or nd is None:
        print(f"{name:<26} skip (full={None if full is None else full['n']}, nd={None if nd is None else nd['n']})",flush=True); continue
    rows.append(dict(disease=name,system=sys,axis=("neuro" if name in NEURO else "cardiometabolic/other"),
        n_full=full["n"],eye_full=full["eye"],cbc_full=full["cbc"],both_full=full["both"],eyeadds_full=full["eye_adds"],
        n_nd=nd["n"],eye_nd=nd["eye"],cbc_nd=nd["cbc"],both_nd=nd["both"],eyeadds_nd=nd["eye_adds"],
        eye_delta=round(nd["eye"]-full["eye"],3),eyeadds_delta=round(nd["eye_adds"]-full["eye_adds"],3)))
    print(f"{name:<26}{sys:>6} | {full['n']:>6}{full['eye']:>6.3f}{full['eye_adds']:>6.3f} | {nd['n']:>6}{nd['eye']:>7.3f}{nd['eye_adds']:>7.3f}",flush=True)
    pd.DataFrame(rows).to_csv(f"{CBCEYE_ROOT}/mimic_diabetes_strat.csv",index=False)

r=pd.DataFrame(rows); r.to_csv(f"{CBCEYE_ROOT}/mimic_diabetes_strat.csv",index=False)
print("\n=== DIABETES INDEPENDENCE (MIMIC, matched case-control) ===")
neu=r[r.axis=="neuro"]; car=r[r.axis!="neuro"]
print(f"NEURO ({len(neu)}): eye still discriminates in non-diabetics? median eye_nd={neu.eye_nd.median():.3f} (full {neu.eye_full.median():.3f}); eye still adds beyond CBC in non-diab for {(neu.eyeadds_nd>0.01).sum()}/{len(neu)}")
print(f"CARDIOMETAB/OTHER ({len(car)}): median eye_nd={car.eye_nd.median():.3f} (full {car.eye_full.median():.3f}); eye still adds beyond CBC in non-diab for {(car.eyeadds_nd>0.01).sum()}/{len(car)}")
print(f"\nOVERALL eye signal change non-diab vs full: median Δeye={r.eye_delta.median():+.3f}; median Δ(eye-adds)={r.eyeadds_delta.median():+.3f}")
print("saved mimic_diabetes_strat.csv",flush=True)
