"""Does CBC add ORTHOGONAL signal the eye can't see?  (MIMIC-IV)

Same eye-condition cohort + per-disease 1:1 matched design as the faithful
replication. For each disease compare AUROC of:  eye-only | CBC-only | eye+CBC.
Disease set spans the hypothesis space:
  - eye-specific (MS via optic neuritis)
  - CBC-specific (sepsis, heme malignancy)
  - both/complementary (diabetes, CKD, heart failure)
CBC = per-patient median of core hematology analytes (continuous).
"""
from config import CBCEYE_ROOT, MIMIC_HOSP
import warnings; warnings.filterwarnings("ignore")
import os, pandas as pd, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

M = MIMIC_HOSP
CBC_CACHE = f"{CBCEYE_ROOT}/cbc_features.parquet"

# ---------- eye cohort + eye features ----------
dx = pd.read_csv(f"{M}/diagnoses_icd.csv.gz", usecols=["subject_id","icd_code","icd_version"], dtype=str)
dx["code"] = dx.icd_code.str.strip()
def is_eye(c,v):
    if v=="9":  return c[:3].isdigit() and 360<=int(c[:3])<=379
    if v=="10": return len(c)>=3 and c[0]=="H" and c[1:3].isdigit() and int(c[1:3])<=59
    return False
dx["eye"]=[is_eye(c,v) for c,v in zip(dx.code,dx.icd_version)]
eye_cohort=set(dx[dx.eye].subject_id.unique())
sub_eye=dx[dx.eye & dx.subject_id.isin(eye_cohort)].groupby("subject_id").code.apply(set)
sub_all=dx[dx.subject_id.isin(eye_cohort)].groupby("subject_id").apply(
    lambda g:list(zip(g.code,g.icd_version)), include_groups=False)
print(f"eye cohort: {len(eye_cohort):,}", flush=True)

# ---------- CBC features (cache) ----------
CBC_RAW=f"{CBCEYE_ROOT}/cbc_raw.parquet"
dl = pd.read_csv(f"{M}/d_labitems.csv.gz", dtype=str)
terms=["white blood cell","red blood cell","hemoglobin","hematocrit","mcv","mch","mchc",
       "rdw","platelet","neutrophil","lymphocyte","monocyte","eosinophil","basophil"]
cand=dl[(dl.category.str.lower()=="hematology") & (dl.fluid.str.lower()=="blood")
        & dl.label.str.lower().str.contains("|".join(terms), na=False)]
itemids=set(cand.itemid); lbl=dict(zip(cand.itemid,cand.label))
if os.path.exists(CBC_RAW):
    med = pd.read_parquet(CBC_RAW)
else:
    print(f"scanning labevents for {len(itemids)} CBC itemids...", flush=True)
    parts=[]
    for ch in pd.read_csv(f"{M}/labevents.csv.gz", usecols=["subject_id","itemid","valuenum"],
                          dtype={"subject_id":str,"itemid":str}, chunksize=3_000_000):
        ch=ch[ch.itemid.isin(itemids) & ch.subject_id.isin(eye_cohort) & ch.valuenum.notna()]
        if len(ch): parts.append(ch)
    le=pd.concat(parts, ignore_index=True)
    med=le.groupby(["subject_id","itemid"]).valuenum.median().unstack()  # columns = itemid
    med.to_parquet(CBC_RAW)
# coverage filter (per itemid), then rename to labels keeping highest-coverage per duplicate label
cov=med.notna().mean()
keepids=sorted([c for c in med.columns if cov[c]>=0.40], key=lambda c:-cov[c])
med=med[keepids]; med.columns=[lbl.get(c,c) for c in keepids]
cbc=med.loc[:, ~pd.Index(med.columns).duplicated()]   # keep first (=highest coverage) per analyte
print(f"CBC features: {cbc.shape[1]} analytes on {cbc.shape[0]:,} patients", flush=True)
print("  analytes:", list(cbc.columns), flush=True)
cbc_med=cbc.median()

# ---------- meta for matching ----------
pt=pd.read_csv(f"{M}/patients.csv.gz", usecols=["subject_id","gender","anchor_age"], dtype=str)
pt["anchor_age"]=pd.to_numeric(pt.anchor_age,errors="coerce")
adm=pd.read_csv(f"{M}/admissions.csv.gz", usecols=["subject_id","race"], dtype=str)
race=adm.groupby("subject_id").race.agg(lambda s:s.mode().iat[0] if len(s.mode()) else "UNK")
meta=pt.set_index("subject_id"); meta=meta[meta.index.isin(eye_cohort)].copy()
meta["race"]=race.reindex(meta.index).fillna("UNK")
meta["ageband"]=(meta.anchor_age//5).astype("Int64"); meta=meta.dropna(subset=["ageband"])

DISEASES={
 "Type 2 diabetes":   dict(d9=["250"], d10=["E11"], hyp="both"),
 "Chronic kidney disease": dict(d9=["585"], d10=["N18"], hyp="both"),
 "Heart failure":     dict(d9=["428"], d10=["I50"], hyp="both"),
 "Multiple sclerosis":dict(d9=["340"], d10=["G35"], hyp="eye"),
 "Autonomic/idiopathic neuropathy": dict(d9=["3370","3569"], d10=["G900","G9009","G609"], hyp="eye"),
 "Sepsis":            dict(d9=["99591","99592","0380","0381","0382","0383","0384","0385","0386","0388","0389"], d10=["A40","A41","R652"], hyp="cbc"),
 "Heme malignancy (leuk/lymphoma)": dict(d9=["200","201","202","203","204","205","206","207","208"], d10=["C81","C82","C83","C84","C85","C86","C88","C90","C91","C92","C93","C94","C95","C96"], hyp="cbc"),
 "Rheumatoid arthritis": dict(d9=["714"], d10=["M05","M06"], hyp="cbc"),
 "Cirrhosis":         dict(d9=["5712","5715","5716"], d10=["K703","K74"], hyp="cbc"),
}
def has(codes,d9,d10):
    for c,v in codes:
        if v=="9" and any(c.startswith(p) for p in d9): return True
        if v=="10" and any(c.startswith(p) for p in d10): return True
    return False

rng=np.random.default_rng(42); cv=StratifiedKFold(5,shuffle=True,random_state=42)
def auc(X,y): return cross_val_score(HistGradientBoostingClassifier(max_iter=200,random_state=42),X,y,cv=cv,scoring="roc_auc").mean()

cbc_idx={s:i for i,s in enumerate(cbc.index)}
cbc_arr=cbc.fillna(cbc_med).values
results=[]
print(f"\n{'disease':<34}{'hyp':>5}{'cases':>7}{'eye':>7}{'CBC':>7}{'both':>7}  verdict", flush=True)
for name,d in DISEASES.items():
    cc={s:sub_all[s] for s in meta.index if s in sub_all.index}
    cases=[s for s in meta.index if s in cc and has(cc[s],d["d9"],d["d10"])]
    if len(cases)<100:
        print(f"{name:<34}{d['hyp']:>5}{len(cases):>7}   (skip)", flush=True); continue
    cs=set(cases); pool=defaultdict(list)
    for s in meta.index:
        if s not in cs: pool[(meta.loc[s,"ageband"],meta.loc[s,"gender"],meta.loc[s,"race"])].append(s)
    for k in pool: rng.shuffle(pool[k])
    ctrl=[]
    for s in cases:
        k=(meta.loc[s,"ageband"],meta.loc[s,"gender"],meta.loc[s,"race"])
        if pool[k]: ctrl.append(pool[k].pop())
    cases=cases[:len(ctrl)]; subs=cases+ctrl; y=np.array([1]*len(cases)+[0]*len(ctrl))
    # eye matrix
    cnt=defaultdict(int)
    for s in subs:
        for c in sub_eye.get(s,set()): cnt[c]+=1
    vocab=[c for c,n in cnt.items() if n>=20]; vidx={c:i for i,c in enumerate(vocab)}
    Xeye=np.zeros((len(subs),len(vocab)))
    for i,s in enumerate(subs):
        for c in sub_eye.get(s,set()):
            if c in vidx: Xeye[i,vidx[c]]=1
    # cbc matrix
    Xcbc=np.vstack([cbc_arr[cbc_idx[s]] if s in cbc_idx else cbc_med.values for s in subs])
    a_e=auc(Xeye,y); a_c=auc(Xcbc,y); a_b=auc(np.hstack([Xeye,Xcbc]),y)
    best=max(a_e,a_c)
    if a_b>best+0.015: v="SYNERGY (both add)"
    elif a_e>a_c+0.03: v="eye-driven"
    elif a_c>a_e+0.03: v="CBC-driven"
    else: v="comparable"
    results.append((name,d["hyp"],len(cases),a_e,a_c,a_b,v))
    print(f"{name:<34}{d['hyp']:>5}{len(cases):>7}{a_e:>7.3f}{a_c:>7.3f}{a_b:>7.3f}  {v}", flush=True)

pd.DataFrame(results,columns=["disease","hypothesis","n_cases","auc_eye","auc_cbc","auc_both","verdict"]).to_csv(
    f"{CBCEYE_ROOT}/orthogonality_results.csv",index=False)
print("\nsaved orthogonality_results.csv", flush=True)
