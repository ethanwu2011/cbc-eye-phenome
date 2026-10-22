"""Full eye+CBC orthogonality LANDSCAPE across a curated cross-system disease panel.
MIMIC-IV eye-condition cohort; per disease 1:1 matched (age/sex/race);
AUROC for eye-only / CBC-only / eye+CBC (HistGBDT, 5-fold).
CBC = MOST RECENT value per analyte (switchable via AGG)."""
from config import CBCEYE_ROOT, MIMIC_HOSP
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

M=MIMIC_HOSP; AGG="last"  # last|first|median
MIN_CASES=150

dx=pd.read_csv(f"{M}/diagnoses_icd.csv.gz",usecols=["subject_id","icd_code","icd_version"],dtype=str)
dx["code"]=dx.icd_code.str.strip()
def is_eye(c,v):
    if v=="9":  return c[:3].isdigit() and 360<=int(c[:3])<=379
    if v=="10": return len(c)>=3 and c[0]=="H" and c[1:3].isdigit() and int(c[1:3])<=59
    return False
dx["eye"]=[is_eye(c,v) for c,v in zip(dx.code,dx.icd_version)]
eye_cohort=set(dx[dx.eye].subject_id.unique())
sub_eye=dx[dx.eye & dx.subject_id.isin(eye_cohort)].groupby("subject_id").code.apply(set).to_dict()
sub_all=dx[dx.subject_id.isin(eye_cohort)].groupby("subject_id").apply(
    lambda g:list(zip(g.code,g.icd_version)),include_groups=False).to_dict()
print(f"eye cohort {len(eye_cohort):,}",flush=True)

# CBC: most-recent per analyte
le=pd.read_parquet(f"{CBCEYE_ROOT}/cbc_long.parquet")
le["charttime"]=pd.to_datetime(le.charttime); le=le.sort_values("charttime")
agg={"last":"last","first":"first","median":"median"}[AGG]
piv=getattr(le.groupby(["subject_id","itemid"]).valuenum,agg)().unstack()
dl=pd.read_csv(f"{M}/d_labitems.csv.gz",dtype=str); lbl=dict(zip(dl.itemid,dl.label))
cov=piv.notna().mean(); keep=sorted([c for c in piv.columns if cov[c]>=0.40],key=lambda c:-cov[c])
piv=piv[keep]; piv.columns=[lbl.get(c,c) for c in keep]
cbc=piv.loc[:,~pd.Index(piv.columns).duplicated()]
cbc_med=cbc.median(); cbc_idx={s:i for i,s in enumerate(cbc.index)}; cbc_arr=cbc.fillna(cbc_med).values
print(f"CBC {cbc.shape[1]} analytes ({AGG}) on {cbc.shape[0]:,} pts",flush=True)

pt=pd.read_csv(f"{M}/patients.csv.gz",usecols=["subject_id","gender","anchor_age"],dtype=str)
pt["anchor_age"]=pd.to_numeric(pt.anchor_age,errors="coerce")
adm=pd.read_csv(f"{M}/admissions.csv.gz",usecols=["subject_id","race"],dtype=str)
race=adm.groupby("subject_id").race.agg(lambda s:s.mode().iat[0] if len(s.mode()) else "UNK")
meta=pt.set_index("subject_id"); meta=meta[meta.index.isin(eye_cohort)].copy()
meta["race"]=race.reindex(meta.index).fillna("UNK"); meta["ageband"]=(meta.anchor_age//5).astype("Int64")
meta=meta.dropna(subset=["ageband"]); members=list(meta.index)

def P(*x): return list(x)
D={
 # endocrine/metabolic
 "Diabetes mellitus":("endocrine",P("250"),P("E08","E09","E10","E11","E13")),
 "Hypothyroidism":("endocrine",P("244"),P("E03")),"Hyperthyroidism":("endocrine",P("242"),P("E05")),
 "Hyperlipidemia":("endocrine",P("2720","2721","2722","2724"),P("E780","E781","E782","E784","E785")),
 "Obesity":("endocrine",P("2780"),P("E66")),"Gout":("endocrine",P("274"),P("M10")),
 # cardiovascular
 "Essential hypertension":("cardiovascular",P("401"),P("I10")),
 "Ischemic heart disease":("cardiovascular",P("410","411","412","414"),P("I20","I21","I22","I24","I25")),
 "Heart failure":("cardiovascular",P("428"),P("I50")),
 "Atrial fibrillation":("cardiovascular",P("42731"),P("I48")),
 "Valvular heart disease":("cardiovascular",P("424"),P("I34","I35","I36","I37")),
 "Peripheral arterial disease":("cardiovascular",P("4439"),P("I739")),
 "Cardiomyopathy":("cardiovascular",P("425"),P("I42")),
 # cerebrovascular / neuro
 "Ischemic stroke":("neuro",P("434","436"),P("I63")),"TIA":("neuro",P("435"),P("G45")),
 "Parkinson disease":("neuro",P("3320"),P("G20")),"Multiple sclerosis":("neuro",P("340"),P("G35")),
 "Epilepsy":("neuro",P("345"),P("G40")),"Dementia/Alzheimer":("neuro",P("3310","290"),P("G30","F00","F03")),
 "Peripheral neuropathy":("neuro",P("356","3572"),P("G60","G62","G63")),
 "Migraine":("neuro",P("346"),P("G43")),
 # renal
 "Chronic kidney disease":("renal",P("585"),P("N18")),"Acute kidney injury":("renal",P("584"),P("N17")),
 # hepatic/GI
 "Cirrhosis":("hepatic_gi",P("5712","5715","5716"),P("K703","K717","K74")),
 "Chronic viral hepatitis":("hepatic_gi",P("0702","0703","0704","0705","5705"),P("B18")),
 "Inflammatory bowel disease":("hepatic_gi",P("555","556"),P("K50","K51")),
 "Peptic ulcer":("hepatic_gi",P("531","532","533"),P("K25","K26","K27")),
 # pulmonary
 "COPD":("pulmonary",P("491","492","496"),P("J41","J42","J43","J44")),
 "Asthma":("pulmonary",P("493"),P("J45")),
 "Pulmonary hypertension":("pulmonary",P("4160","4168"),P("I270","I272")),
 "Obstructive sleep apnea":("pulmonary",P("32723"),P("G4733")),
 "Interstitial lung disease":("pulmonary",P("515","5163"),P("J84")),
 # autoimmune/rheum
 "Rheumatoid arthritis":("autoimmune",P("714"),P("M05","M06")),
 "Systemic lupus erythematosus":("autoimmune",P("7100"),P("M32")),
 "Psoriasis":("autoimmune",P("6961"),P("L40")),"Sjogren syndrome":("autoimmune",P("7102"),P("M350")),
 "Ankylosing spondylitis":("autoimmune",P("7200"),P("M45")),
 "Sarcoidosis":("autoimmune",P("135"),P("D86")),"Systemic sclerosis":("autoimmune",P("7101"),P("M34")),
 # heme
 "Anemia":("heme",P("280","281","285"),P("D50","D51","D52","D53","D64")),
 "Leukemia":("heme",P("204","205","206","207","208"),P("C91","C92","C93","C94","C95")),
 "Lymphoma":("heme",P("200","201","202"),P("C81","C82","C83","C84","C85","C86")),
 "Multiple myeloma":("heme",P("203"),P("C90")),
 "Thrombocytopenia":("heme",P("2873","2874","2875"),P("D69")),
 "Sickle cell disease":("heme",P("2826"),P("D57")),
 "Venous thromboembolism":("heme",P("4151","4534","4536","4538","4539"),P("I26","I82")),
 # infectious
 "Sepsis":("infectious",P("99591","99592","0380","0381","0382","0383","0384","0385","0386","0388","0389"),P("A40","A41","R652")),
 "HIV":("infectious",P("042"),P("B20")),
 "Endocarditis":("infectious",P("421"),P("I33")),
 # oncologic (solid)
 "Lung cancer":("oncologic",P("162"),P("C34")),"Breast cancer":("oncologic",P("174","175"),P("C50")),
 "Prostate cancer":("oncologic",P("185"),P("C61")),
 "Colorectal cancer":("oncologic",P("153","154"),P("C18","C19","C20")),
 "Pancreatic cancer":("oncologic",P("157"),P("C25")),
 # psych
 "Major depression":("psych",P("2962","2963","311"),P("F32","F33")),
 "Anxiety disorder":("psych",P("3000","30002"),P("F41")),
 "Bipolar disorder":("psych",P("2960","2964","2965","2966"),P("F31")),
 "Schizophrenia":("psych",P("295"),P("F20")),
 # bone
 "Osteoporosis":("bone",P("7330"),P("M81","M80")),
}
def has(codes,d9,d10):
    for c,v in codes:
        if v=="9" and any(c.startswith(p) for p in d9): return True
        if v=="10" and any(c.startswith(p) for p in d10): return True
    return False

rng=np.random.default_rng(42); cv=StratifiedKFold(5,shuffle=True,random_state=42)
def auc(X,y):
    return cross_val_score(HistGradientBoostingClassifier(max_iter=150,random_state=42),X,y,cv=cv,scoring="roc_auc").mean()

mb=meta[["ageband","gender","race"]]
res=[]
print(f"\n{'disease':<30}{'sys':<14}{'n':>6}{'eye':>7}{'CBC':>7}{'both':>7}",flush=True)
for name,(sys,d9,d10) in D.items():
    cases=[s for s in members if s in sub_all and has(sub_all[s],d9,d10)]
    if len(cases)<MIN_CASES:
        continue
    cs=set(cases); pool=defaultdict(list)
    for s in members:
        if s not in cs: pool[(mb.at[s,"ageband"],mb.at[s,"gender"],mb.at[s,"race"])].append(s)
    for k in pool: rng.shuffle(pool[k])
    ctrl=[]
    for s in cases:
        k=(mb.at[s,"ageband"],mb.at[s,"gender"],mb.at[s,"race"])
        if pool[k]: ctrl.append(pool[k].pop())
    cases=cases[:len(ctrl)]; subs=cases+ctrl; y=np.array([1]*len(cases)+[0]*len(ctrl))
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
    res.append(dict(disease=name,system=sys,n_cases=len(cases),n_eye_feats=len(vocab),
                    auc_eye=a_e,auc_cbc=a_c,auc_both=a_b,
                    gain=a_b-max(a_e,a_c)))
    print(f"{name:<30}{sys:<14}{len(cases):>6}{a_e:>7.3f}{a_c:>7.3f}{a_b:>7.3f}",flush=True)
    pd.DataFrame(res).to_csv(f"{CBCEYE_ROOT}/landscape_results.csv",index=False)

df=pd.DataFrame(res)
def tag(r):
    if r.auc_both>max(r.auc_eye,r.auc_cbc)+0.015: return "synergy"
    if r.auc_eye>r.auc_cbc+0.03: return "eye-specific"
    if r.auc_cbc>r.auc_eye+0.03: return "CBC-specific"
    return "comparable"
df["class"]=df.apply(tag,axis=1)
df.to_csv(f"{CBCEYE_ROOT}/landscape_results.csv",index=False)
print("\n=== taxonomy ==="); print(df["class"].value_counts().to_string())
print(f"\ndiseases modeled: {len(df)}; mean gain from combining: {df.gain.mean():.3f}")
print("saved landscape_results.csv",flush=True)
