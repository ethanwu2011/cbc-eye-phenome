"""NHANES 2005-2008 convergent validation of eye+CBC complementarity (v3, comprehensive).
Multi-domain eye exposure (retinopathy + AMD + glaucoma FDT + acuity); ~50 systemic conditions
from labs + measured exam + the full MCQ series + condition questionnaires. Population-based,
imaging-based, public data. Per condition: AUROC of demo / +eye / +CBC / +both (age/sex/race always)."""
from config import NHANES_DIR
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D=NHANES_DIR; SEED=42
def rd(f): return pd.read_sas(f"{D}/{f}.xpt")
def clean(df):
    df=df.copy()
    for c in df.columns:
        if df[c].dtype.kind=="f": df.loc[df[c].abs()<1e-6,c]=0.0
    return df
FILES=["CBC","OPXRET","OPXFDT","VIX","DIQ","GHB","GLU","BPQ","BPX","BIOPRO","TCHOL","HDL","TRIGLY",
       "CRP","ALB_CR","BMX","MCQ","DPQ","RDQ","KIQ_U","SLQ","OSQ","CDQ","PFQ","AUQ","HSQ","RHQ","KIQ_P",
       "FETIB","VID","FERTIN","THYROD"]
parts=[]
for suf in ["D","E"]:
    m=rd(f"DEMO_{suf}")[["SEQN","RIDAGEYR","RIAGENDR","RIDRETH1","WTMEC2YR"]]
    for f in FILES:
        try:
            d=clean(rd(f"{f}_{suf}")); new=[c for c in d.columns if c not in m.columns]
            m=m.merge(d[["SEQN"]+[c for c in new if c!="SEQN"]],on="SEQN",how="left")
        except Exception as e: print("skip",f,suf,str(e)[:40])
    parts.append(m)
df=pd.concat(parts,ignore_index=True)
df=df[df.get("OPDURET").notna()].copy()
print(f"cohort with gradable retinal imaging: {len(df):,}")

# ---- feature blocks ----
def usable(cols,mincov=0.40):
    return [c for c in cols if c in df.columns and df[c].dtype.kind in "fi" and df[c].notna().mean()>=mincov]
EYE=list(dict.fromkeys(usable([c for c in df.columns if c.startswith("OPDU")])+
     usable(["OPDODFDT","OPDOSFDT","VIXNC1","VIXNC2","VIXPRS","VIXPRC","VIXPRA"])))
CBC=usable(["LBXWBCSI","LBXLYPCT","LBXMOPCT","LBXNEPCT","LBXEOPCT","LBXBAPCT","LBDLYMNO","LBDMONO",
     "LBDNENO","LBDEONO","LBDBANO","LBXRBCSI","LBXHGB","LBXHCT","LBXMCVSI","LBXMCHSI","LBXMC",
     "LBXRDW","LBXPLTSI","LBXMPSI"],0.3)
df["age"]=df.RIDAGEYR; df["female"]=(df.RIAGENDR==2).astype(float)
race=pd.get_dummies(df.RIDRETH1.fillna(0).astype(int),prefix="race").astype(float)
df=pd.concat([df,race],axis=1); DEMO=["age","female"]+list(race.columns)
WT=(df.get("WTMEC2YR")/2); WT=WT.fillna(WT.median()).to_numpy(float)   # combined 4-yr MEC weight
print(f"eye features {len(EYE)} | CBC features {len(CBC)}")

# ---- outcome helpers (proper missing handling) ----
def binr(col):
    s=df.get(col);
    return None if s is None else pd.Series(np.where(s==1,1.0,np.where(s==2,0.0,np.nan)),index=df.index)
def labge(s,k):  # continuous lab/measure >= k, NaN preserved
    return (s>=k).astype(float).where(s.notna()) if s is not None else None
def lable(s,k):  # continuous lab < k (low-value abnormality), NaN preserved
    return (s<k).astype(float).where(s.notna()) if s is not None else None
def geq(col,k):  # ordinal scale >= k, drop refused/dk
    s=df.get(col)
    if s is None: return None
    s=s.where(~s.isin([7,9,77,99,777,999]))
    return (s>=k).astype(float).where(s.notna())
sbp=df[[c for c in ['BPXSY1','BPXSY2','BPXSY3'] if c in df.columns]].mean(axis=1)
egfr=175*(df.get("LBXSCR").clip(lower=0.1)**-1.154)*(df.age**-0.203)*np.where(df.female==1,0.742,1)*np.where(df.RIDRETH1==4,1.212,1)
acr=(df.get("URXUMA")/df.get("URXUCR")*100) if "URXUMA" in df.columns else pd.Series(np.nan,index=df.index)
phq=df[[f"DPQ0{i}0" for i in range(1,10) if f"DPQ0{i}0" in df.columns]].replace({7:np.nan,9:np.nan}).sum(axis=1,min_count=5)
hdl=df.get("LBDHDD"); lowhdl=(hdl< np.where(df.female==1,50,40)).astype(float).where(hdl.notna())
hgb=df.get("LBXHGB"); anem=(hgb< np.where(df.female==1,12,13)).astype(float).where(hgb.notna())
wst=df.get("BMXWAIST"); abdo=(wst> np.where(df.female==1,88,102)).astype(float).where(wst.notna()) if wst is not None else None
rose=((df.get("CDQ001")==1)&(df.get("CDQ010")==1)).astype(float).where(df.get("CDQ001").isin([1,2]))
OUT={
 "Diabetes [eye=DR, expected]": labge((df.get("DIQ010")==1)|(df.get("LBXGH")>=6.5)|(df.get("LBXGLU")>=126),0.5),
 "Prediabetes (HbA1c 5.7-6.4)": ((df.get("LBXGH")>=5.7)&(df.get("LBXGH")<6.5)).astype(float).where(df.get("LBXGH").notna()),
 "Prediabetes (told)": binr("DIQ160"),
 "Hypertension": ((df.get("BPQ020")==1)|(sbp>=140)).astype(float).where((df.get("BPQ020").isin([1,2]))|sbp.notna()),
 "High total cholesterol": labge(df.get("LBXTC"),240), "Low HDL": lowhdl,
 "High triglycerides": labge(df.get("LBXTR"),200), "High LDL": labge(df.get("LBDLDL"),160),
 "Obesity (BMI>=30)": labge(df.get("BMXBMI"),30), "Abdominal obesity (waist)": abdo,
 "CKD (eGFR<60)": lable(egfr,60), "Albuminuria (ACR>=30)": labge(acr,30),
 "Hyperuricaemia": labge(df.get("LBXSUA"),7), "Elevated ALT (liver)": labge(df.get("LBXSATSI"),40),
 "Low albumin": (df.get("LBXSAL")<3.5).astype(float).where(df.get("LBXSAL").notna()),
 "High CRP (inflammation)": labge(df.get("LBXCRP"),0.3), "Anemia [CBC-circular]": anem,
 "Asthma": binr("MCQ010"), "Treated anaemia (hx)": binr("MCQ053"), "Arthritis": binr("MCQ160A"),
 "Congestive heart failure": binr("MCQ160B"), "Coronary heart disease": binr("MCQ160C"),
 "Angina (self-report)": binr("MCQ160D"), "Heart attack (MI)": binr("MCQ160E"), "Stroke": binr("MCQ160F"),
 "Emphysema": binr("MCQ160G"), "Chronic bronchitis": binr("MCQ160K"), "Liver condition": binr("MCQ160L"),
 "Thyroid problem": binr("MCQ160M"), "Cancer (any)": binr("MCQ220"), "Overweight (told)": binr("MCQ080"),
 "Any cardiovascular": df[[c for c in ['MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F'] if c in df.columns]].eq(1).any(axis=1).astype(float),
 "Rose angina (exertional)": rose, "Depression (PHQ-9>=10)": labge(phq,10),
 "Functional limitation": binr("PFQ059"), "Memory/confusion problem": binr("PFQ057"),
 "Hearing impairment": geq("AUQ131",3), "Fair/poor self-rated health": geq("HSD010",4),
 "Recent respiratory illness": ((df.get("HSQ500")==1)|(df.get("HSQ520")==1)).astype(float).where(df.get("HSQ500").isin([1,2])),
 "Urinary incontinence": binr("KIQ042"), "Weak/failing kidneys": binr("KIQ022"),
 "Sleep apnea (snort/stop)": geq("SLQ040",3), "Trouble sleeping (told)": binr("SLQ050"),
 "Hip fracture": binr("OSQ010A"), "Wrist fracture": binr("OSQ010B"), "Spine fracture": binr("OSQ010C"),
 "Osteoporosis": binr("OSQ060"), "Chronic cough": binr("RDQ031"), "Chronic wheeze": binr("RDQ070"),
 "Hysterectomy (women)": binr("RHQ280"),
 # ---- biochemistry-defined (BIOPRO chem panel) ----
 "Hyponatraemia (Na<135)": lable(df.get("LBXSNASI"),135), "Hyperkalaemia (K>5.0)": labge(df.get("LBXSKSI"),5.0),
 "Hypercalcaemia (Ca>10.5)": labge(df.get("LBXSCA"),10.5), "Hypocalcaemia (Ca<8.5)": lable(df.get("LBXSCA"),8.5),
 "High BUN (azotaemia)": labge(df.get("LBXSBU"),25), "Elevated bilirubin": labge(df.get("LBXSTB"),1.2),
 "Elevated GGT": labge(df.get("LBXSGTSI"),50), "Elevated alk phosphatase": labge(df.get("LBXSAPSI"),120),
 "Elevated AST": labge(df.get("LBXSASSI"),40), "Elevated LDH": labge(df.get("LBXSLDSI"),250),
 "High globulin": labge(df.get("LBXSGB"),3.5), "Low total protein": lable(df.get("LBXSTP"),6.4),
 "Low bicarbonate (acidosis)": lable(df.get("LBXSC3SI"),22), "High phosphorus": labge(df.get("LBXSPH"),4.5),
 # ---- iron / vitamin D / thyroid ----
 "Iron deficiency (ferritin<15)": lable(df.get("LBXFER"),15), "High ferritin": labge(df.get("LBXFER"),300),
 "Low transferrin saturation": lable(df.get("LBDPCT"),20), "Vitamin D deficiency": lable(df.get("LBDVIDMS"),50),
 "Hypothyroid (TSH>4.5)": labge(df.get("LBXTSH1"),4.5), "Hyperthyroid (TSH<0.4)": lable(df.get("LBXTSH1"),0.4),
}

def auc_block(X,y):
    X=np.asarray(X,float); y=np.asarray(y,float); ok=~np.isnan(y); X,y,wk=X[ok],y[ok],WT[ok]
    if y.sum()<30 or (len(y)-y.sum())<30: return (np.nan,np.nan)
    oof=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y):
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med); Xi=np.where(np.isnan(X),med,X)
        c=HistGradientBoostingClassifier(max_iter=200,early_stopping=True,random_state=SEED)
        c.fit(Xi[tr],y[tr]); oof[te]=c.predict_proba(Xi[te])[:,1]
    return roc_auc_score(y,oof), roc_auc_score(y,oof,sample_weight=wk)   # unweighted, survey-weighted
Xd,Xe,Xc=df[DEMO].to_numpy(float),df[EYE].to_numpy(float),df[CBC].to_numpy(float)
rows=[]
for name,y in OUT.items():
    if y is None: continue
    y=np.asarray(y,float)
    if np.nansum(y==1)<30 or np.nansum(y==0)<30: continue
    (a_d,wd)=auc_block(Xd,y); (a_e,_)=auc_block(np.hstack([Xd,Xe]),y)
    (a_c,_)=auc_block(np.hstack([Xd,Xc]),y); (a_b,wb)=auc_block(np.hstack([Xd,Xe,Xc]),y)
    ea,ca=a_b-a_c,a_b-a_e
    k=("synergy" if ea>0.01 and ca>0.01 else "eye-specific" if (a_e-a_c)>0.01 else
       "CBC-specific" if (a_c-a_e)>0.01 else "comparable")
    rows.append(dict(condition=name,n=int(np.nansum(y==1)),demo=round(a_d,3),eye=round(a_e,3),
                     cbc=round(a_c,3),both=round(a_b,3),eye_adds=round(ea,3),cbc_adds=round(ca,3),klass=k,
                     demo_wt=round(wd,3),both_wt=round(wb,3)))
res=pd.DataFrame(rows).sort_values("both",ascending=False)
print("\n"+res.to_string(index=False))
res.to_csv("nhanes_validation_results.csv",index=False)
bs=res[["eye","cbc"]].max(axis=1); n=len(res)
print(f"\n=== {n} conditions modelled ===")
print("taxonomy (mirrors MIMIC):",res.klass.value_counts().to_dict())
print(f"\nCOMBINING IS THE STORY:")
print(f"  eye+CBC > better single modality (>0.005): {(res.both>bs+0.005).sum()}/{n}   median gain over best single = +{(res.both-bs).median():.3f}")
print(f"  eye+CBC > demographics-only (>0.01):       {(res.both>res.demo+0.01).sum()}/{n}   median gain over demo = +{(res.both-res.demo).median():.3f}")
print(f"  [SURVEY-WEIGHTED] eye+CBC > demo (>0.01):  {((res.both_wt-res.demo_wt)>0.01).sum()}/{n}   median weighted gain = +{(res.both_wt-res.demo_wt).median():.3f}")
print(f"COMPLEMENTARITY (helps beyond):")
print(f"  eye adds beyond CBC (>0.01): {(res.eye_adds>0.01).sum()}/{n}    CBC adds beyond eye (>0.01): {(res.cbc_adds>0.01).sum()}/{n}")
print(f"  BOTH modalities add (synergy): {(res.klass=='synergy').sum()}/{n}")
