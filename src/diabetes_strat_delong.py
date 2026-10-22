"""Put inference on the diabetes-stratification (reviewer P1). For the 23 NHANES conditions,
DeLong paired tests + BH-FDR for three contrasts: (i) eye+CBC > demo+diabetes+HbA1c (full),
(ii) eye+CBC > demo (non-diabetics), (iii) eye+CBC > CBC = eye adds (non-diabetics). Reports
FDR-surviving counts with 95% CIs, replacing the bare ΔAUROC>0.01 tallies. Reuses NHANES prep."""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import stats_utils as su
exec(open(f"{CBCEYE_ROOT}/nhanes_validation.py").read().split("Xd,Xe,Xc=")[0])
dm=((df.get("DIQ010")==1)|(df.get("LBXGH")>=6.5)|(df.get("LBXGLU")>=126)).fillna(False).to_numpy()
df["dm_status"]=dm.astype(float); df["hba1c"]=df.get("LBXGH").fillna(df.get("LBXGH").median())
DEMO_DM=DEMO+["dm_status","hba1c"]; cv=StratifiedKFold(5,shuffle=True,random_state=42)
def oof(cols,mask,y):
    X=df[cols].to_numpy(float)[mask]; yy=np.asarray(y,float)[mask]; ok=~np.isnan(yy); X,yy=X[ok],yy[ok]
    if yy.sum()<25 or (len(yy)-yy.sum())<25: return None,None
    s=np.zeros(len(yy))
    for tr,te in cv.split(X,yy):
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med); Xi=np.where(np.isnan(X),med,X)
        c=HistGradientBoostingClassifier(max_iter=200,early_stopping=True,random_state=42); c.fit(Xi[tr],yy[tr]); s[te]=c.predict_proba(Xi[te])[:,1]
    return yy,s
KEEP=["CKD (eGFR<60)","Albuminuria (ACR>=30)","Congestive heart failure","Heart attack (MI)",
 "Coronary heart disease","Angina (self-report)","Stroke","Hypertension","High CRP (inflammation)",
 "Obesity (BMI>=30)","Low HDL","High triglycerides","Hyperuricaemia","Elevated ALT (liver)","Elevated GGT",
 "Memory/confusion problem","Weak/failing kidneys","Depression (PHQ-9>=10)","Hyponatraemia (Na<135)",
 "High BUN (azotaemia)","Any cardiovascular","Hearing impairment","Fair/poor self-rated health"]
allm=np.ones(len(df),bool); nd=~dm
rows=[]
for name in KEEP:
    y=OUT.get(name)
    if y is None: continue
    rec=dict(condition=name)
    # contrast i: both vs demoDM (full)
    yy,sb=oof(DEMO+EYE+CBC,allm,y); _,sd=oof(DEMO_DM,allm,y)
    if yy is not None:
        D=su.delong_analyze(yy,{"both":sb,"demoDM":sd}); dd=D["delta"].get("both-demoDM")
        rec["i_delta"]=dd["delta"]; rec["i_p"]=dd["p_gt"]; rec["i_ci"]=D["ci"]["both"]
    # contrast ii: both vs demo (non-diab)
    yy,sb=oof(DEMO+EYE+CBC,nd,y); _,sdm=oof(DEMO,nd,y)
    if yy is not None:
        D=su.delong_analyze(yy,{"both":sb,"demo":sdm}); dd=D["delta"].get("both-demo")
        rec["ii_delta"]=dd["delta"]; rec["ii_p"]=dd["p_gt"]
    # contrast iii: both vs cbc (non-diab) = eye adds
    yy,sb=oof(DEMO+EYE+CBC,nd,y); _,sc=oof(DEMO+CBC,nd,y)
    if yy is not None:
        D=su.delong_analyze(yy,{"both":sb,"cbc":sc}); dd=D["delta"].get("both-cbc")
        rec["iii_delta"]=dd["delta"]; rec["iii_p"]=dd["p_gt"]; rec["iii_ci_eyeadds_se"]=dd["se"]
    rows.append(rec)
r=pd.DataFrame(rows)
for k in ["i","ii","iii"]:
    p=r[f"{k}_p"]; q=np.full(len(r),np.nan); m=p.notna().to_numpy()
    q[m]=su.benjamini_hochberg(p[m].to_numpy()); r[f"{k}_q"]=q
r.to_csv(f"{CBCEYE_ROOT}/diabetes_strat_delong.csv",index=False)
def line(k,lab):
    d=r[f"{k}_delta"]; p=r[f"{k}_p"]; q=r[f"{k}_q"]; n=p.notna().sum()
    print(f"  {lab}\n     ΔAUROC>0.01: {int((d>0.01).sum())}/{n} | nominal p<.05 & Δ>0: {int(((p<0.05)&(d>0)).sum())}/{n} | BH-FDR q<.05 & Δ>0: {int(((q<0.05)&(d>0)).sum())}/{n}")
print("=== DIABETES-STRAT with DeLong + BH-FDR (n=23 conditions) ===")
line("i","(i) eye+CBC > demo+diabetes-status+HbA1c  [full cohort]")
line("ii","(ii) eye+CBC > demographics              [non-diabetics]")
line("iii","(iii) eye+CBC > CBC  (eye adds beyond CBC)[non-diabetics]")
print("\n  eye-adds non-diab survivors (q<0.05):",
      r.loc[(r.iii_q<0.05)&(r.iii_delta>0),"condition"].tolist())
print("saved diabetes_strat_delong.csv",flush=True)
