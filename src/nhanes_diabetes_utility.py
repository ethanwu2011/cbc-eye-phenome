"""Push toward NM: (1) does eye+CBC complementarity survive diabetes adjustment / hold in
non-diabetics?  (2) clinical-utility operating points (sens/spec/PPV/NPV + decision-curve net
benefit) at population prevalence. Reuses the NHANES prep from nhanes_validation.py."""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
# reuse prep (df, EYE, CBC, DEMO, WT, OUT) WITHOUT running the main analysis loop:
exec(open(f"{CBCEYE_ROOT}/nhanes_validation.py").read().split("Xd,Xe,Xc=")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
SEED=42

dm=((df.get("DIQ010")==1)|(df.get("LBXGH")>=6.5)|(df.get("LBXGLU")>=126)).fillna(False).to_numpy()
print(f"diabetic {int(dm.sum()):,} | non-diabetic {int((~dm).sum()):,}")
df["dm_status"]=dm.astype(float); df["hba1c"]=df.get("LBXGH").fillna(df.get("LBXGH").median())
DEMO_DM=DEMO+["dm_status","hba1c"]

def aucm(cols,y,mask):
    X=df[cols].to_numpy(float)[mask]; yy=np.asarray(y,float)[mask]; ok=~np.isnan(yy); X,yy=X[ok],yy[ok]
    if yy.sum()<25 or (len(yy)-yy.sum())<25: return (np.nan,None,None)
    oof=np.zeros(len(yy))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,yy):
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med); Xi=np.where(np.isnan(X),med,X)
        c=HistGradientBoostingClassifier(max_iter=200,early_stopping=True,random_state=SEED)
        c.fit(Xi[tr],yy[tr]); oof[te]=c.predict_proba(Xi[te])[:,1]
    return (roc_auc_score(yy,oof),yy,oof)

# clinically meaningful, non-circular, diabetes-confound-relevant or neuro
KEEP=["CKD (eGFR<60)","Albuminuria (ACR>=30)","Congestive heart failure","Heart attack (MI)",
 "Coronary heart disease","Angina (self-report)","Stroke","Hypertension","High CRP (inflammation)",
 "Obesity (BMI>=30)","Low HDL","High triglycerides","Hyperuricemia","Elevated ALT (liver)","Elevated GGT",
 "Memory/confusion problem","Weak/failing kidneys","Depression (PHQ-9>=10)","Hyponatremia (Na<135)",
 "High BUN (azotemia)","Any cardiovascular","Hearing impairment","Fair/poor self-rated health"]
allm=np.ones(len(df),bool); nd=~dm
rows=[]
for name in KEEP:
    y=OUT.get(name)
    if y is None: continue
    base=aucm(DEMO,y,allm)[0]; both=aucm(DEMO+EYE+CBC,y,allm)[0]
    baseDM=aucm(DEMO_DM,y,allm)[0]; bothDM=aucm(DEMO_DM+EYE+CBC,y,allm)[0]
    bnd=aucm(DEMO,y,nd)[0]; bothnd=aucm(DEMO+EYE+CBC,y,nd)[0]; cbcnd=aucm(DEMO+CBC,y,nd)[0]
    f=lambda a,b: round(a-b,3) if not (np.isnan(a) or np.isnan(b)) else None
    rows.append(dict(condition=name, both_vs_demo=f(both,base),
        both_vs_demoPlusDiabetes=f(bothDM,baseDM), both_vs_demo_NONdiab=f(bothnd,bnd),
        eye_adds_NONdiab=f(bothnd,cbcnd)))
r=pd.DataFrame(rows); print("\n"+r.to_string(index=False)); r.to_csv("nhanes_diabetes_adjusted.csv",index=False)
def cnt(col): s=r[col].dropna(); return f"{int((s>0.01).sum())}/{len(s)}"
print("\n=== DIABETES INDEPENDENCE ===")
print(f"  eye+CBC still beats (demo + diabetes status + HbA1c):  {cnt('both_vs_demoPlusDiabetes')}")
print(f"  eye+CBC still beats demo in NON-diabetics only:        {cnt('both_vs_demo_NONdiab')}")
print(f"  eye still adds beyond CBC in NON-diabetics:            {cnt('eye_adds_NONdiab')}")

print("\n=== CLINICAL UTILITY (eye+CBC+demo; Youden threshold; population prevalence) ===")
for name in ["CKD (eGFR<60)","Congestive heart failure","Stroke","Albuminuria (ACR>=30)","Any cardiovascular"]:
    auc,yy,oof=aucm(DEMO+EYE+CBC,OUT.get(name),allm)
    if yy is None: continue
    fpr,tpr,thr=roc_curve(yy,oof); j=int(np.argmax(tpr-fpr)); se,sp=tpr[j],1-fpr[j]; prev=yy.mean()
    ppv=se*prev/(se*prev+(1-sp)*(1-prev)); npv=sp*(1-prev)/(sp*(1-prev)+(1-se)*prev)
    pt=prev; pred=(oof>=thr[j]).astype(int)
    TP=((pred==1)&(yy==1)).mean(); FP=((pred==1)&(yy==0)).mean()
    nb=TP-FP*(pt/(1-pt)); nb_all=prev-(1-prev)*(pt/(1-pt))
    print(f"  {name:<26} AUROC {auc:.3f} prev {prev:4.1%} | sens {se:.2f} spec {sp:.2f} PPV {ppv:.2f} NPV {npv:.2f} | net-benefit {nb:+.3f} vs treat-all {nb_all:+.3f}")
