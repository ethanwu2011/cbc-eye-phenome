"""Rebuild §2.8 clinical-utility evidence to TRIPOD-AI standard (Lancet blocker).
Held-out evaluation by NHANES cycle (train 2005-06 D / test 2007-08 E, and reverse, cross-fitted),
with the Youden threshold chosen on the TRAINING cycle only. Reports held-out AUROC + OBSERVED
(count-based) sens/spec/PPV/NPV with bootstrap CIs, calibration (Brier + 10-bin ECE + reliability
curve), and a TRUE decision curve over a threshold-probability grid (model vs treat-all/treat-none,
classification and weighting at the same p_t). Also reports demo+CBC vs demo+eye+CBC to expose the
eye's marginal contribution. Reuses NHANES prep."""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
exec(open(f"{CBCEYE_ROOT}/nhanes_validation.py").read().split("Xd,Xe,Xc=")[0])
SEQN=df["SEQN"].to_numpy(float); cyc=np.where(SEQN<=41474,"D","E")
print("cycle split:",{k:int((cyc==k).sum()) for k in ["D","E"]},flush=True)
rng=np.random.default_rng(42)

def heldout_preds(cols,y):
    """cross-fitted: train each cycle, predict the OTHER; threshold = Youden on training cycle in-sample."""
    X=df[cols].to_numpy(float); yy=np.asarray(y,float); ok=~np.isnan(yy)
    P=np.full(len(yy),np.nan); THR=np.full(len(yy),np.nan)
    for tr_c,te_c in [("D","E"),("E","D")]:
        tr=ok&(cyc==tr_c); te=ok&(cyc==te_c)
        if yy[tr].sum()<15 or yy[te].sum()<8: return None
        med=np.nanmedian(X[tr],0); med=np.where(np.isnan(med),0,med)
        Xtr=np.where(np.isnan(X[tr]),med,X[tr]); Xte=np.where(np.isnan(X[te]),med,X[te])
        m=HistGradientBoostingClassifier(max_iter=200,early_stopping=True,random_state=42); m.fit(Xtr,yy[tr])
        ptr=m.predict_proba(Xtr)[:,1]
        fpr,tpr,thr=roc_curve(yy[tr],ptr); j=np.argmax(tpr-fpr); t=thr[j]
        P[te]=m.predict_proba(Xte)[:,1]; THR[te]=t
    keep=~np.isnan(P)
    return yy[keep],P[keep],THR[keep]

def opmetrics(y,p,thr):
    pred=(p>=thr).astype(int); P=y.sum(); N=len(y)-P
    tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
    sens=tp/max(P,1); spec=tn/max(N,1); ppv=tp/max(tp+fp,1); npv=tn/max(tn+fn,1)
    return sens,spec,ppv,npv
def ece(y,p,bins=10):
    e=0; idx=np.clip((p*bins).astype(int),0,bins-1)
    for b in range(bins):
        m=idx==b
        if m.sum(): e+=abs(p[m].mean()-y[m].mean())*m.sum()/len(y)
    return e

EX=["CKD (eGFR<60)","Albuminuria (ACR>=30)","Memory/confusion problem","Any cardiovascular",
    "Congestive heart failure","Stroke"]
rows=[]; dca=[]; calib=[]
for name in EX:
    y=OUT.get(name)
    if y is None: continue
    full=heldout_preds(DEMO+EYE+CBC,y); dc=heldout_preds(DEMO+CBC,y)
    if full is None: print(f"  skip {name} (too few held-out cases)",flush=True); continue
    yf,pf,thf=full; prev=yf.mean()
    auc_full=roc_auc_score(yf,pf); auc_dc=roc_auc_score(*dc[:2]) if dc else np.nan
    sens,spec,ppv,npv=opmetrics(yf,pf,thf)
    # bootstrap CIs (resample held-out subjects)
    B=600; bs=np.zeros((B,5))
    for b in range(B):
        ix=rng.integers(0,len(yf),len(yf)); yb,pb,tb=yf[ix],pf[ix],thf[ix]
        if yb.sum()<3 or (len(yb)-yb.sum())<3: bs[b]=np.nan; continue
        bs[b,0]=roc_auc_score(yb,pb); bs[b,1:5]=opmetrics(yb,pb,tb)
    lo,hi=np.nanpercentile(bs,2.5,0),np.nanpercentile(bs,97.5,0)
    rows.append(dict(condition=name,prev=round(prev,3),n_test=len(yf),cases_test=int(yf.sum()),
        auc=round(auc_full,3),auc_lo=round(lo[0],3),auc_hi=round(hi[0],3),
        auc_demoCBC=round(auc_dc,3),eye_marginal_auc=round(auc_full-auc_dc,3),
        sens=round(sens,2),sens_ci=f"[{lo[1]:.2f},{hi[1]:.2f}]",
        spec=round(spec,2),spec_ci=f"[{lo[2]:.2f},{hi[2]:.2f}]",
        ppv=round(ppv,2),ppv_ci=f"[{lo[3]:.2f},{hi[3]:.2f}]",
        npv=round(npv,2),npv_ci=f"[{lo[4]:.2f},{hi[4]:.2f}]",
        brier=round(brier_score_loss(yf,pf),3),ece=round(ece(yf,pf),3)))
    print(f"{name:<26} prev {prev:5.1%} AUROC {auc_full:.3f}[{lo[0]:.3f},{hi[0]:.3f}] eyeΔauc {auc_full-auc_dc:+.3f} | sens {sens:.2f} spec {spec:.2f} PPV {ppv:.2f}[{lo[3]:.2f},{hi[3]:.2f}] NPV {npv:.2f} | Brier {brier_score_loss(yf,pf):.3f} ECE {ece(yf,pf):.3f}",flush=True)
    # decision curve (true DCA over p_t grid)
    for pt in np.arange(0.02,0.51,0.02):
        pred=(pf>=pt).astype(int); tp=((pred==1)&(yf==1)).mean(); fp=((pred==1)&(yf==0)).mean()
        nb=tp-fp*(pt/(1-pt)); nb_all=prev-(1-prev)*(pt/(1-pt))
        dca.append(dict(condition=name,pt=round(pt,3),nb_model=round(nb,4),nb_all=round(nb_all,4),nb_none=0.0))
    # reliability points (deciles)
    idx=np.clip((pf*10).astype(int),0,9)
    for b in range(10):
        mb=idx==b
        if mb.sum()>=10: calib.append(dict(condition=name,bin=b,pred=round(pf[mb].mean(),3),obs=round(yf[mb].mean(),3),n=int(mb.sum())))
pd.DataFrame(rows).to_csv(f"{CBCEYE_ROOT}/utility_rebuild.csv",index=False)
pd.DataFrame(dca).to_csv(f"{CBCEYE_ROOT}/utility_dca.csv",index=False)
pd.DataFrame(calib).to_csv(f"{CBCEYE_ROOT}/utility_calib.csv",index=False)
print("\nsaved utility_rebuild.csv / utility_dca.csv / utility_calib.csv",flush=True)
