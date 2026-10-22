"""Fair re-run of the eye-by-CBC interaction test.

The shipped test (orthogonality_interaction_test.py) handicaps the interaction model twice.
It carries ~300 extra product terms AND is fitted at C=0.5 against the additive model's C=1.0,
so in sklearn's parameterization it is regularized twice as hard. The Methods already concede
the test is "biased against detecting interaction" and name the fix. This runs it.

  PASS 1 (all eligible diseases)  equal penalty, both models at C=1.0, DeLong p, BH-FDR.
                                  This is the "equal regularization" fix named in the Methods.
  PASS 2 (bounded subset)         penalty tuned independently for EACH model by inner 3-fold CV,
                                  so the larger model is not handicapped by carrying more terms.

If interaction stays null under both, additivity is a finding rather than an artefact.
Usage:  interaction_fair.py [n_pass1] [n_pass2]
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import cohort_build as V
import stats_utils as su

HERE=CBCEYE_ROOT; OUT=f"{HERE}/interaction_fair.csv"; OUT2=f"{HERE}/interaction_fair_tuned.csv"
CBC_K=[i for i,c in enumerate(V.ctx_cols) if c.startswith("CBC_")]
MIN_CASES=250; N_EYE=15
GRID=[0.1,0.5,1.0,4.0]

def _blocks(pc):
    subs,y,X,C_,vocab=V.build(pc)
    if len(vocab)<5 or y.sum()<MIN_CASES: return None
    Xe=X[:,:N_EYE]; Cc=C_[:,CBC_K]
    med=np.nanmedian(Cc,0); Cc=np.where(np.isnan(Cc),med,Cc)
    return subs,y,Xe,Cc

def _fold_mats(Xe,Cc,tr,te):
    sc=StandardScaler().fit(Cc[tr]); Ctr,Cte=sc.transform(Cc[tr]),sc.transform(Cc[te])
    a_tr,a_te=np.hstack([Xe[tr],Ctr]),np.hstack([Xe[te],Cte])
    i_tr=np.hstack([a_tr,(Xe[tr][:,:,None]*Ctr[:,None,:]).reshape(len(tr),-1)])
    i_te=np.hstack([a_te,(Xe[te][:,:,None]*Cte[:,None,:]).reshape(len(te),-1)])
    return a_tr,a_te,i_tr,i_te

def pass1(pc):
    """equal penalty, the fix the Methods names"""
    b=_blocks(pc)
    if b is None: return None
    subs,y,Xe,Cc=b; n=len(y); oa,oi=np.zeros(n),np.zeros(n)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(subs,y):
        a_tr,a_te,i_tr,i_te=_fold_mats(Xe,Cc,tr,te)
        oa[te]=LogisticRegression(max_iter=3000,C=1.0).fit(a_tr,y[tr]).predict_proba(a_te)[:,1]
        oi[te]=LogisticRegression(max_iter=3000,C=1.0).fit(i_tr,y[tr]).predict_proba(i_te)[:,1]
    r=su.delong_analyze(y,{"add":oa,"int":oi})
    D=r["delta"].get("add-int"); p=(1.0-D["p_gt"]) if D is not None else r["delta"]["int-add"]["p_gt"]
    return dict(phecode=pc,n_cases=int(y.sum()),
                auc_add=roc_auc_score(y,oa),auc_int=roc_auc_score(y,oi),
                d_equal=roc_auc_score(y,oi)-roc_auc_score(y,oa),p_equal=p)

def _tune(Xtr,ytr):
    best,bestC=-1,1.0
    for C in GRID:
        s=[]
        for itr,ite in StratifiedKFold(3,shuffle=True,random_state=0).split(Xtr,ytr):
            if ytr[itr].sum()<10 or ytr[ite].sum()<3: continue
            m=LogisticRegression(max_iter=1500,C=C).fit(Xtr[itr],ytr[itr])
            s.append(roc_auc_score(ytr[ite],m.predict_proba(Xtr[ite])[:,1]))
        if s and np.mean(s)>best: best,bestC=np.mean(s),C
    return bestC

def pass2(pc):
    """each model gets its own penalty, so the larger one is not handicapped"""
    b=_blocks(pc)
    if b is None: return None
    subs,y,Xe,Cc=b; n=len(y); oa,oi=np.zeros(n),np.zeros(n); Ca_,Ci_=[],[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(subs,y):
        a_tr,a_te,i_tr,i_te=_fold_mats(Xe,Cc,tr,te)
        Ca=_tune(a_tr,y[tr]); Ci=_tune(i_tr,y[tr]); Ca_.append(Ca); Ci_.append(Ci)
        oa[te]=LogisticRegression(max_iter=3000,C=Ca).fit(a_tr,y[tr]).predict_proba(a_te)[:,1]
        oi[te]=LogisticRegression(max_iter=3000,C=Ci).fit(i_tr,y[tr]).predict_proba(i_te)[:,1]
    r=su.delong_analyze(y,{"add":oa,"int":oi})
    D=r["delta"].get("add-int"); p=(1.0-D["p_gt"]) if D is not None else r["delta"]["int-add"]["p_gt"]
    return dict(phecode=pc,n_cases=int(y.sum()),d_tuned=roc_auc_score(y,oi)-roc_auc_score(y,oa),
                p_tuned=p,C_add=float(np.median(Ca_)),C_int=float(np.median(Ci_)))

def _drive(todo,fn,out,info,label):
    done=set(pd.read_csv(out).phecode.astype(str)) if os.path.exists(out) else set()
    todo=[p for p in todo if p not in done]
    print(f"{label}: {len(todo)} to run ({len(done)} cached)",flush=True)
    rows=[]
    for i,pc in enumerate(todo,1):
        try: r=fn(pc)
        except Exception as e: print(f"  [{pc}] {e}",flush=True); continue
        if not r: continue
        r["description"]=info.loc[pc,"description"]; rows.append(r)
        if i%15==0:
            k=[c for c in rows[0] if c.startswith("d_")][0]
            print(f"  [{i}/{len(todo)}] mean {k} = {np.mean([x[k] for x in rows]):+.4f}",flush=True)
            cur=pd.concat([pd.read_csv(out),pd.DataFrame(rows)],ignore_index=True) if os.path.exists(out) else pd.DataFrame(rows)
            cur.drop_duplicates("phecode",keep="last").to_csv(out,index=False); rows=[]
    if rows:
        cur=pd.concat([pd.read_csv(out),pd.DataFrame(rows)],ignore_index=True) if os.path.exists(out) else pd.DataFrame(rows)
        cur.drop_duplicates("phecode",keep="last").to_csv(out,index=False)

def main():
    n1=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
    n2=int(sys.argv[2]) if len(sys.argv)>2 else 60
    land=pd.read_parquet(f"{HERE}/landscape_full.parquet"); land["phecode"]=land.phecode.astype(str)
    cand=land[(land.circularity_tier=="clean")&(land.klass!="no signal")&(land.n_cases>=MIN_CASES)]
    cand=cand.sort_values("n_cases",ascending=False)
    info=land.set_index("phecode")[["description","klass"]]
    _drive(list(cand.phecode)[:n1],pass1,OUT,info,"PASS 1 equal penalty")
    _drive(list(cand.phecode)[:n2],pass2,OUT2,info,"PASS 2 independently tuned")
    d1=pd.read_csv(OUT); d1["q_equal"]=su.benjamini_hochberg(d1.p_equal.values); d1.to_csv(OUT,index=False)
    print(f"\n=== FAIR INTERACTION TEST ===")
    print(f"  shipped   C=0.5 vs C=1.0  mean delta -0.0212   0 of 435 significant")
    print(f"  PASS 1    equal C=1.0     mean delta {d1.d_equal.mean():+.4f}   median {d1.d_equal.median():+.4f}   n={len(d1)}")
    print(f"            significant (q<0.05 and delta>0.005): {((d1.q_equal<0.05)&(d1.d_equal>0.005)).sum()} / {len(d1)}")
    print(f"            diseases with ANY positive delta: {(d1.d_equal>0).sum()} / {len(d1)}")
    if os.path.exists(OUT2):
        d2=pd.read_csv(OUT2); d2["q_tuned"]=su.benjamini_hochberg(d2.p_tuned.values); d2.to_csv(OUT2,index=False)
        print(f"  PASS 2    tuned per model mean delta {d2.d_tuned.mean():+.4f}   median {d2.d_tuned.median():+.4f}   n={len(d2)}")
        print(f"            median chosen C  additive {d2.C_add.median():.2f}   interaction {d2.C_int.median():.2f}")
        print(f"            significant (q<0.05 and delta>0.005): {((d2.q_tuned<0.05)&(d2.d_tuned>0.005)).sum()} / {len(d2)}")

if __name__=="__main__": main()
