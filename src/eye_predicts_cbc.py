"""Which blood markers does the eye 'see'?  eye codes -> CBC analyte abnormality.

For each CBC analyte, define abnormal tails (low = bottom 15%, high = top 15% of
measured values) and ask whether eye ICD codes predict them, vs an age+sex baseline.
High eye AUC over baseline = that blood-count state has a retinal correlate.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = CBCEYE_ROOT
base = pd.read_parquet(f"{HERE}/temporal_base.parquet")
base["sex"] = (base.gender == "M").astype(int)
cbc_cols = [c for c in base.columns if c.startswith("CBC_")]
eye = pd.read_parquet(f"{HERE}/eye_at_t0.parquet")
sub_eye = eye.groupby("subject_id").code.apply(set).to_dict()

# eye vocab: codes in >=50 cohort patients
cnt = defaultdict(int)
for s in base.index:
    for c in sub_eye.get(s, ()): cnt[c] += 1
vocab = [c for c,n in cnt.items() if n >= 50]; vidx = {c:i for i,c in enumerate(vocab)}
Xeye_all = np.zeros((len(base), len(vocab)), np.float32)
for i,s in enumerate(base.index):
    for c in sub_eye.get(s, ()):
        if c in vidx: Xeye_all[i, vidx[c]] = 1.0
age_sex = base[["anchor_age","sex"]].values.astype(float)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

def auc_for(mask_pos, mask_neg, X):
    keep = mask_pos | mask_neg
    y = mask_pos[keep].astype(int)
    if y.sum() < 80 or (~y.astype(bool)).sum() < 80: return np.nan, int(y.sum())
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=120, random_state=0),
                          X[keep], y, cv=cv, method="predict_proba")[:,1]
    return roc_auc_score(y, p), int(y.sum())

rows = []
for c in cbc_cols:
    v = base[c].values.astype(float); ok = ~np.isnan(v)
    lo, hi = np.nanquantile(v, 0.15), np.nanquantile(v, 0.85)
    mid = ok & (v > lo) & (v < hi)
    for tail, m_pos in [("low", ok & (v <= lo)), ("high", ok & (v >= hi))]:
        a_eye, n = auc_for(m_pos, mid, Xeye_all)
        a_base, _ = auc_for(m_pos, mid, age_sex)
        if not np.isnan(a_eye):
            rows.append(dict(analyte=c[4:], tail=tail, n_abnormal=n,
                             auc_eye=a_eye, auc_agesex=a_base, lift=a_eye-a_base))
df = pd.DataFrame(rows).sort_values("auc_eye", ascending=False)
df.to_csv(f"{HERE}/eye_predicts_cbc.csv", index=False)
print("Which blood markers does the EYE see? (eye-code AUROC for analyte abnormality)\n")
print(f"{'analyte':<26}{'tail':<6}{'n':>6}{'eye':>7}{'age+sex':>9}{'lift':>7}")
for _,r in df.iterrows():
    flag = "  <-- retinal correlate" if (r.auc_eye>=0.62 and r.lift>=0.03) else ""
    print(f"{r['analyte']:<26}{r['tail']:<6}{int(r['n_abnormal']):>6}{r['auc_eye']:>7.3f}{r['auc_agesex']:>9.3f}{r['lift']:>+7.3f}{flag}")
print(f"\nsaved eye_predicts_cbc.csv  ({(df.auc_eye>=0.62).sum()} analyte-states the eye predicts at AUC>=0.62)")
