"""Is eye+CBC ADDITIVE (orthogonal) or do they INTERACT?

Per disease, compare 5-fold OOF AUROC of:
  additive    = logistic[eye main + CBC main]
  interaction = logistic[eye main + CBC main + eye x CBC products]
If interaction does NOT beat additive (ΔAUC ~ 0) across the phenome, the two
modalities combine additively -> orthogonal / independent axes (no effect-modification).
DeLong p per disease; BH-FDR across diseases for 'any real interaction'.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import cohort_build as V
import stats_utils as su

HERE = CBCEYE_ROOT; OUT = f"{HERE}/orthogonality_interaction_test.csv"
CBC_K = [i for i, c in enumerate(V.ctx_cols) if c.startswith("CBC_")]
MIN_CASES = 250; N_EYE = 15  # top-15 eye x ~20 CBC interactions

def run(pc):
    subs, y, X, C, vocab = V.build(pc)
    if len(vocab) < 5 or y.sum() < MIN_CASES: return None
    Xe = X[:, :N_EYE]                                   # top eye features (freq-sorted)
    Cc = C[:, CBC_K]; med = np.nanmedian(Cc, 0); Cc = np.where(np.isnan(Cc), med, Cc)
    oof_a, oof_i = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(subs, y):
        sc = StandardScaler().fit(Cc[tr]); Ctr, Cte = sc.transform(Cc[tr]), sc.transform(Cc[te])
        add_tr, add_te = np.hstack([Xe[tr], Ctr]), np.hstack([Xe[te], Cte])
        # interaction block = eye_j * cbc_k
        it_tr = (Xe[tr][:, :, None] * Ctr[:, None, :]).reshape(len(tr), -1)
        it_te = (Xe[te][:, :, None] * Cte[:, None, :]).reshape(len(te), -1)
        int_tr, int_te = np.hstack([add_tr, it_tr]), np.hstack([add_te, it_te])
        oof_a[te] = LogisticRegression(max_iter=3000, C=1.0).fit(add_tr, y[tr]).predict_proba(add_te)[:, 1]
        oof_i[te] = LogisticRegression(max_iter=3000, C=0.5).fit(int_tr, y[tr]).predict_proba(int_te)[:, 1]
    r = su.delong_analyze(y, {"add": oof_a, "int": oof_i})
    return dict(phecode=pc, auc_additive=r["auc"]["add"], auc_interaction=r["auc"]["int"],
                d_interaction=r["auc"]["int"] - r["auc"]["add"],
                p_int_gt_add=1.0 - r["delta"]["add-int"]["p_gt"])

def main():
    land = pd.read_parquet(f"{HERE}/landscape_full.parquet"); land["phecode"] = land.phecode.astype(str)
    cand = land[(land.circularity_tier == "clean") & (land.klass != "no signal") & (land.n_cases >= MIN_CASES)]
    cand = cand.sort_values("n_cases", ascending=False)
    done = set(pd.read_csv(OUT).phecode.astype(str)) if os.path.exists(OUT) else set()
    todo = cand[~cand.phecode.isin(done)]
    print(f"clean n>={MIN_CASES}: {len(cand)} | done {len(done)} | to run {len(todo)}", flush=True)
    info = land.set_index("phecode")[["description", "klass"]]
    rows = []
    for i, pc in enumerate(todo.phecode, 1):
        r = run(pc)
        if r:
            r["description"] = info.loc[pc, "description"]; r["klass"] = info.loc[pc, "klass"]
            rows.append(r)
            if i % 25 == 0:
                print(f"  [{i}/{len(todo)}] running mean Δ={np.mean([x['d_interaction'] for x in rows]):+.4f}", flush=True)
                cur = pd.concat([pd.read_csv(OUT), pd.DataFrame(rows)], ignore_index=True) if os.path.exists(OUT) else pd.DataFrame(rows)
                cur.drop_duplicates("phecode", keep="last").to_csv(OUT, index=False); rows = []
    if rows:
        cur = pd.concat([pd.read_csv(OUT), pd.DataFrame(rows)], ignore_index=True) if os.path.exists(OUT) else pd.DataFrame(rows)
        cur.drop_duplicates("phecode", keep="last").to_csv(OUT, index=False)
    df = pd.read_csv(OUT)
    df["q"] = su.benjamini_hochberg(df.p_int_gt_add.values)
    df.to_csv(OUT, index=False)
    print(f"\n=== RESULT ({len(df)} diseases) ===", flush=True)
    print(f"mean Δ(interaction - additive) AUROC = {df.d_interaction.mean():+.4f}", flush=True)
    print(f"median Δ = {df.d_interaction.median():+.4f}", flush=True)
    print(f"diseases where interaction significantly beats additive (q<0.05 & Δ>0.005): "
          f"{((df.q<0.05)&(df.d_interaction>0.005)).sum()} / {len(df)}", flush=True)
    print("\n-> if mean Δ ~ 0 and few significant: eye+CBC combine ADDITIVELY = orthogonal", flush=True)

if __name__ == "__main__":
    main()
