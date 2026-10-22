"""Phenome-wide eye + CBC -> systemic disease landscape (MIMIC-IV).

Per PheCode (>=MIN_CASES, excluding sense-organ phenotypes):
  - cases = eye-cohort patients with the phecode; controls = 1:1 matched on
    age-band/sex/race (sex dropped from matching for sex-specific phenotypes).
  - features: binary eye ICD codes (vocab built IN-FOLD on train only) and
    most-recent CBC analytes (imputed with TRAIN median in-fold).
  - subject appears once per disease -> StratifiedKFold is subject-safe.
  - HistGBDT eye-only / CBC-only / both; pooled out-of-fold predictions.
  - DeLong AUROC CIs + paired Delta-AUROC tests (stats_utils).
  - circularity tags (circularity); eye-manifestation ablation where applicable.
Parallel over phecodes (joblib), incremental + resumable to landscape_full.parquet.

Usage: python landscape_phewide.py [n_jobs] [n_seeds]
"""
from config import CBCEYE_ROOT, MIMIC_HOSP
import os, sys, warnings, hashlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import stats_utils as su
import circularity as circ

HERE = CBCEYE_ROOT; M = MIMIC_HOSP
OUT = f"{HERE}/landscape_full.parquet"
MIN_CASES = 150
N_FOLDS = 5
EYE_MIN_TRAIN = 10          # eye code must appear in >=10 TRAIN patients to be a feature

_STATE = None
def get_state():
    global _STATE
    if _STATE is not None:
        return _STATE
    dx = pd.read_csv(f"{M}/diagnoses_icd.csv.gz", usecols=["subject_id","icd_code","icd_version"], dtype=str)
    dx["code"] = dx.icd_code.str.strip()
    def is_eye(c, v):
        if v == "9":  return c[:3].isdigit() and 360 <= int(c[:3]) <= 379
        if v == "10": return len(c) >= 3 and c[0] == "H" and c[1:3].isdigit() and int(c[1:3]) <= 59
        return False
    dx["eye"] = [is_eye(c, v) for c, v in zip(dx.code, dx.icd_version)]
    eye = dx[dx.eye]
    eye_cohort = set(eye.subject_id.unique())
    sub_eye = eye.groupby("subject_id").code.apply(set).to_dict()
    dl = pd.read_csv(f"{M}/d_icd_diagnoses.csv.gz", dtype=str)
    code_title = dict(zip(dl.icd_code, dl.long_title))

    # CBC most-recent matrix
    le = pd.read_parquet(f"{HERE}/cbc_long.parquet")
    le["charttime"] = pd.to_datetime(le.charttime); le = le.sort_values("charttime")
    piv = le.groupby(["subject_id","itemid"]).valuenum.last().unstack()
    lab = pd.read_csv(f"{M}/d_labitems.csv.gz", dtype=str)
    lbl = dict(zip(lab.itemid, lab.label))
    cov = piv.notna().mean(); keep = sorted([c for c in piv.columns if cov[c] >= 0.40], key=lambda c: -cov[c])
    piv = piv[keep]; piv.columns = [lbl.get(c, c) for c in keep]
    cbc = piv.loc[:, ~pd.Index(piv.columns).duplicated()]
    cbc_med = cbc.median().values
    cbc_idx = {s: i for i, s in enumerate(cbc.index)}
    cbc_arr = cbc.values            # RAW, NaNs intact -> imputed IN-FOLD (no leakage)
    analytes = list(cbc.columns)

    pt = pd.read_csv(f"{M}/patients.csv.gz", usecols=["subject_id","gender","anchor_age"], dtype=str)
    pt["anchor_age"] = pd.to_numeric(pt.anchor_age, errors="coerce")
    adm = pd.read_csv(f"{M}/admissions.csv.gz", usecols=["subject_id","race"], dtype=str)
    race = adm.groupby("subject_id").race.agg(lambda s: s.mode().iat[0] if len(s.mode()) else "UNK")
    meta = pt.set_index("subject_id"); meta = meta[meta.index.isin(eye_cohort)].copy()
    meta["race"] = race.reindex(meta.index).fillna("UNK")
    meta["ageband"] = (meta.anchor_age // 5)
    meta = meta.dropna(subset=["ageband"]); meta["ageband"] = meta["ageband"].astype(int)
    _STATE = dict(members=list(meta.index), gender=meta.gender.to_dict(), race=meta.race.to_dict(),
                  ageband=meta.ageband.to_dict(), sub_eye=sub_eye, code_title=code_title,
                  cbc_idx=cbc_idx, cbc_arr=cbc_arr, cbc_med=cbc_med, analytes=analytes)
    return _STATE


def _eye_matrix(subs, vocab_idx, sub_eye):
    X = np.zeros((len(subs), len(vocab_idx)), dtype=np.float32)
    for i, s in enumerate(subs):
        for c in sub_eye.get(s, ()):
            j = vocab_idx.get(c)
            if j is not None: X[i, j] = 1.0
    return X


def _cbc_matrix(subs, st):
    idx, arr = st["cbc_idx"], st["cbc_arr"]
    nanrow = np.full(arr.shape[1], np.nan)
    return np.vstack([arr[idx[s]] if s in idx else nanrow for s in subs])


def _rank_auc(v, y):
    order = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[order] = np.arange(1, len(v)+1)
    n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return 0.5
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)


def run_one(phecode, desc, group, cases, seed=0):
    try:
        st = get_state()
        members, gen, rac, age = st["members"], st["gender"], st["race"], st["ageband"]
        sub_eye = st["sub_eye"]
        cases = [s for s in cases if s in age]
        if len(cases) < MIN_CASES: return None
        caseset = set(cases)
        # sex-specific? (>95% one gender among cases) -> restrict cohort, don't match on sex
        gvals = [gen[s] for s in cases]
        frac_f = np.mean([g == "F" for g in gvals]); sexsp = frac_f > 0.95 or frac_f < 0.05
        sex_restrict = "F" if frac_f > 0.95 else ("M" if frac_f < 0.05 else None)

        rng = np.random.default_rng(abs(hash((phecode, seed))) % (2**32))
        pool = defaultdict(list)
        for s in members:
            if s in caseset: continue
            if sex_restrict and gen[s] != sex_restrict: continue
            key = (age[s], rac[s]) if sexsp else (age[s], gen[s], rac[s])
            pool[key].append(s)
        for k in pool: rng.shuffle(pool[k])
        ctrl = []
        for s in cases:
            key = (age[s], rac[s]) if sexsp else (age[s], gen[s], rac[s])
            if pool[key]: ctrl.append(pool[key].pop())
        if len(ctrl) < MIN_CASES: return None
        cs = cases[:len(ctrl)]
        subs = cs + ctrl; y = np.array([1]*len(cs) + [0]*len(ctrl))

        # full-set eye vocab (descriptive count) + manifestation codes
        cnt = defaultdict(int)
        for s in subs:
            for c in sub_eye.get(s, ()): cnt[c] += 1
        full_vocab = [c for c, n in cnt.items() if n >= EYE_MIN_TRAIN]
        manifest = circ.manifestation_eye_codes(desc, {c: st["code_title"].get(c, "") for c in full_vocab})

        Xcbc_full = _cbc_matrix(subs, st)
        oof = {k: np.zeros(len(subs)) for k in ("eye","cbc","both","eye_nm")}
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in skf.split(subs, y):
            tr_subs = [subs[i] for i in tr]
            # in-fold eye vocab from TRAIN only
            c2 = defaultdict(int)
            for s in tr_subs:
                for c in sub_eye.get(s, ()): c2[c] += 1
            vocab = [c for c, n in c2.items() if n >= EYE_MIN_TRAIN]
            vidx = {c: i for i, c in enumerate(vocab)}
            Xe = _eye_matrix(subs, vidx, sub_eye)
            Xenm = None
            if manifest:
                vidx2 = {c: i for i, c in enumerate([c for c in vocab if c not in manifest])}
                Xenm = _eye_matrix(subs, vidx2, sub_eye)
            # in-fold CBC imputation: TRAIN median only
            med_tr = np.nanmedian(Xcbc_full[tr], axis=0)
            med_tr = np.where(np.isnan(med_tr), st["cbc_med"], med_tr)
            Xc = np.where(np.isnan(Xcbc_full), med_tr, Xcbc_full)
            def fit_pred(Xtr_full):
                g = HistGradientBoostingClassifier(max_iter=150, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=12, random_state=seed)
                g.fit(Xtr_full[tr], y[tr]); return g.predict_proba(Xtr_full[te])[:, 1]
            oof["eye"][te] = fit_pred(Xe)
            oof["cbc"][te] = fit_pred(Xc)
            oof["both"][te] = fit_pred(np.hstack([Xe, Xc]))
            if manifest is not None and len(manifest) and Xenm is not None and Xenm.shape[1] > 0:
                oof["eye_nm"][te] = fit_pred(Xenm)

        r = su.delong_analyze(y, {"eye": oof["eye"], "cbc": oof["cbc"], "both": oof["both"]})
        best = max(r["auc"], key=r["auc"].get)
        # eye-manifestation ablation drop (eye-only AUC lost when removing the disease's own ocular codes)
        manifest_drop = np.nan
        if manifest:
            manifest_drop = r["auc"]["eye"] - _rank_auc(oof["eye_nm"], y)
        # single CBC analyte dominance (univariate)
        uni = [ _rank_auc(Xcbc_full[:, k], y) for k in range(Xcbc_full.shape[1]) ]
        uni = [max(a, 1-a) for a in uni]
        top_uni = max(uni) if uni else 0.5
        single_dom = (r["auc"]["cbc"] > 0.70) and (top_uni >= 0.95 * r["auc"]["cbc"])
        tier = circ.tier(desc, eye_ablation_drop=(0 if np.isnan(manifest_drop) else manifest_drop),
                         single_analyte_dominant=single_dom)

        # delta keys are ordered "a-b" for the input order eye,cbc,both
        d_bc = r["delta"]["cbc-both"]; d_be = r["delta"]["eye-both"]; d_ec = r["delta"]["eye-cbc"]
        return dict(phecode=phecode, description=desc, group=group, n_cases=len(cs),
            n_eye_feats=len(full_vocab),
            auc_eye=r["auc"]["eye"], auc_cbc=r["auc"]["cbc"], auc_both=r["auc"]["both"],
            ci_eye_lo=r["ci"]["eye"][0], ci_eye_hi=r["ci"]["eye"][1],
            ci_cbc_lo=r["ci"]["cbc"][0], ci_cbc_hi=r["ci"]["cbc"][1],
            ci_both_lo=r["ci"]["both"][0], ci_both_hi=r["ci"]["both"][1],
            best_model=best, p_best_gt_half=su.auc_gt_half_p(r["auc"][best], r["var"][best]),
            d_both_vs_cbc=-d_bc["delta"], p_both_gt_cbc=1.0 - d_bc["p_gt"],
            d_both_vs_eye=-d_be["delta"], p_both_gt_eye=1.0 - d_be["p_gt"],
            d_eye_vs_cbc=d_ec["delta"], p_eye_vs_cbc_two=d_ec["p_two"],
            manifest_n=len(manifest), manifest_drop=manifest_drop,
            single_analyte_dominant=single_dom, circularity_tier=tier)
    except Exception as e:
        return dict(phecode=phecode, description=desc, error=str(e))


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    long = pd.read_parquet(f"{HERE}/phecode_long.parquet")
    info = long.drop_duplicates("phecode")[["phecode","description","group"]].set_index("phecode")
    cases_by = long.groupby("phecode").subject_id.apply(list)
    counts = long.groupby("phecode").subject_id.nunique()
    todo = [p for p in counts.index if counts[p] >= MIN_CASES
            and pd.notna(info.loc[p, "description"]) and pd.notna(info.loc[p, "group"])
            and info.loc[p, "group"] != "sense organs"]
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_parquet(OUT).phecode)
    jobs = [(p, info.loc[p, "description"], info.loc[p, "group"], cases_by[p]) for p in todo if p not in done]
    print(f"phecodes total>= {MIN_CASES}: {len(todo)} | already done: {len(done)} | to run: {len(jobs)} | n_jobs={n_jobs}", flush=True)

    rows = []
    n = 0
    for res in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(run_one)(p, d, g, c) for p, d, g, c in jobs):
        n += 1
        if res is None: continue
        rows.append(res)
        if "error" in res:
            print(f"  [{n}] ERR {res['phecode']} {res.get('description','')[:30]}: {res['error'][:60]}", flush=True)
        else:
            print(f"  [{n}/{len(jobs)}] {str(res['description'])[:34]:<34} n={res['n_cases']:>5} "
                  f"eye={res['auc_eye']:.3f} cbc={res['auc_cbc']:.3f} both={res['auc_both']:.3f} {res['circularity_tier']}", flush=True)
        if len(rows) % 40 == 0:
            cur = pd.concat([pd.read_parquet(OUT), pd.DataFrame(rows)], ignore_index=True) if os.path.exists(OUT) else pd.DataFrame(rows)
            cur.drop_duplicates("phecode", keep="last").to_parquet(OUT); rows = []
    if rows or not os.path.exists(OUT):
        cur = pd.concat([pd.read_parquet(OUT), pd.DataFrame(rows)], ignore_index=True) if os.path.exists(OUT) else pd.DataFrame(rows)
        cur.drop_duplicates("phecode", keep="last").to_parquet(OUT)
    final = pd.read_parquet(OUT)
    print(f"\nDONE. {len(final)} phecodes in {OUT}; errors: {final['error'].notna().sum() if 'error' in final else 0}", flush=True)


if __name__ == "__main__":
    main()
