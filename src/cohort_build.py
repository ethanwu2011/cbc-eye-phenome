"""Matched case-control cohort assembly, shared by the orthogonality and interaction tests.

For a PheCode, this builds the 1 to 1 matched design the paper uses throughout:
cases are matched to controls on five-year age band, sex and race, the ophthalmic
vocabulary is the most frequent codes in the matched set subject to a minimum
count, and the context block is age, the 20 CBC analytes and the two utilization
counters.

Inputs are the intermediate tables written by temporal_base.py and
build_phecode_onset.py. They are patient level and are therefore not
redistributed with this repository. Regenerate them from your own credentialed
copy of MIMIC-IV before running anything that imports this module.

This logic previously lived inside a model-fitting script that also imported
PyTorch. It is factored out here so the tests reported in the paper carry no
deep-learning dependency.
"""
import numpy as np
import pandas as pd
from collections import defaultdict

from config import CBCEYE_ROOT, MIMIC_HOSP, require

require(MIMIC_HOSP, "MIMIC_HOSP")

base = pd.read_parquet(f"{CBCEYE_ROOT}/temporal_base.parquet")
base["sex"] = (base.gender == "M").astype(int)
base["ageband"] = (base.anchor_age // 5)

cbc_cols = [c for c in base.columns if c.startswith("CBC_")]
ctx_cols = ["anchor_age"] + cbc_cols + ["n_codes_le_t0", "n_adm_le_t0"]

eye = pd.read_parquet(f"{CBCEYE_ROOT}/eye_at_t0.parquet")
sub_eye = eye.groupby("subject_id").code.apply(set).to_dict()

long = pd.read_parquet(f"{CBCEYE_ROOT}/phecode_long.parquet")
long["phecode"] = long.phecode.astype(str)
cases_by = long.groupby("phecode").subject_id.apply(set)
desc_by = long.drop_duplicates("phecode").set_index("phecode").description

dl = pd.read_csv(f"{MIMIC_HOSP}/d_icd_diagnoses.csv.gz", dtype=str)
code_title = dict(zip(dl.icd_code, dl.long_title))


def build(phecode, max_eye=30):
    """Return (subjects, y, X_eye, C_context, vocab) for one PheCode.

    Matching is on (age band, sex, race). Controls are drawn without replacement
    from the non-case pool, and cases are truncated to the number of controls
    actually found, so the design stays exactly 1 to 1.
    """
    cases = [s for s in cases_by[phecode] if s in base.index]
    rng = np.random.default_rng(42)
    pool = defaultdict(list)
    cset = set(cases)
    for s in base.index:
        if s not in cset:
            pool[(base.at[s, "ageband"], base.at[s, "sex"], base.at[s, "race"])].append(s)
    for k in pool:
        rng.shuffle(pool[k])
    ctrl = []
    for s in cases:
        k = (base.at[s, "ageband"], base.at[s, "sex"], base.at[s, "race"])
        if pool[k]:
            ctrl.append(pool[k].pop())
    cases = cases[:len(ctrl)]
    subs = cases + ctrl
    y = np.array([1] * len(cases) + [0] * len(ctrl))

    cnt = defaultdict(int)
    for s in subs:
        for c in sub_eye.get(s, ()):
            cnt[c] += 1
    vocab = [c for c, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:max_eye] if cnt[c] >= 20]
    vidx = {c: i for i, c in enumerate(vocab)}
    X = np.zeros((len(subs), len(vocab)), np.float32)
    for i, s in enumerate(subs):
        for c in sub_eye.get(s, ()):
            if c in vidx:
                X[i, vidx[c]] = 1.0

    C = base.loc[subs, ctx_cols].values.astype(float)
    return subs, y, X, C, vocab
