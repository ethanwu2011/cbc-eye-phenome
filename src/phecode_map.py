"""Map MIMIC-IV ICD-9-CM / ICD-10-CM diagnoses -> PheCode v1.2 phenotypes.

Uses the PheWAS/PheWAS v1.2 maps (cached .rda in phecode/). MIMIC codes are
dot-less; map codes have dots -> normalize by stripping dots. ICD-10-CM is finer
than the WHO-style ICD-10 map, so we match EXACT first, then roll up (trim the
trailing char) until a phecode is found (standard hierarchical phecode behavior).

Outputs phecode_long.parquet = one row per (subject_id, phecode) with description,
group (organ-system category). Sense-organ (eye/ear) phecodes are kept in the table
but flagged so the landscape can exclude them as outcomes (eye is the FEATURE side).
"""
from config import CBCEYE_ROOT, MIMIC_HOSP
import os, re, pyreadr
import pandas as pd

HERE = CBCEYE_ROOT
M = MIMIC_HOSP
OUT = f"{HERE}/phecode_long.parquet"


def _norm(code):
    return re.sub(r"\.", "", str(code).strip().upper())


def load_maps():
    icd9 = pyreadr.read_r(f"{HERE}/phecode/phecode_map.rda")["phecode_map"]
    icd10 = pyreadr.read_r(f"{HERE}/phecode/phecode_map_icd10.rda")["phecode_map_icd10"]
    info = pyreadr.read_r(f"{HERE}/phecode/pheinfo.rda")["pheinfo"]
    # phecode_map.rda ships BOTH vocabularies (13,707 ICD9CM + 70,825 ICD10CM rows). Building the
    # ICD-9 dictionary from the whole table lets ICD-9 codes match ICD-10 entries during roll-up:
    # ICD-9 E849.7 (accident in residential institution) trims to E84 and returns PheCode 499
    # (cystic fibrosis), because E84 is the ICD-10 code for cystic fibrosis. Restrict each
    # dictionary to its own vocabulary. Legitimate roll-ups inside ICD9CM are unaffected, e.g.
    # V12.79 still resolves to 564.9.
    if "vocabulary_id" in icd9.columns:
        icd9 = icd9[icd9.vocabulary_id == "ICD9CM"]
    d9 = {_norm(c): p for c, p in zip(icd9.code, icd9.phecode)}
    d10 = {_norm(c): p for c, p in zip(icd10.code, icd10.phecode)}
    info["phecode"] = info["phecode"].astype(str)
    return d9, d10, info


def _lookup(code, d, min_len=3):
    c = _norm(code)
    while len(c) >= min_len:
        if c in d:
            return d[c]
        c = c[:-1]
    return None


def build(eye_cohort_only=True):
    d9, d10, info = load_maps()
    dx = pd.read_csv(f"{M}/diagnoses_icd.csv.gz",
                     usecols=["subject_id", "icd_code", "icd_version"], dtype=str)
    dx["code"] = dx.icd_code.str.strip()

    if eye_cohort_only:
        def is_eye(c, v):
            if v == "9":  return c[:3].isdigit() and 360 <= int(c[:3]) <= 379
            if v == "10": return len(c) >= 3 and c[0] == "H" and c[1:3].isdigit() and int(c[1:3]) <= 59
            return False
        eye = set(dx[[is_eye(c, v) for c, v in zip(dx.code, dx.icd_version)]].subject_id.unique())
        dx = dx[dx.subject_id.isin(eye)]
        print(f"restricted to eye cohort: {len(eye):,} patients")

    # map each unique (version, code) once
    uniq = dx[["icd_version", "code"]].drop_duplicates()
    ph = []
    for v, c in zip(uniq.icd_version, uniq.code):
        ph.append(_lookup(c, d9 if v == "9" else d10))
    uniq["phecode"] = ph
    mapped_frac = uniq.phecode.notna().mean()
    dx = dx.merge(uniq, on=["icd_version", "code"], how="left")
    row_cov = dx.phecode.notna().mean()

    long = (dx[dx.phecode.notna()][["subject_id", "phecode"]].drop_duplicates())
    long = long.merge(info[["phecode", "description", "group"]], on="phecode", how="left")
    long.to_parquet(OUT)

    print(f"unique codes mapped: {mapped_frac:.1%} | diagnosis rows covered: {row_cov:.1%}")
    print(f"patients with >=1 phecode: {long.subject_id.nunique():,}")
    print(f"distinct phecodes: {long.phecode.nunique():,}")
    print(f"distinct phecodes w/ group (excl sense organs): "
          f"{long[long.group!='sense organs'].phecode.nunique():,}")
    print("\ncases per phecode — modelable counts:")
    pc = long.groupby("phecode").subject_id.nunique()
    nonsense = set(long[long.group != "sense organs"].phecode)
    pcs = pc[pc.index.isin(nonsense)]
    for thr in [50, 100, 150, 200]:
        print(f"  systemic phecodes with >= {thr:>3} cases: {(pcs >= thr).sum()}")
    print(f"\nwrote {OUT}")
    return long


if __name__ == "__main__":
    build()
