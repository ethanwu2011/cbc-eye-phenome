"""Per-(patient, phecode) onset time = earliest admission carrying that phecode,
in days relative to the patient's T0. Enables incident/survival analysis for ANY
phecode (generalizes the hardcoded-ICD temporal analysis)."""
from config import CBCEYE_ROOT, MIMIC_HOSP
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import phecode_map as pm

M = MIMIC_HOSP; HERE = CBCEYE_ROOT
d9, d10, _ = pm.load_maps()
base = pd.read_parquet(f"{HERE}/temporal_base.parquet")[["T0"]]

dx = pd.read_csv(f"{M}/diagnoses_icd.csv.gz", usecols=["subject_id","hadm_id","icd_code","icd_version"], dtype=str)
dx = dx[dx.subject_id.isin(base.index)].copy()
dx["code"] = dx.icd_code.str.strip()
uniq = dx[["icd_version","code"]].drop_duplicates()
uniq["phecode"] = [pm._lookup(c, d9 if v=="9" else d10) for v,c in zip(uniq.icd_version, uniq.code)]
dx = dx.merge(uniq, on=["icd_version","code"], how="left").dropna(subset=["phecode"])

adm = pd.read_csv(f"{M}/admissions.csv.gz", usecols=["hadm_id","admittime"], dtype=str)
dx["admit"] = pd.to_datetime(dx.hadm_id.map(dict(zip(adm.hadm_id, pd.to_datetime(adm.admittime)))))
dx = dx.dropna(subset=["admit"]).merge(base, left_on="subject_id", right_index=True, how="inner")
dx["onset_days"] = (dx.admit - dx.T0).dt.days

onset = dx.groupby(["subject_id","phecode"]).onset_days.min().reset_index()
onset.to_parquet(f"{HERE}/phecode_onset.parquet")
print(f"phecode_onset rows: {len(onset):,} | patients: {onset.subject_id.nunique():,} | phecodes: {onset.phecode.nunique():,}")
