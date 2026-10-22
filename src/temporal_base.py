"""Temporal base table for the incident/early-detection analysis (MIMIC-IV).

T0 = admittime of each patient's FIRST eye-coded admission (the index 'eye visit').
All predictors are restricted to <= T0 (CBC allowed through T0+2d to capture index-
admission baseline labs; still far before the >=T0+90d outcome window, so no leakage).
Follow-up end = last admittime; death from patients.dod. Times stored as days vs T0.

Outputs:
  temporal_base.parquet  — per-subject: T0, demo, utilization, follow_days, death_days, CBC analytes
  eye_at_t0.parquet      — (subject_id, code) eye ICD codes observed <= T0
"""
from config import CBCEYE_ROOT, MIMIC_HOSP
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
M = MIMIC_HOSP; HERE = CBCEYE_ROOT

dx = pd.read_csv(f"{M}/diagnoses_icd.csv.gz", usecols=["subject_id","hadm_id","icd_code","icd_version"], dtype=str)
dx["code"] = dx.icd_code.str.strip()
def is_eye(c, v):
    if v == "9":  return c[:3].isdigit() and 360 <= int(c[:3]) <= 379
    if v == "10": return len(c) >= 3 and c[0] == "H" and c[1:3].isdigit() and int(c[1:3]) <= 59
    return False
dx["eye"] = [is_eye(c, v) for c, v in zip(dx.code, dx.icd_version)]

adm = pd.read_csv(f"{M}/admissions.csv.gz", usecols=["subject_id","hadm_id","admittime","race"], dtype=str)
adm["admittime"] = pd.to_datetime(adm.admittime)
htime = dict(zip(adm.hadm_id, adm.admittime))
dx["admit"] = dx.hadm_id.map(htime)
dx = dx.dropna(subset=["admit"])

# T0 = first admittime among eye-coded admissions
eye_rows = dx[dx.eye]
T0 = eye_rows.groupby("subject_id").admit.min().rename("T0")
print(f"eye-cohort patients with T0: {T0.size:,}")

dx = dx.merge(T0, left_on="subject_id", right_index=True, how="inner")
le = dx[dx.admit <= dx.T0]                      # everything at/before T0
eye_at_t0 = le[le.eye][["subject_id","code"]].drop_duplicates()
eye_at_t0.to_parquet(f"{HERE}/eye_at_t0.parquet")

util = le.groupby("subject_id").agg(n_codes_le_t0=("code","nunique"),
                                    n_adm_le_t0=("hadm_id","nunique"))
last_act = dx.groupby("subject_id").admit.max().rename("last_activity")

pt = pd.read_csv(f"{M}/patients.csv.gz", usecols=["subject_id","gender","anchor_age","dod"], dtype=str)
pt["anchor_age"] = pd.to_numeric(pt.anchor_age, errors="coerce")
pt["dod"] = pd.to_datetime(pt.dod, errors="coerce")
race = adm.groupby("subject_id").race.agg(lambda s: s.mode().iat[0] if len(s.mode()) else "UNK")

base = pd.DataFrame(T0).join(util).join(last_act)
base = base.join(pt.set_index("subject_id")[["gender","anchor_age","dod"]]).join(race)
base["follow_days"] = (base.last_activity - base.T0).dt.days
base["death_days"] = (base.dod - base.T0).dt.days
base = base.drop(columns=["last_activity","dod"])

# CBC most-recent with charttime <= T0 + 2 days
cbc = pd.read_parquet(f"{HERE}/cbc_long.parquet")
cbc["charttime"] = pd.to_datetime(cbc.charttime)
cbc = cbc.merge(T0, left_on="subject_id", right_index=True, how="inner")
cbc = cbc[cbc.charttime <= cbc.T0 + pd.Timedelta(days=2)].sort_values("charttime")
piv = cbc.groupby(["subject_id","itemid"]).valuenum.last().unstack()
lab = pd.read_csv(f"{M}/d_labitems.csv.gz", dtype=str); lbl = dict(zip(lab.itemid, lab.label))
covg = piv.notna().mean(); keep = sorted([c for c in piv.columns if covg[c] >= 0.40], key=lambda c: -covg[c])
piv = piv[keep]; piv.columns = ["CBC_"+str(lbl.get(c, c)) for c in keep]
piv = piv.loc[:, ~pd.Index(piv.columns).duplicated()]
base = base.join(piv)
base.to_parquet(f"{HERE}/temporal_base.parquet")

print(f"base rows: {len(base):,} | CBC<=T0 coverage: {base.filter(like='CBC_').notna().any(axis=1).mean():.1%} "
      f"| CBC analytes: {piv.shape[1]}")
print("follow_days: " + " ".join(f"{q}p={base.follow_days.quantile(q/100):.0f}" for q in [25,50,75,90]))
print(f"reach >=365d follow-up: {(base.follow_days>=365).mean():.1%} | >=1095d: {(base.follow_days>=1095).mean():.1%}")

# incident feasibility for a few eligible diseases
ELIG = {"Diabetes":(["250"],["E08","E09","E10","E11","E13"]),
        "CKD":(["585"],["N18"]), "Heart failure":(["428"],["I50"]),
        "Hypertension":(["401"],["I10"]), "IHD":(["410","411","412","414"],["I20","I21","I22","I24","I25"]),
        "Periph neuropathy":(["356","3572"],["G60","G62","G63"]),
        "Hyperlipidemia":(["2720","2721","2722","2724"],["E780","E781","E782","E784","E785"])}
def onset(d9,d10):
    m = dx[((dx.icd_version=="9")&(dx.code.str.startswith(tuple(d9)))) |
           ((dx.icd_version=="10")&(dx.code.str.startswith(tuple(d10))))]
    return m.groupby("subject_id").admit.min()
print("\nincident feasibility (blanking 90d, horizon 1y, need follow-up):")
print(f"{'disease':<20}{'prevalent':>10}{'incident<=1y':>13}{'at-risk':>9}")
for name,(d9,d10) in ELIG.items():
    on = onset(d9,d10).rename("onset")
    b = base.join(on)
    days = (b.onset - b.T0).dt.days
    prevalent = (days <= 90).sum()
    atrisk = b[(days.isna()) | (days > 90)]
    a_days = (atrisk.onset - atrisk.T0).dt.days
    incident_1y = ((a_days > 90) & (a_days <= 365)).sum()
    print(f"{name:<20}{prevalent:>10}{incident_1y:>13}{len(atrisk):>9}")
print("\nwrote temporal_base.parquet, eye_at_t0.parquet")
