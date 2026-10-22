"""Why did the neurological axis not reproduce prospectively?

Two competing explanations, and they are distinguishable:
  (A) POWER / PREVALENCE — the classic eye-neuro phenotypes (stroke, dementia, MS, Parkinson,
      neuropathy) never reached the incident-case floor, because in an eye-coded cohort most
      of these patients ALREADY have the diagnosis at T0. Then "neuro did not replicate"
      really means "neuro was never tested".
  (B) MANIFESTATION — they were tested and genuinely failed, which is what the sequela
      ablation predicts: the cross-sectional neuro signal rides on post-event ocular codes
      (hemianopia, field defects, optic neuritis) that by construction cannot exist BEFORE
      the diagnosis. Prospectively there is nothing left to carry it.

This prints, per phenotype and per cell: eligible / prevalent-at-T0 / blanked / incident,
and whether it cleared the floor.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

HERE=CBCEYE_ROOT; MIN_CASES=60
base=pd.read_parquet(f"{HERE}/temporal_base.parquet")
onset=pd.read_parquet(f"{HERE}/phecode_onset.parquet")
meta=pd.read_parquet(f"{HERE}/phecode_long.parquet",columns=["phecode","description","group"]).drop_duplicates("phecode")
meta["phecode"]=meta.phecode.astype(str); onset["phecode"]=onset.phecode.astype(str)
d=dict(zip(meta.phecode,meta.description))

PAT=r"stroke|cerebral (infarct|ischemi)|cerebrovascular|transient cerebral|dementia|alzheimer|" \
    r"multiple sclerosis|demyelin|parkinson|epilep|convuls|neuropathy|polyneuropathy|migraine|" \
    r"cognitive|memory|encephalopathy"
neuro=meta[meta.description.astype(str).str.contains(PAT,case=False,na=False)].copy()
print(f"candidate eye-relevant neuro phenotypes: {len(neuro)}")

CELLS=[(90,365,"≥3mo·1yr"),(365,730,"≥1yr·2yr"),(365,1095,"≥1yr·3yr"),(730,1825,"≥2yr·5yr")]
og={k:dict(zip(v.subject_id,v.onset_days)) for k,v in onset.groupby("phecode")}
rows=[]
for _,r in neuro.iterrows():
    p=r.phecode; o=og.get(p)
    if not o: continue
    ever=len(o)
    if ever<40: continue
    prev0=sum(1 for v in o.values() if v<=0)          # already coded at/before the index visit
    rec=dict(phecode=p,description=str(r.description)[:44],group=r.group,ever=ever,
             prevalent_at_T0=prev0,pct_prevalent=round(100*prev0/ever))
    for B,H,lab in CELLS:
        elig=base[base.follow_days>=H]
        subs=[s for s in elig.index if o.get(s,np.inf)>B]
        inc=sum(1 for s in subs if B<o.get(s,np.inf)<=H)
        rec[f"inc_{lab}"]=inc
        rec[f"ok_{lab}"]="Y" if (inc>=MIN_CASES and len(subs)-inc>=MIN_CASES and len(subs)>=200) else "-"
    rows.append(rec)
t=pd.DataFrame(rows).sort_values("ever",ascending=False)
t.to_csv(f"{HERE}/neuro_why_not.csv",index=False)

print("\n"+"="*112)
print(f"{'phenotype':<46}{'ever':>6}{'prevAtT0':>9}{'%prev':>7}   " + "".join(f"{l:>11}" for _,_,l in CELLS))
print("="*112)
for _,r in t.head(24).iterrows():
    cells="".join(f"{str(r[f'inc_{l}'])+r[f'ok_{l}']:>11}" for _,_,l in CELLS)
    print(f"{r.description:<46}{r.ever:>6}{r.prevalent_at_T0:>9}{r.pct_prevalent:>6}%   {cells}")
print("\n  ('Y' = cleared the >=60 incident-case floor and was actually modelled; '-' = never tested)")

nt=t[[f"ok_{l}" for _,_,l in CELLS]].apply(lambda c:(c=="Y")).sum()
print(f"\n  neuro phenotypes actually TESTED per cell: {dict(zip([l for _,_,l in CELLS],nt.tolist()))}")
print(f"  median % already prevalent at T0 across these phenotypes: {t.pct_prevalent.median():.0f}%")
print("\nsaved neuro_why_not.csv")
