"""PROSPECTIVE organ partition — reproducing the cross-sectional taxonomy in TIME.

Fix vs the first version: aggregate over ALL modelled true-screen phenotypes, not only the
FDR-significant ones. The partition is about WHERE each modality carries signal, so filtering
to significance first discards most organ systems (that is what hid neurological, hematopoietic,
respiratory, dermatologic, musculoskeletal and neoplasms).

Applies the SAME 4-way taxonomy as the cross-sectional landscape (mimic_landscape.py):
    synergy       both > max(eye,cbc) + 0.015
    eye-specific  eye > cbc + 0.03
    CBC-specific  cbc > eye + 0.03
    comparable    otherwise
and reports it per organ system, prospectively, at each required lead time.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, re

HERE=CBCEYE_ROOT
lt=pd.read_csv(f"{HERE}/temporal_leadtime.csv",dtype={"phecode":str})
if "description" in lt.columns: lt=lt.drop(columns=["description"])
meta=pd.read_parquet(f"{HERE}/phecode_long.parquet",columns=["phecode","description","group"]).drop_duplicates("phecode")
meta["phecode"]=meta.phecode.astype(str)
lt=lt.merge(meta,on="phecode",how="left")
lf=pd.read_parquet(f"{HERE}/landscape_full.parquet",columns=["phecode","circularity_tier"]).drop_duplicates("phecode")
lf["phecode"]=lf.phecode.astype(str)
lt=lt.merge(lf,on="phecode",how="left"); lt["circularity_tier"]=lt.circularity_tier.fillna("untagged")

NON_DISEASE_GROUPS={"symptoms","injuries & poisonings","congenital anomalies",
                    "pregnancy complications","sense organs"}
CARE_PROCESS=r"encounter|long-term \(current\) use|pump user|\bstatus\b|aftercare|screening for|history of|noncompliance|dialysis|transplant"
CONGENITAL  =r"congenital|hereditary|cystic fibrosis|sickle cell|chromosom|down syndrome"
ACUTE       =r"\bsepsis\b|septicemia|hypotension|\bshock\b|cardiac arrest|respiratory failure|acidosis|alkalosis|hypovolemia|\bdelirium\b|poisoning|\binjury\b|fracture"
SYMPTOM     =r"nonspecific|\bpain\b|nausea|vomiting|\bfever\b|malaise|fatigue|abnormal findings|other symptoms|functional digestive"
CBC_DEFINED =r"anemia|anaemia|neutropen|thrombocytopen|leukocytos|leukopen|pancytopen|polycythem|white blood cell|platelet|coagulation|hemoglobin"
def verdict(r):
    if r.circularity_tier!="clean":                 return f"circular:{r.circularity_tier}"
    d=str(r.description).lower()
    if str(r.group).lower() in NON_DISEASE_GROUPS:  return "non-disease-group"
    if re.search(CBC_DEFINED,d):  return "CBC-definitional"
    if re.search(CARE_PROCESS,d): return "care-process"
    if re.search(CONGENITAL,d):   return "congenital"
    if re.search(ACUTE,d):        return "acute-event"
    if re.search(SYMPTOM,d):      return "symptom/nonspecific"
    return ""
lt["excluded_because"]=lt.apply(verdict,axis=1)
lt["true_screen"]=lt.excluded_because==""
lt["eye_gain"]=lt.eye-lt.base; lt["cbc_gain"]=lt.cbc-lt.base
def tax(r):
    if r.both>max(r.eye,r.cbc)+0.015: return "synergy"
    if r.eye>r.cbc+0.03:              return "eye-specific"
    if r.cbc>r.eye+0.03:              return "CBC-specific"
    return "comparable"
lt["klass"]=lt.apply(tax,axis=1)
ts=lt[lt.true_screen].copy()
ts.to_csv(f"{HERE}/temporal_leadtime_organ.csv",index=False)

print("="*94)
print(f"PROSPECTIVE TAXONOMY — ALL {len(ts)} modelled true-screen phenotype-cells (not just FDR-significant)")
print("="*94)
print("  overall:",ts.klass.value_counts().to_dict())
print(f"  SYNERGY fraction: {100*(ts.klass=='synergy').mean():.0f}%   |  eye-specific {100*(ts.klass=='eye-specific').mean():.0f}%   CBC-specific {100*(ts.klass=='CBC-specific').mean():.0f}%")

print("\n"+"="*94); print("BY ORGAN SYSTEM — every organ with >=3 modelled true screens (pooled across lead times)"); print("="*94)
print(f"  {'organ system':<26}{'n':>4}{'eye':>9}{'CBC':>9}{'eye-CBC':>10}{'synergy':>9}{'eyeSpec':>9}{'cbcSpec':>9}   leads")
rows=[]
for grp,g in ts.groupby("group"):
    if len(g)<3: continue
    e,c=g.eye_gain.median(),g.cbc_gain.median()
    syn=100*(g.klass=="synergy").mean(); es=100*(g.klass=="eye-specific").mean(); cs=100*(g.klass=="CBC-specific").mean()
    lead="EYE" if e>c+0.01 else ("CBC" if c>e+0.01 else "comparable")
    rows.append(dict(group=grp,n=len(g),eye=e,cbc=c,diff=e-c,synergy_pct=syn,eyespec_pct=es,cbcspec_pct=cs,leads=lead))
for r in sorted(rows,key=lambda x:-x["diff"]):
    print(f"  {r['group']:<26}{r['n']:>4}{r['eye']:>+9.3f}{r['cbc']:>+9.3f}{r['diff']:>+10.3f}{r['synergy_pct']:>8.0f}%{r['eyespec_pct']:>8.0f}%{r['cbcspec_pct']:>8.0f}%   {r['leads']}")
pd.DataFrame(rows).to_csv(f"{HERE}/leadtime_organ_partition.csv",index=False)

print("\n"+"="*94); print("SYNERGY — where BOTH modalities are needed prospectively (top true screens)"); print("="*94)
syn=ts[ts.klass=="synergy"].sort_values("gain",ascending=False)
print(f"  {len(syn)} synergy phenotype-cells; FDR-significant: {((syn.q<0.05)&(syn.gain>0)).sum()}")
for _,r in syn[(syn.q<0.05)].head(12).iterrows():
    print(f"  {str(r.description)[:40]:<42}{str(r.group)[:17]:<19}{r.cell:<20} eye {r.eye:.3f} cbc {r.cbc:.3f} -> both {r.both:.3f} (Δ{r.gain:+.3f})")

print("\n"+"="*94); print("NEUROLOGICAL specifically (all modelled true screens, any significance)"); print("="*94)
n=ts[ts.group=="neurological"].sort_values("gain",ascending=False)
if len(n)==0: print("  none modelled")
for _,r in n.head(12).iterrows():
    print(f"  {str(r.description)[:40]:<42}{r.cell:<20} eye {r.eye:.3f} cbc {r.cbc:.3f} both {r.both:.3f} ({r.klass}, q={r.q:.2g})")
print("\nsaved temporal_leadtime_organ.csv + leadtime_organ_partition.csv")
