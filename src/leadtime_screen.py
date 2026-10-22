"""Apply the paper's circularity screen + a screening-target screen to the lead-time results.

Two filters, both necessary before any "we detect disease N years early" claim:
 (1) CIRCULARITY  — reuse landscape_full.parquet's circularity_tier (tautological /
     expected_cbc / expected_manifestation), the same screen used for the landscape.
 (2) SCREENING-TARGET — a lead-time hit is only meaningful if the phenotype is a chronic
     disease you would plausibly screen for. Excludes: congenital conditions (cannot be
     incident in an adult; e.g. cystic fibrosis), care-process / treatment codes
     ("insulin pump user", "long-term use of anticoagulants"), and acute events
     (sepsis, hypotension) whose "onset" is an admission event, not a detectable disease.
"""
from config import CBCEYE_ROOT
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, re

HERE=CBCEYE_ROOT
lt=pd.read_csv(f"{HERE}/temporal_leadtime.csv",dtype={"phecode":str})   # dtype str: '499' must not become '499.0'
if "description" in lt.columns: lt=lt.drop(columns=["description"])
desc=pd.read_parquet(f"{HERE}/phecode_long.parquet",columns=["phecode","description"]).drop_duplicates("phecode")
desc["phecode"]=desc.phecode.astype(str)
lt=lt.merge(desc,on="phecode",how="left")
print(f"description matched for {lt.description.notna().sum()}/{len(lt)} rows")

# ---- (1) circularity tier from the landscape ----
lf=pd.read_parquet(f"{HERE}/landscape_full.parquet",columns=["phecode","circularity_tier"]).drop_duplicates("phecode")
lf["phecode"]=lf.phecode.astype(str)
lt=lt.merge(lf,on="phecode",how="left")
lt["circularity_tier"]=lt.circularity_tier.fillna("untagged")

# ---- (2) screening-target screen ----
CARE_PROCESS=r"encounter|long-term \(current\) use|pump user|status|aftercare|screening for|history of|noncompliance|dialysis"
CONGENITAL=r"cystic fibrosis|congenital|hereditary|sickle cell|down syndrome|chromosom"
ACUTE=r"\bsepsis\b|septicemia|hypotension|shock|cardiac arrest|respiratory failure|electrolyte|acidosis|alkalosis|hypovolemia|delirium"
def not_screenable(d):
    d=str(d).lower()
    if re.search(CARE_PROCESS,d): return "care-process"
    if re.search(CONGENITAL,d):   return "congenital"
    if re.search(ACUTE,d):        return "acute-event"
    return ""
lt["nonscreen"]=lt.description.map(not_screenable)
lt["excluded_because"]=np.where(lt.circularity_tier!="clean","circular:"+lt.circularity_tier,lt.nonscreen)
lt["kept"]=lt.excluded_because==""

lt.to_csv(f"{HERE}/temporal_leadtime_screened.csv",index=False)

print("="*84)
print("LEAD-TIME RESULTS AFTER CIRCULARITY + SCREENING-TARGET SCREENS")
print("="*84)
for lab,g in lt.groupby("cell",sort=False):
    sig=g[(g.q<0.05)&(g.gain>0)]
    sig_k=sig[sig.kept]
    print(f"\n### {lab}   (n={g.n.max():,} eligible)")
    print(f"  modelled {len(g)} | FDR-sig {len(sig)} | **FDR-sig AND clean+screenable: {len(sig_k)}**")
    if len(sig)>len(sig_k):
        drop=sig[~sig.kept].excluded_because.value_counts().to_dict()
        print(f"  dropped by screens: {drop}")
    for _,r in sig_k.sort_values("gain",ascending=False).head(10).iterrows():
        print(f"    {str(r.description)[:46]:<48} cases={int(r.cases):>4}  base {r.base:.3f} -> both {r.both:.3f}  (Δ{r.gain:+.3f}, q={r.q:.2g})")
    if len(sig_k)==0: print("    (nothing survives both screens in this cell)")

print("\n"+"="*84)
print("HEADLINE — surviving lead time (clean, screenable, FDR<0.05):")
for lab,g in lt.groupby("cell",sort=False):
    s=g[(g.q<0.05)&(g.gain>0)&g.kept]
    med=f"{s.gain.median():+.3f}" if len(s) else "—"
    print(f"  {lab:<24} {len(s):>3} phenotypes | median gain {med}")
print("\nsaved temporal_leadtime_screened.csv")
