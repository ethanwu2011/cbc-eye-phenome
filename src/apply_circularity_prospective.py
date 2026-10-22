"""Port the paper's circularity screen onto the PROSPECTIVE leg.

The concurrent landscape screens 639 -> 41 circular -> 559 clean, but the prospective
scripts never applied it. Consequence before the fix: the largest single effect in the
paper was primary open-angle glaucoma (+0.203, the eye predicting the eye), and 16 of the
47 "CBC-led" phenotypes were hematopoietic, i.e. the CBC predicting a CBC measurement.

Three pre-declared exclusion tiers, applied to BOTH arms symmetrically:
  (a) outcome IS a blood-count measurement          -> circular for the CBC arm
  (b) outcome IS an ocular condition                -> circular for the eye arm
  (c) outcome is definitionally implied by an exposure code (diabetic retinopathy -> 250.x)
      or is a care-process/status code rather than a disease
plus the landscape's own circularity_tier (tautological / expected_cbc / expected_manifestation).

The bootstrap statistics are unchanged; this only labels rows, so nothing is re-fit.
"""
from config import CBCEYE_ROOT
import pandas as pd, numpy as np, re
HERE=CBCEYE_ROOT

CBC_DEFINITIONAL=(r"an[ae]mia|neutropen|thrombocytopen|thrombocytos|leukocytos|leukopen|pancytopen|"
                  r"polycythem|white blood cell|red blood cell|platelet|coagulation|h[ae]moglobin|"
                  r"myelodysplas|myeloprolifer|eosinophil|basophil|monocytos|lymphocytos|lymphopen|"
                  r"sickle cell|hematocrit|blood count|agranulocytos|aplastic")
OCULAR=(r"glaucoma|conjunctiv|cataract|macular|retinopath|ophthalmic|blind|visual|eyelid|corneal|"
        r"uveitis|retinal|optic|keratitis|scleritis|strabismus|amblyopia|vitreous|lacrimal|iridocyclitis")
DM_IMPLIED=r"\bdiabet|insulin|antidiabetic"
CARE_PROCESS=r"encounter for|long-term \(current\) use|\bstatus\b|aftercare|screening for|history of|noncompliance"

def screen(df):
    d=df.description.astype(str)
    tier=df.get("circularity_tier", pd.Series(["untagged"]*len(df),index=df.index)).fillna("untagged")
    out=np.where(tier.isin(["tautological","expected_cbc","expected_manifestation"]),"landscape-tier:"+tier,
        np.where(d.str.contains(CBC_DEFINITIONAL,case=False,na=False),"CBC-definitional",
        np.where(d.str.contains(OCULAR,case=False,na=False),"ocular (eye->eye)",
        np.where(d.str.contains(DM_IMPLIED,case=False,na=False),"diabetes-implied (DR->250.x)",
        np.where(d.str.contains(CARE_PROCESS,case=False,na=False),"care-process","")))))
    return pd.Series(out,index=df.index)

lf=pd.read_parquet(f"{HERE}/landscape_full.parquet",columns=["phecode","circularity_tier"]).drop_duplicates("phecode")
lf["phecode"]=lf.phecode.astype(str)

for f in ["prospective_significance.csv","prospective_landscape.csv",
          "prospective_landscape_B90.csv","prospective_landscape_B730.csv"]:
    try: t=pd.read_csv(f"{HERE}/{f}",dtype={"phecode":str})
    except FileNotFoundError: print(f"  (skip {f})"); continue
    t=t.drop(columns=[c for c in ("circularity_tier","excluded_because","clean") if c in t.columns])
    t=t.merge(lf,on="phecode",how="left")
    t["excluded_because"]=screen(t); t["clean"]=t.excluded_because==""
    t.to_csv(f"{HERE}/{f}",index=False)
    print(f"  {f:<38} {int(t.clean.sum()):>4}/{len(t)} clean")

s=pd.read_csv(f"{HERE}/prospective_significance.csv",dtype={"phecode":str})
k=s[s.clean]
print("\n"+"="*86)
print("PROSPECTIVE CLASSES — circularity screen ported (bootstrap stats unchanged)")
print("="*86)
print(f"  {'class':<14}{'pre-screen':>12}{'post-screen':>13}")
for c,l in [("eye","eye-led"),("CBC","CBC-led")]:
    print(f"  {l:<14}{int((s.sig_lead==c).sum()):>12}{int((k.sig_lead==c).sum()):>13}")
print(f"  {'synergy':<14}{int(s.sig_syn.sum()):>12}{int(k.sig_syn.sum()):>13}")
print(f"  {'indeterminate':<14}{int((s.sig_lead=='ns').sum()):>12}{int((k.sig_lead=='ns').sum()):>13}")
print(f"\n  excluded ({len(s)-len(k)}):",s[~s.clean].excluded_because.value_counts().to_dict())
print("\n  post-screen eye-led (verbatim PheCode descriptions):")
for _,r in k[k.sig_lead=="eye"].sort_values("d_eye_cbc",ascending=False).iterrows():
    print(f"    {str(r.description)[:48]:<50} {r.d_eye_cbc:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]  ev={r.events}")
print("\n  post-screen CBC-led, top 10:")
for _,r in k[k.sig_lead=="CBC"].sort_values("d_eye_cbc").head(10).iterrows():
    print(f"    {str(r.description)[:48]:<50} {r.d_eye_cbc:+.3f}  ev={r.events}")
print("\n  post-screen synergy:")
for _,r in k[k.sig_syn].sort_values("d_syn",ascending=False).iterrows():
    print(f"    {str(r.description)[:48]:<50} eye {r.c_eye:.3f} cbc {r.c_cbc:.3f} both {r.c_both:.3f} (+{r.d_syn:.3f})")
