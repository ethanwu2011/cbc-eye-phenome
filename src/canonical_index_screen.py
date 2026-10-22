"""One authoritative screen for the index-anchored landscape, at every required lead.

The manuscript states that index-anchored outcomes were screened by "the same circularity
screen used for the concurrent landscape". Earlier exports did not do that. They merged
the concurrent tier by PheCode, so any phenotype absent from the concurrent landscape was
never screened, and they did not apply the concurrent eligibility rule. That let through
fifteen codes that are not PheCode v1.2 entries, one row with no PheCode at all, and seven
care-process codes the concurrent screen removes.

This applies, to every index-anchored phenotype, exactly what the concurrent landscape
applies (landscape_phewide.py and circularity.py):

  1. eligibility   the PheCode has a description and an organ group in the PheCode v1.2
                   information table, and the group is not "sense organs"
  2. tier()        the concurrent circularity function, run on that description, which
                   removes tautological, expected-CBC and care-process phenotypes
  3. manifestation phenotypes the concurrent ablation tagged expected_manifestation
  4. the prospective vocabulary screen already in the export (ocular and diabetes-implied
                   outcomes, CBC-definitional outcomes), which mirrors the concurrent
                   sense-organ exclusion and manifestation rule for codes the concurrent
                   ablation never saw

Writes prospective_landscape{tag}_canonical.csv for each lead, with a single `canonical`
flag and the reason any phenotype was dropped.
"""
from config import CBCEYE_ROOT
import pandas as pd
import pyreadr

import circularity as C

HERE = CBCEYE_ROOT
LEADS = [("_B90", 90), ("", 365), ("_B730", 730)]


def load_reference():
    info = pyreadr.read_r(f"{HERE}/phecode/pheinfo.rda")["pheinfo"]
    info["phecode"] = info.phecode.astype(str)
    info = info.drop_duplicates("phecode").set_index("phecode")
    lf = pd.read_parquet(f"{HERE}/landscape_full.parquet").drop_duplicates("phecode")
    lf["phecode"] = lf.phecode.astype(str)
    manifest = set(lf[lf.circularity_tier == "expected_manifestation"].phecode)
    return info, manifest


def screen(df, info, manifest):
    df = df.copy()
    df["phecode"] = df.phecode.astype(str)
    desc = df.phecode.map(info.description)
    grp = df.phecode.map(info.group)
    # the authoritative description and group come from the PheCode table, not the export
    df["description"] = desc
    df["group"] = grp
    reasons = []
    for p, d, g, prior in zip(df.phecode, desc, grp, df.get("clean", pd.Series(True, index=df.index))):
        if p in ("nan", "None", "") or pd.isna(d) or pd.isna(g):
            reasons.append("ineligible: not a PheCode v1.2 entry with a description and group")
        elif g == "sense organs":
            reasons.append("ineligible: sense-organ phenotype")
        elif (t := C.tier(d)) != "clean":
            reasons.append(f"concurrent tier: {t}")
        elif p in manifest:
            reasons.append("concurrent tier: expected_manifestation")
        elif not bool(prior):
            reasons.append("prospective vocabulary screen")
        else:
            reasons.append("")
    df["excluded_because"] = reasons
    df["canonical"] = df.excluded_because == ""
    return df


def main():
    info, manifest = load_reference()
    for tag, days in LEADS:
        src = pd.read_csv(f"{HERE}/prospective_landscape{tag}.csv", dtype={"phecode": str})
        out = screen(src, info, manifest)
        out.to_csv(f"{HERE}/prospective_landscape{tag}_canonical.csv", index=False)
        k = out[out.canonical]
        prev = int(src.clean.sum()) if "clean" in src.columns else len(src)
        print(f"\nlead {days:>3} d   {len(src)} tested   previously retained {prev}   now {len(k)}")
        print("   dropped:", out[~out.canonical].excluded_because.str.split(":").str[0]
              .value_counts().to_dict())
        newly = out[(~out.canonical) & src.get("clean", pd.Series(True, index=src.index)).values]
        if len(newly):
            print(f"   newly excluded vs the earlier export ({len(newly)}):",
                  newly.excluded_because.value_counts().to_dict())


if __name__ == "__main__":
    main()
