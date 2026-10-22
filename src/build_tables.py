"""Build manuscript Tables T1 (cohort+design), T2 (representative diseases by class), and a
T3 generator (per-class MIMIC<->AoU concordance, fills after the AoU run). Aggregates only (DUA).
Writes markdown + csv into final_manuscript/."""
from config import CBCEYE_ROOT
import os, re, numpy as np, pandas as pd
HERE=CBCEYE_ROOT; OUT=f"{HERE}/final_manuscript"; os.makedirs(OUT,exist_ok=True)
L=pd.read_parquet(f"{HERE}/landscape_full.parquet")
clean=L[(L.circularity_tier=="clean")&(L.klass!="no signal")].copy()

# ---------- driver cleaning ----------
CLEAN={"Background diabetic re":"background diabetic retinopathy","Proliferative diabetic":"proliferative retinopathy",
 "Dry eye syndrome of un":"dry-eye syndrome","Dry eye syndrome of bi":"dry-eye syndrome","Unspecified macular de":"macular degeneration",
 "Conjunctivitis, unspec":"conjunctivitis","Unspecified conjunctiv":"conjunctivitis","Visual field":"visual-field defect",
 "homonymous":"homonymous field defect","heteronymous":"heteronymous field defect","Diplopia":"diplopia","Nystagmus":"nystagmus",
 "Ptosis":"ptosis","Optic":"optic-nerve sign","Retinal vascular":"retinal vascular occlusion",
 "Other visual":"visual disturbance","Anisocoria":"anisocoria","Visual disturban":"visual disturbance"}
def first_driver(s,n=1):
    if not isinstance(s,str) or not s.strip(): return "—"
    parts=[p.split("(")[0].strip() for p in s.split(";")][:n]
    out=[]
    for p in parts:
        rep=next((v for k,v in CLEAN.items() if p.lower().startswith(k.lower())), p)
        out.append(rep)
    return "; ".join(out)

# ========================= TABLE 1 =========================
tb=pd.read_parquet(f"{HERE}/temporal_base.parquet")
N=len(tb)
def med_iqr(s): s=s.dropna(); return f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"
age=med_iqr(tb.anchor_age)
pct_f=100*(tb.gender.astype(str).str.upper().str[0]=="F").mean()
def racegrp(r):
    r=str(r).upper()
    if "WHITE" in r: return "White"
    if "BLACK" in r or "AFRICAN" in r: return "Black"
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic/Latino"
    if "ASIAN" in r: return "Asian"
    return "Other/Unknown"
race=(tb.race.map(racegrp).value_counts(normalize=True)*100).round(1)
ncodes=med_iqr(tb.n_codes_le_t0); nadm=med_iqr(tb.n_adm_le_t0); fup=med_iqr(tb.follow_days)
pct_death=100*tb.death_days.notna().mean()

t1=[]
t1.append("## Table 1 — Cohort characteristics and study design\n")
t1.append("| Characteristic | Value |")
t1.append("|---|---|")
t1.append(f"| Patients (≥1 ophthalmic ICD code) | {N:,} |")
t1.append(f"| Age at index, years — median [IQR] | {age} |")
t1.append(f"| Female sex, % | {pct_f:.1f} |")
for k,v in race.items(): t1.append(f"| Race — {k}, % | {v} |")
t1.append(f"| Distinct ICD codes by index — median [IQR] | {ncodes} |")
t1.append(f"| Hospital admissions by index — median [IQR] | {nadm} |")
t1.append(f"| Follow-up after index, days — median [IQR] | {fup} |")
t1.append(f"| Death recorded during follow-up, % | {pct_death:.1f} |")
t1.append("| **Features** | |")
t1.append("| Ophthalmic ICD codes (binary) | 187 |")
t1.append("| Complete-blood-count analytes (continuous) | 20 |")
t1.append("| PheCode v1.2 diagnosis-row coverage, % | 87 |")
t1.append("| **Systemic phenotypes** | |")
t1.append(f"| Modelled (≥150 cases) | {len(L)} |")
t1.append("| Circular — tautological / expected-CBC / expected-manifestation | 41 (13 / 5 / 23) |")
t1.append(f"| Clean | {(L.circularity_tier=='clean').sum()} |")
t1.append(f"| Clean with discriminative signal (analysis set) | {len(clean)} |")
t1.append("| **Taxonomy of the 559** | |")
for k in ["CBC-specific","comparable","synergy","eye-specific"]:
    t1.append(f"| &nbsp;&nbsp;{k} | {(clean.klass==k).sum()} |")
open(f"{OUT}/Table1.md","w").write("\n".join(t1)+"\n")

# ========================= TABLE 2 =========================
md=pd.read_csv(f"{HERE}/modality_drivers.csv")
m=clean.merge(md[["name","top_eye_drivers","top_cbc_drivers"]],left_on="description",right_on="name",how="left")
def ci(r,k): return f"{r['auc_'+k]:.2f} ({r['ci_'+k+'_lo']:.2f}–{r['ci_'+k+'_hi']:.2f})"
# curated recognisable exemplars, VERIFIED to belong to each class (no cross-class fallback)
PICKS={
 "eye-specific":["Multiple sclerosis","Peripheral autonomic neuropathy","Cerebral artery occlusion, with cerebral infarction","Gastroparesis"],
 "synergy":["Nephritis and nephropathy in diseases","Chronic Kidney Disease, Stage III","Insulin pump user","Congestive heart failure (CHF) NOS"],
 "comparable":["Epilepsy, recurrent seizures","Malaise and fatigue"],
 "CBC-specific":["Atrial fibrillation and flutter","Gram negative septicemia","Cirrhosis of liver without","Respiratory failure"]}
SHORT={"Cerebral artery occlusion, with cerebral infarction":"Cerebral artery occlusion (ischaemic stroke)",
 "Nephritis and nephropathy in diseases classified elsewhere":"Nephropathy in systemic disease",
 "Congestive heart failure (CHF) NOS":"Congestive heart failure","Hyposmolality and/or hyponatremia":"Hyponatraemia",
 "Cirrhosis of liver without mention of alcohol":"Cirrhosis of liver","Epilepsy, recurrent seizures, convulsions":"Epilepsy",
 "Atrial fibrillation and flutter":"Atrial fibrillation / flutter","Chronic Kidney Disease, Stage III":"Chronic kidney disease, stage III"}
DSIG=0.58   # show a modality's driver only if that modality actually discriminates (AUROC>=DSIG)
rows=[]
for klass,names in PICKS.items():
    for nm in names:
        sub=m[(m.klass==klass)&(m.description.str.contains(nm,case=False,regex=False))]
        if not len(sub): print(f"  [warn] no {klass} match for '{nm}'"); continue
        r=sub.iloc[0]
        rows.append(dict(Disease=SHORT.get(r.description, r.description), Class=r.klass,
            Organ=str(r.group).replace(" system","").replace(" diseases",""),
            N=int(r.n_cases),Eye=ci(r,"eye"),CBC=ci(r,"cbc"),Both=ci(r,"both"),
            TopEye=(first_driver(r.top_eye_drivers) if r.auc_eye>=DSIG else "—"),
            TopCBC=(first_driver(r.top_cbc_drivers) if r.auc_cbc>=DSIG else "—")))
T2=pd.DataFrame(rows).drop_duplicates("Disease")
T2.to_csv(f"{OUT}/Table2.csv",index=False)
t2=["## Table 2 — Representative systemic phenotypes by modality class\n",
    "AUROC (95% CI, DeLong). Drivers = dominant ophthalmic code / CBC analyte (top signed-weight feature); "
    "shown only where that modality discriminates (AUROC≥0.58) and the disease is in the n≥250 driver atlas, else —.\n",
    "| Disease | Class | Organ | N | AUROC eye | AUROC CBC | AUROC eye+CBC | Top eye driver | Top CBC driver |",
    "|---|---|---|--:|---|---|---|---|---|"]
for _,r in T2.iterrows():
    t2.append(f"| {r.Disease} | {r.Class} | {r.Organ} | {r.N:,} | {r.Eye} | {r.CBC} | {r.Both} | {r.TopEye} | {r.TopCBC} |")
open(f"{OUT}/Table2.md","w").write("\n".join(t2)+"\n")

# ========================= TABLE 3 (generator + skeleton) =========================
def build_t3(merged_csv=f"{HERE}/aou_replication/aou_vs_mimic_merged.csv"):
    from scipy import stats
    cmp=pd.read_csv(merged_csv)
    out=["## Table 3 — Cross-cohort concordance, MIMIC-IV vs All of Us\n",
         "| Class | n | mean AUROC eye+CBC (MIMIC) | mean AUROC eye+CBC (AoU) | Spearman ρ (both) | replication rate* |",
         "|---|--:|--:|--:|--:|--:|"]
    for k in ["eye-specific","synergy","comparable","CBC-specific","ALL"]:
        sub=cmp if k=="ALL" else cmp[cmp.klass==k]
        if len(sub)<3: continue
        rho=stats.spearmanr(sub.auc_both,sub.auc_both_aou).correlation
        rep=((sub.auc_both_aou>0.55)&(sub.auc_both>0.55)).mean()  # same-direction signal
        out.append(f"| {k} | {len(sub)} | {sub.auc_both.mean():.3f} | {sub.auc_both_aou.mean():.3f} | {rho:.2f} | {rep:.0%} |")
    out.append("\n*replication rate = fraction with AUROC>0.55 in both cohorts.")
    open(f"{OUT}/Table3.md","w").write("\n".join(out)+"\n"); print("built Table3.md from",merged_csv)

if os.path.exists(f"{HERE}/aou_replication/aou_vs_mimic_merged.csv"):
    build_t3()
else:
    # skeleton with MIMIC side pre-filled
    sk=["## Table 3 — Cross-cohort concordance, MIMIC-IV vs All of Us  *(AoU columns pending the Workbench run)*\n",
        "| Class | n | mean AUROC eye+CBC (MIMIC) | mean AUROC eye+CBC (AoU) | Spearman ρ (both) | replication rate |",
        "|---|--:|--:|--:|--:|--:|"]
    for k in ["eye-specific","synergy","comparable","CBC-specific"]:
        sub=clean[clean.klass==k]; sk.append(f"| {k} | {len(sub)} | {sub.auc_both.mean():.3f} | _pending_ | _pending_ | _pending_ |")
    sk.append(f"| ALL | {len(clean)} | {clean.auc_both.mean():.3f} | _pending_ | _pending_ | _pending_ |")
    sk.append("\n_Preliminary eye-only name-matched check (not this table): organ-level Pearson r=0.62; disease-level Spearman ρ≈0.24._")
    sk.append("_Run `build_tables.py` after `aou_vs_mimic_merged.csv` lands to fill the AoU columns._")
    open(f"{OUT}/Table3.md","w").write("\n".join(sk)+"\n")
    print("wrote Table3.md SKELETON (AoU pending)")

print("\nwrote Table1.md, Table2.md/.csv, Table3.md ->",OUT)
print(f"\nT1 quick check: N={N:,}  age {age}  %F {pct_f:.1f}  death {pct_death:.1f}%")
print("T2 rows:",len(T2))
