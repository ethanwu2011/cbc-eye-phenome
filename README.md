# Ophthalmic and hematologic data carry organ-partitioned, non-redundant information about systemic disease

Analysis code and aggregate results for a phenome-wide comparison of routine ophthalmic
diagnosis codes against the complete blood count (CBC) as competing predictors of systemic
disease, in MIMIC-IV and in NHANES.

## What is here

```
src/        the analysis pipeline, in dependency order below
results/    aggregate per-phenotype result files, one row per phenotype or condition
figures/    output directory, populated by the figure scripts
```

## What is not here, and why

No patient-level data. Every file in `results/` is aggregated to the phenotype or
condition level: a PheCode or condition name, case counts, AUROCs or C-indices with
confidence intervals, and class labels. There are no subject identifiers, no admission
identifiers and no timestamps anywhere in this repository.

The intermediate tables that the pipeline builds (`temporal_base.parquet`,
`eye_at_t0.parquet`, `phecode_long.parquet`, `phecode_onset.parquet`, `cbc_raw.parquet`)
are patient level. They are deliberately not redistributed, and `.gitignore` blocks them.
Regenerate them from your own credentialed copy of MIMIC-IV.

## Data access

**MIMIC-IV v3.1** requires CITI training and a signed data use agreement through PhysioNet:
https://physionet.org/content/mimiciv/ . The pipeline reads the `hosp` module.

**PheCode v1.2 maps** (`phecode_map.rda`, `phecode_map_icd10.rda`, `pheinfo.rda`) are public from
the PheWAS Catalog, https://phewascatalog.org/phecodes . Place them in `phecode/` under the
repository root. They are needed by `phecode_map.py` and `canonical_index_screen.py`.

**NHANES** cycles 2005-2006 and 2007-2008 are public:
https://www.cdc.gov/nchs/nhanes/ . Retinal grading comes from the ophthalmology
examination files for those two cycles.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MIMIC_HOSP=/path/to/mimiciv-3.1/hosp     # required to rebuild from raw data
export NHANES_DIR=/path/to/nhanes               # required for the NHANES replication
```

`src/config.py` resolves the repository root automatically. Scripts are run from the
repository root, for example `python src/landscape_phewide.py`.

The figure panels and the territory permutation test can be regenerated from the aggregate
results alone, without MIMIC-IV. Stage the results into a working root first:

```bash
python src/stage_results.py work
export CBCEYE_ROOT="$PWD/work"
python src/build_fig5_territory.py
```

`figures/README.md` lists the script behind each panel. Table 1 and Figure 2e are computed
from the patient-level intermediate tables.

## Pipeline

Build the intermediate tables, then run any stage independently.

| Stage | Scripts |
|---|---|
| Phenotyping and base tables | `phecode_map.py`, `temporal_base.py`, `build_phecode_onset.py` |
| Circularity screen | `circularity.py`, `apply_circularity_prospective.py` |
| AUROC inference | `stats_utils.py` (fast DeLong covariance, used throughout) |
| Concurrent landscape | `landscape_phewide.py`, `mimic_landscape.py`, `make_figures.py` |
| Complementarity tests | `mimic_orthogonality.py`, `eye_predicts_cbc.py`, `orthogonality_interaction_test.py`, `interaction_fair.py` |
| Index-anchored landscape | `prospective_landscape.py`, `canonical_index_screen.py`, `foldwise_sensitivity.py`, `prospective_significance.py`, `leadtime_organ_partition.py`, `temporal_leadtime.py`, `leadtime_screen.py`, `leadtime_survival.py` |
| Confound controls | `mimic_diabetes_strat.py`, `diabetes_strat_delong.py`, `mimic_neuro_ablation.py` |
| Neurological and microvascular axis | `neuro_axis.py`, `neuro_why_not.py`, `micro_vs_macro.py`, `territory_permutation.py` |
| NHANES replication | `nhanes_validation.py`, `nhanes_diabetes_utility.py`, `utility_rebuild.py` |
| Figures and tables | `stage_results.py`, `build_plotly_figs.py`, `build_fig2d_flowmap.py`, `compose_figures.py`, `build_fig3_final.py`, `build_fig5_territory.py`, `build_fig_triangulation.py`, `build_fig_eye_gain.py`, `build_nhanes_figure.py`, `build_landscape_plotly.py`, `build_fig1b_organ.py`, `build_tables.py` |

`cohort_build.py` assembles the matched case-control design and is imported by the
orthogonality and interaction tests.

## The analysis set

`results/landscape_full.csv` holds all 635 screened phenotypes. **The 528-phenotype
analysis set reported in the paper is the subset with `circularity_tier == "clean"` and
`klass != "no signal"`.** Filtering on `circularity_tier` alone gives 574 rows and
different numbers. The CBC-definitional lexicon in `circularity.py` covers plural and
plain-language forms (deficiency anemias, decreased white blood cell count); an earlier
version without them admitted seven blood-count phenotypes as clean. The 449-phenotype index-anchored set is `results/prospective_landscape_B730_canonical.csv`
filtered to `canonical == True`. `canonical_index_screen.py` produces it by applying the
concurrent eligibility rule (a PheCode v1.2 description and organ group, sense-organ phenotypes
excluded) and the concurrent circularity screen to every index-anchored phenotype, with the
reason any phenotype was dropped in `excluded_because`. The same screen is applied at the 90 and
365-day leads.

Class labels in the file are `eye-specific`, `CBC-specific`,
`comparable` and `synergy`; the manuscript renames `comparable` to `indeterminate`.

```python
import pandas as pd
d = pd.read_csv("results/landscape_full.csv")
analysis = d[(d.circularity_tier == "clean") & (d.klass != "no signal")]   # 528 rows
```

## Result files behind each display item

| Display item | File |
|---|---|
| Figure 1, Table 2 | `landscape_full.csv`, `modality_drivers.csv` |
| Figure 2 complementarity | `orthogonality_results.csv`, `eye_predicts_cbc.csv`, `orthogonality_interaction_test.csv`, `interaction_fair.csv`, `interaction_fair_tuned.csv`, `modality_drivers.csv`, `flowmap_edge_data.csv` |
| Figure 3 index-anchored | `prospective_landscape_B730_canonical.csv` and the `_B90` and 365-day canonical files, `prospective_significance_B730.csv`, `fig3_stats.csv` |
| Figure 4 NHANES | `nhanes_validation_results.csv`, `nhanes_diabetes_adjusted.csv` |
| Figure 5 territories | `fig5_territory.csv`, `territory_permutation.csv`, `prospective_landscape_B730_canonical.csv`, `nhanes_validation_results.csv` |
| Supplementary Figure 1 | `mimic_diabetes_strat.csv`, `diabetes_strat_delong.csv`, `mimic_neuro_ablation.csv` |
| Supplementary Figure 2 | `utility_rebuild.csv`, `utility_calib.csv`, `utility_dca.csv` |
| Methods sensitivity analysis | `foldwise_sensitivity_B730.csv` (index-anchored landscape refit with vocabulary selection and imputation inside each fold) |

## Notes on the methods

Discrimination is cross-validated AUROC with analytic DeLong confidence intervals and
paired contrasts, implemented locally in `stats_utils.py` and unit tested against an
independent AUROC implementation. The ophthalmic feature vocabulary is built inside each
training fold for the concurrent landscape. Three scripts deliberately build the
vocabulary and imputation once on the full eligible set rather than fold-wise, which is
label blind but a weaker control; they are the cross-predictability test, the
index-anchored landscape and the diabetes stratification, and the manuscript Methods say
so. `foldwise_sensitivity.py` refits the index-anchored landscape with both steps moved
inside each training fold; base and CBC C-indices are unchanged and the eye C-index moves by a
median of 0.002. A fold whose regularized fit fails numerically is refitted on a penalty path
ending at a larger penalty, and the `fallback` column records where (one fold in each of four
phenotypes); every row is complete. Random operations are seeded. The published concurrent run seeded each
phenotype's matched-control draw from Python's `hash()`, which is salted per process, so that
draw cannot be replayed exactly; `landscape_phewide.py` now derives the seed from SHA-256, and a
rerun reproduces the landscape up to matched-control sampling.

The neurological and microvascular-renal territories in `build_fig5_territory.py` are keyword
sets on the phenotype description, with an `EXCLUDE` pattern that removes lexical matches that
are not mechanistic members (poisonings, neoplasms, gastrointestinal ulcers, myeloproliferative
disease, paralytic ileus, spondylosis without myelopathy).

## Citation

Wu E, Navidzadeh JO, Sahel J-A, Xing EP, Chhablani J. Ophthalmic and hematologic data
carry organ-partitioned, non-redundant information about systemic disease.

## License

MIT, see `LICENSE`. MIMIC-IV and NHANES carry their own terms; this license covers only
the code and the aggregate result files in this repository.
