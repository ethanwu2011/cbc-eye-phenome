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

`src/config.py` resolves the repository root automatically. Set `CBCEYE_ROOT` only if you
move `results/` elsewhere. Scripts are run from the repository root, for example
`python src/landscape_phewide.py`.

Reproducing only the figures and tables needs neither environment variable, because the
aggregate results they read are already in `results/`.

## Pipeline

Build the intermediate tables, then run any stage independently.

| Stage | Scripts |
|---|---|
| Phenotyping and base tables | `phecode_map.py`, `temporal_base.py`, `build_phecode_onset.py` |
| Circularity screen | `circularity.py`, `apply_circularity_prospective.py` |
| AUROC inference | `stats_utils.py` (fast DeLong covariance, used throughout) |
| Concurrent landscape | `landscape_phewide.py`, `mimic_landscape.py`, `make_figures.py` |
| Complementarity tests | `mimic_orthogonality.py`, `eye_predicts_cbc.py`, `orthogonality_interaction_test.py`, `interaction_fair.py` |
| Index-anchored landscape | `prospective_landscape.py`, `prospective_significance.py`, `leadtime_organ_partition.py`, `temporal_leadtime.py`, `leadtime_screen.py`, `leadtime_survival.py` |
| Confound controls | `mimic_diabetes_strat.py`, `diabetes_strat_delong.py`, `mimic_neuro_ablation.py` |
| Neurological and microvascular axis | `neuro_axis.py`, `neuro_why_not.py`, `micro_vs_macro.py` |
| NHANES replication | `nhanes_validation.py`, `nhanes_diabetes_utility.py`, `utility_rebuild.py` |
| Figures and tables | `compose_figures.py`, `build_fig3_final.py`, `build_fig5_territory.py`, `build_fig_triangulation.py`, `build_fig_eye_gain.py`, `build_nhanes_figure.py`, `build_landscape_plotly.py`, `build_fig1b_organ.py`, `build_tables.py` |

`cohort_build.py` assembles the matched case-control design and is imported by the
orthogonality and interaction tests.

## The analysis set

`results/landscape_full.csv` holds all 635 screened phenotypes. **The 535-phenotype
analysis set reported in the paper is the subset with `circularity_tier == "clean"` and
`klass != "no signal"`.** Filtering on `circularity_tier` alone gives 581 rows and
different numbers. Class labels in the file are `eye-specific`, `CBC-specific`,
`comparable` and `synergy`; the manuscript renames `comparable` to `indeterminate`.

```python
import pandas as pd
d = pd.read_csv("results/landscape_full.csv")
analysis = d[(d.circularity_tier == "clean") & (d.klass != "no signal")]   # 535 rows
```

## Result files behind each display item

| Display item | File |
|---|---|
| Figure 1, Table 1, Table 2 | `landscape_full.csv`, `panel.csv` |
| Figure 2 complementarity | `orthogonality_results.csv`, `eye_predicts_cbc.csv`, `orthogonality_interaction_test.csv`, `interaction_fair.csv`, `interaction_fair_tuned.csv`, `modality_drivers.csv` |
| Figure 3 index-anchored | `prospective_landscape_B730.csv`, `prospective_significance_B730.csv`, `temporal_leadtime.csv`, `fig3_stats.csv` |
| Figure 4 NHANES | `nhanes_validation_results.csv`, `nhanes_diabetes_adjusted.csv` |
| Figure 5 territories | `fig5_territory.csv`, `triangulation.csv`, `leadtime_organ_partition.csv` |
| Supplementary Figure 1 | `mimic_diabetes_strat.csv`, `diabetes_strat_delong.csv`, `mimic_neuro_ablation.csv` |
| Supplementary Figure 2 | `utility_rebuild.csv`, `utility_calib.csv`, `utility_dca.csv` |

## Notes on the methods

Discrimination is cross-validated AUROC with analytic DeLong confidence intervals and
paired contrasts, implemented locally in `stats_utils.py` and unit tested against an
independent AUROC implementation. The ophthalmic feature vocabulary is built inside each
training fold for the concurrent landscape. Three scripts deliberately build the
vocabulary and imputation once on the full eligible set rather than fold-wise, which is
label blind but a weaker control; they are the cross-predictability test, the
index-anchored landscape and the diabetes stratification, and the manuscript Methods say
so. Random operations are seeded.

## Citation

Wu E, Navidzadeh JO, Sahel J-A, Xing EP, Chhablani J. Ophthalmic and hematologic data
carry organ-partitioned, non-redundant information about systemic disease.

## License

MIT, see `LICENSE`. MIMIC-IV and NHANES carry their own terms; this license covers only
the code and the aggregate result files in this repository.
