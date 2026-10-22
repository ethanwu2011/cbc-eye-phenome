"""Lay out the shipped aggregate results the way the pipeline leaves them, so the figure
scripts and the territory test run on a fresh clone without MIMIC-IV.

    python src/stage_results.py work
    export CBCEYE_ROOT="$PWD/work"
    python src/build_fig5_territory.py

Copies every file in results/ into the working root (and into results/ under it, for the
scripts that read through config.RESULTS), writes landscape_full.parquet from its CSV, and
creates the figure output directories.
"""
import os, shutil, sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")

dst = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "work"))
for d in (("final_manuscript", "figures"), ("figures",), ("results",)):
    os.makedirs(os.path.join(dst, *d), exist_ok=True)
for f in sorted(os.listdir(RESULTS)):
    if f.endswith(".csv"):
        shutil.copy2(os.path.join(RESULTS, f), dst)
        shutil.copy2(os.path.join(RESULTS, f), os.path.join(dst, "results"))
pd.read_csv(os.path.join(RESULTS, "landscape_full.csv"), dtype={"phecode": str}).to_parquet(
    os.path.join(dst, "landscape_full.parquet"))
print(f"staged {RESULTS} -> {dst}\nnow run:  export CBCEYE_ROOT={dst}")
