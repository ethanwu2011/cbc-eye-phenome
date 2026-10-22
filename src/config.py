"""Paths. Override with environment variables; defaults assume the repository layout.

    CBCEYE_ROOT   repository root. Defaults to the parent of src/.
    MIMIC_HOSP    MIMIC-IV v3.1 `hosp` module directory. Required only by the scripts that
                  build the intermediate tables from raw MIMIC.
    NHANES_DIR    directory of NHANES XPT files. Required only for the NHANES replication.

Scripts read and write their files under CBCEYE_ROOT. The aggregate result files shipped in
this repository live in results/; the intermediate patient-level tables (*.parquet) are not
redistributed and must be regenerated from credentialed MIMIC-IV access.
"""
import os

CBCEYE_ROOT = os.environ.get(
    "CBCEYE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIMIC_HOSP = os.environ.get("MIMIC_HOSP", "")
NHANES_DIR = os.environ.get("NHANES_DIR", "")

RESULTS = os.path.join(CBCEYE_ROOT, "results")
FIGURES = os.path.join(CBCEYE_ROOT, "figures")


def require(var, name):
    if not var:
        raise SystemExit(
            f"{name} is not set. Point it at your local copy, for example\n"
            f"  export {name}=/path/to/data\n"
            "See the README for how to obtain MIMIC-IV and NHANES.")
    return var
