"""Automated circularity detection so we don't hand-curate tautologies.

Two mechanisms:
  (1) CBC-definitional: the disease IS a blood-count abnormality (anemia, the
      cytopenias/cytoses, polycythemia) -> predicting it from CBC is tautological.
      Heme malignancies (leukemia/lymphoma/myeloma) flagged separately as 'expected'
      (CBC strongly abnormal, but a real finding, not the definition).
  (2) Eye-manifestation: an eye feature is the ocular sequela of the systemic
      disease (diabetic retinopathy <-> diabetes, hypertensive retinopathy <-> HTN).
      Detected generically by shared word-root between the disease name and an eye
      code's title; the driver additionally measures the AUROC drop when these eye
      codes are ablated.

Policy: tag, do not silently drop. Headline analysis runs on 'clean'; tautological
diseases are reported as positive controls.
"""
import re

CBC_DEFINITIONAL = re.compile(
    r"\b(anemia|anaemia|thrombocytopenia|thrombocytosis|thrombocythemia|"
    r"neutropenia|neutrophilia|leukopenia|leukocytosis|lymphocytosis|lymphopenia|"
    r"lymphocytopenia|monocytosis|eosinophilia|basophilia|pancytopenia|"
    r"polycythemia|agranulocytosis|granulocytopenia|erythrocytosis)\b", re.I)

HEME_NEOPLASM = re.compile(
    r"\b(leukemia|leukaemia|lymphoma|myeloma|myelodysplas|"
    r"myeloprolifer|hodgkin|polycythemia vera)\b", re.I)

# Care-process phenotypes record a treatment status or an encounter rather than a disease.
# They index secondary-prevention status and care intensity, which is the utilization confound
# the paper controls for elsewhere, so an eye block that discriminates them is reading how much
# care a patient receives. The prospective arm already screened these; the concurrent arm did not,
# which left three of them classed eye-specific (long-term use of aspirin, eye AUROC 0.708 against
# the CBC's 0.582). Both arms now use this one screen.
CARE_PROCESS = re.compile(
    r"encounter for|long-term \(current\) use|\bstatus\b|aftercare|"
    r"\bdialysis\b|chemotherapy|\bdevice\b|transplant status|"
    r"presence of|fitting and adjustment", re.I)


def is_care_process(description):
    return bool(CARE_PROCESS.search(str(description)))

_STOP = {"disorder", "disorders", "disease", "diseases", "chronic", "acute",
         "other", "unspecified", "primary", "secondary", "syndrome", "mellitus",
         "essential", "without", "complications", "complication", "type"}


def is_cbc_definitional(description):
    return bool(CBC_DEFINITIONAL.search(str(description)))


def is_heme_neoplasm(description):
    return bool(HEME_NEOPLASM.search(str(description)))


def _roots(description):
    return {w[:6] for w in re.findall(r"[a-z]{5,}", str(description).lower())
            if w not in _STOP}


def manifestation_eye_codes(description, code_to_title):
    """Eye codes whose title shares a >=6-char word-root with the disease name
    (e.g. 'diabetic' retinopathy for diabetes; 'hypertensive' retinopathy for HTN)."""
    roots = {r for r in _roots(description) if len(r) >= 6}
    if not roots:
        return set()
    hits = set()
    for code, title in code_to_title.items():
        t = str(title).lower()
        if any(r in t for r in roots):
            hits.add(code)
    return hits


def tier(description, eye_ablation_drop=0.0, single_analyte_dominant=False):
    """Combine signals into the headline-filter tier.

    NOTE: single_analyte_dominant is recorded but does NOT force 'tautological' —
    one analyte dominating (e.g. a cancer shifting platelets) is a real finding,
    unlike anemia where the count IS the diagnosis. Only definitional (lexicon)
    diseases are tautological."""
    if is_cbc_definitional(description):
        return "tautological"
    if is_heme_neoplasm(description):
        return "expected_cbc"
    if is_care_process(description):
        return "care_process"
    if eye_ablation_drop >= 0.05:
        return "expected_manifestation"
    return "clean"


if __name__ == "__main__":
    titles = {"36201": "Background diabetic retinopathy",
              "3625": "Hypertensive retinopathy",
              "37300": "Blepharitis", "377": "Optic neuritis"}
    for d in ["Diabetes mellitus", "Essential hypertension", "Multiple sclerosis",
              "Iron deficiency anemia", "Acute leukemia", "Chronic kidney disease"]:
        print(f"{d:<26} cbc_def={is_cbc_definitional(d)!s:<5} "
              f"heme={is_heme_neoplasm(d)!s:<5} "
              f"manifest_eye={sorted(manifestation_eye_codes(d, titles))}")
