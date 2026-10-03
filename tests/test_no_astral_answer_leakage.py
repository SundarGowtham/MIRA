"""
tests/test_no_astral_answer_leakage.py — Phase 16, "Decisions after
pass 2" item 1 (misc/PHASE16_INSTRUCTIONS.md), replacing
tests/test_no_astral_circularity.py's data-overlap half.

This is the STRICT gate that must pass. It fails only on what rule 1
(§1) actually forbids: training on an ASTRAL *answer* -- its precursor
sets or its principles -- not on a target formula merely appearing as
a prompt (that is ordinary test-set contamination, measured and
disclosed, not gated, by scripts/astral_overlap_report.py ->
results/phase16/astral_overlap.json).

1. No module under core/ or data_curation/ references an ASTRAL path
   in actual code (reused verbatim from the original circularity test).
2. No training record's stored ANSWER has a precursor set equal to
   ASTRAL's rule-picked ("predicted") set for its target. Equal to the
   conventional ("traditional") set is contamination, not leakage of
   the thing ASTRAL is for -- recorded by the overlap report, not
   gated here.

Run:  uv run python tests/test_no_astral_answer_leakage.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Imported by explicit file path, not `from scripts.astral_overlap_report
# import ...` -- a pip-installed package happens to also be named
# `scripts` and shadows this repo's scripts/ directory in the module
# namespace, confirmed directly (sys.path resolves `scripts` to
# .../site-packages/scripts/__init__.py) rather than guessed.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase16_astral_overlap_report", REPO_ROOT / "scripts" / "astral_overlap_report.py")
_overlap_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_overlap_report)
PRE_EXISTING_DATASETS = _overlap_report.PRE_EXISTING_DATASETS
PHASE16_CREATED_DATASETS = _overlap_report.PHASE16_CREATED_DATASETS
scan_dataset = _overlap_report.scan_dataset

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


ASTRAL_PATH_MARKERS = ("astral_validation_set.json", "astral_validation_set")


def scan_source_for_astral_paths(directory: Path) -> list[str]:
    hits = []
    for path in sorted(directory.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        in_docstring = False
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            stripped = line.strip()
            triple_count = stripped.count('"""') + stripped.count("'''")
            marker_hit = any(marker in line.lower() for marker in ASTRAL_PATH_MARKERS)
            is_comment = stripped.startswith("#")
            currently_in_docstring = in_docstring
            if triple_count % 2 == 1:
                in_docstring = not in_docstring
            if marker_hit and not is_comment and not currently_in_docstring:
                hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    return hits


print("== core/ and data_curation/ never reference an ASTRAL path ==")
core_hits = scan_source_for_astral_paths(REPO_ROOT / "core")
data_curation_hits = scan_source_for_astral_paths(REPO_ROOT / "data_curation")
check("core/*.py has zero ASTRAL-path references", len(core_hits) == 0, str(core_hits))
check("data_curation/*.py has zero ASTRAL-path references", len(data_curation_hits) == 0,
      str(data_curation_hits))

print("== no training record's ANSWER uses ASTRAL's rule-picked precursor set ==")
astral_entries = json.loads((REPO_ROOT / "results" / "astral_validation_set.json").read_text())["targets"]
astral_by_target = {t["target"]: t for t in astral_entries}

leaks = []
for name, path in {**PRE_EXISTING_DATASETS, **PHASE16_CREATED_DATASETS}.items():
    for row in scan_dataset(path, astral_by_target):
        if row["match"] == "matches_predicted":
            leaks.append({"dataset": name, **row})
check("zero training records whose answer equals ASTRAL's rule-picked set",
      len(leaks) == 0, str(leaks))

print("== datasets Phase 16 creates have zero ASTRAL overlap of any kind ==")
phase16_overlap = []
for name, path in PHASE16_CREATED_DATASETS.items():
    rows = scan_dataset(path, astral_by_target)
    if rows:
        phase16_overlap.append({"dataset": name, "rows": rows})
check("zero overlap (prompt or answer) in any Phase-16-created dataset",
      len(phase16_overlap) == 0, str(phase16_overlap))


if __name__ == "__main__":
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("All tests passed.")
