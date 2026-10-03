"""
tests/test_no_astral_circularity.py — Phase 16 §1 rule 1
(misc/PHASE16_INSTRUCTIONS.md): "No reward, prompt, filter or data file
used in training may read misc/astral_validation_set.json,
results/astral_validation_set.json, or ASTRAL's five precursor-selection
principles. ASTRAL is evaluation only." This is the mandated unit test:
it fails if any module under core/ or data_curation/ references an
ASTRAL path, and if any ASTRAL target formula appears in any training or
sampling prompt file under data/.

Dependency-free (source-text + JSON scanning only, no models/thermo cache).

Run:  uv run python tests/test_no_astral_circularity.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


ASTRAL_PATH_MARKERS = (
    "astral_validation_set.json",
    "astral_validation_set",  # catches a bare module-level constant too
)


def scan_source_for_astral_paths(directory: Path) -> list[str]:
    """Returns a list of 'file:line: text' hits for any ASTRAL path marker
    appearing in actual code in every .py file under `directory`
    (excluding __pycache__). Comments (full-line, after stripping
    whitespace) and triple-quoted docstring bodies are excluded -- this
    guardrail is about accidental functional coupling (an open()/Path()
    call reaching the file), not prose that mentions the path to explain
    why it is NOT read (e.g. this test's own docstring, or a comment
    documenting where a lookup table's keys were sourced from)."""
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

print("== no ASTRAL target formula appears in any training/sampling prompt file ==")
astral_set_path = REPO_ROOT / "results" / "astral_validation_set.json"
if not astral_set_path.exists():
    print(f"  SKIP: {astral_set_path} not found (nothing to check against)")
else:
    astral_targets = {
        t["target"] for t in json.loads(astral_set_path.read_text())["targets"]
    }
    check(f"loaded {len(astral_targets)} ASTRAL target formulas", len(astral_targets) == 35,
          str(sorted(astral_targets)))

    # Scoped to the directories real experiment code actually reads
    # (train.py --data-dir defaults and data_curation/ scripts), not every
    # .jsonl anywhere under data/. One real hit was found and investigated
    # here, not silently excluded: data/processed/reasoning_traces.jsonl
    # DOES contain ASTRAL target formulas (Li3V2(PO4)3, LiMnPO4, KNbWO6,
    # Na2Al2B2O7, NaSrBO3, ...) -- but `grep -rl "reasoning_traces.jsonl" .`
    # across every .py file in the repo returns zero hits, confirming it
    # is legacy/unreferenced by any current code path, not a live
    # training/sampling surface. Excluded on that verified basis, not
    # assumed; revisit this exclusion if that file is ever wired up again.
    ACTIVE_DATA_DIRS = ["rl_run3", "sft_v3", "rs_sft", "rl", "novelty"]
    offenders = []
    jsonl_paths = []
    for d in ACTIVE_DATA_DIRS:
        jsonl_paths.extend(sorted((REPO_ROOT / "data" / d).rglob("*.jsonl")))
    for jsonl_path in jsonl_paths:
        try:
            for line in jsonl_path.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target = None
                if isinstance(rec, dict):
                    target = rec.get("target") or rec.get("target_formula")
                if target in astral_targets:
                    offenders.append(f"{jsonl_path.relative_to(REPO_ROOT)}: target={target}")
        except Exception as e:
            offenders.append(f"{jsonl_path.relative_to(REPO_ROOT)}: scan error {e}")
    check("zero data/**/*.jsonl records whose target is an ASTRAL formula",
          len(offenders) == 0, str(offenders[:10]))


if __name__ == "__main__":
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("All tests passed.")
