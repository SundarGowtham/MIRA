#!/usr/bin/env python
"""
research/phase17_ra_record_fixes.py — Block R triage patch (regular
Claude, 2026-10-22), fixing three things flagged in R-a
(results/phase17/ra_symmetry_axioms.json) WITHOUT overwriting any
existing number:

1. "AS REGISTERED" vs "CORRECTED" for monotonicity. The pre-registered
   test (append one unknown-type op to an otherwise-clean route) is a
   SPECIFICATION ERROR, not a validator bug: _check_operation_order
   EXCLUDES any adjacent pair touching an unrecognized op type from
   both `assessed` and `violations`, so appending one such op can
   never move the score on any version, buggy or not. The original
   run's literal per-version counts (preserved below, read from
   run_logs/phase17_ra_symmetry_axioms.log, never re-derived) show
   monotonicity flagging ALL FIVE versions including current_HEAD and
   v3 (both expected-clean) -- a test with zero specificity, not a
   real bug detector. Handled the same way as Phase 15's eps=-20: the
   spec was wrong, state what the registered test actually found, then
   report the corrected one next to it, never instead of it.

2. Atmosphere axioms (notation_atmosphere_synonym, token_identity):
   the existing JSON already explains WHY 0 violations has no power
   (every one of the 60 sampled routes' targets landed in
   _check_chempot_atmosphere's "flexible" regime, where the atmosphere
   string is never consulted). This patch (a) adds an explicit
   report_card_value="untested (no power)" field so the report card
   does not read these as a pass, and (b) runs a genuine power-fix:
   scans the full RS-SFT target pool for targets whose chempot
   envelope DOES require oxidizing or reducing conditions (computed
   straight from thermo_checker.composition_chempots, independent of
   any route -- the same inputs _check_chempot_atmosphere itself
   reads), builds the same two case types against THOSE targets'
   real routes, and reruns across all 5 versions. Labeled
   [E, post-hoc power fix] throughout; stored under new keys
   (*_powerfix), the original 0-power run is left untouched.

3. Boundary axiom: states explicitly what was found (target_stability
   and its gradeability tag, every version) rather than just
   "0 violations" -- extracted directly from the already-scored
   scratch JSONs (no rerun needed), confirming 1.0/"discrete" (a real
   0.0 e_above_hull correctly distinguished from "missing") on every
   one of the 5 tested versions.

NO EDIT to validator.py. NO commits (per regular Claude's triage
instructions). Edits only results/phase17/ra_symmetry_axioms.json,
additively.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/users/gsundar/projects/MIRA")
from research.phase17_ra_symmetry_axioms import (  # noqa: E402
    REPO_ROOT, OUT_PATH, SCRATCH, SOURCES, VERSION_ORDER, WORKTREE_REFS,
    setup_worktree, teardown_worktree, score_version, evaluate_case,
)
from core.reward import parse_completion  # noqa: E402
from validator import (  # noqa: E402
    ThermoChecker, SynthesisValidator, MU_O_OXIDIZING_REQUIRED, MU_O_REDUCING_REQUIRED, HEATING_OP_TYPES,
)
import pickle  # noqa: E402

# ---------------------------------------------------------------------------
# Item 1: monotonicity -- the registered run's literal numbers, from the log
# (run_logs/phase17_ra_symmetry_axioms.log lines 32-36), never re-run,
# never re-derived -- just transcribed here so they live next to the fix.
# ---------------------------------------------------------------------------
MONOTONICITY_AS_REGISTERED = {
    "v0_pre_scale_fix_2026-07-15": {"n_cases": 200, "n_scored": 200, "n_skipped": 0, "n_violations": 200,
                                     "violation_rate_of_scored": 1.0, "expected_buggy": True},
    "v1_pre_hydrate_fix_2026-07-20": {"n_cases": 200, "n_scored": 200, "n_skipped": 0, "n_violations": 200,
                                       "violation_rate_of_scored": 1.0, "expected_buggy": True},
    "v2_pre_atmtoken_monotonicity_fix_2026-07-21": {"n_cases": 200, "n_scored": 200, "n_skipped": 0,
                                                     "n_violations": 200, "violation_rate_of_scored": 1.0,
                                                     "expected_buggy": True},
    "v3_pre_noneprop_fix_2026-07-22": {"n_cases": 200, "n_scored": 200, "n_skipped": 0, "n_violations": 200,
                                        "violation_rate_of_scored": 1.0, "expected_buggy": False},
    "current_HEAD": {"n_cases": 200, "n_scored": 200, "n_skipped": 0, "n_violations": 200,
                      "violation_rate_of_scored": 1.0, "expected_buggy": False},
}


def properly_flagged(axiom_summary: dict, expected_buggy: set, named: list) -> int:
    """Stricter than the original script's n_flagged: an axiom counts
    as PROPERLY flagged only if it shows violations>0 at every expected-
    buggy version AND violations==0 at every expected-clean version --
    i.e. it actually discriminates, not just fires once somewhere."""
    n = 0
    for axiom in named:
        versions = axiom_summary[axiom]
        ok = True
        for v, cell in versions.items():
            is_buggy_here = v in expected_buggy.get(axiom, set())
            if is_buggy_here and cell["n_violations"] == 0:
                ok = False
            if not is_buggy_here and cell["n_violations"] > 0:
                ok = False
        if ok and any(v in expected_buggy.get(axiom, set()) for v in versions):
            n += 1
    return n


def main():
    report = json.loads(OUT_PATH.read_text())
    axiom_summary = report["axiom_version_summary"]
    expected_buggy = report["expected_buggy_versions"]
    named = report.get("named_historical_bugs") or [
        "scale", "notation_hydrate", "notation_atmosphere_synonym",
        "token_identity", "monotonicity", "abstention",
    ]

    # ---- Item 1: monotonicity as-registered vs corrected -----------------
    corrected_mono = axiom_summary["monotonicity"]
    as_registered_summary = dict(axiom_summary)
    as_registered_summary["monotonicity"] = MONOTONICITY_AS_REGISTERED
    n_flagged_registered = properly_flagged(as_registered_summary, expected_buggy, named)
    n_flagged_corrected = properly_flagged(axiom_summary, expected_buggy, named)
    n_viol_current_registered = sum(as_registered_summary[ax]["current_HEAD"]["n_violations"] for ax in axiom_summary)
    n_viol_current_corrected = sum(axiom_summary[ax]["current_HEAD"]["n_violations"] for ax in axiom_summary)

    report["monotonicity_as_registered"] = MONOTONICITY_AS_REGISTERED
    report["monotonicity_deviation"] = {
        "statement": "The pre-registered monotonicity transformation (append ONE unknown-type "
                     "operation to an otherwise-clean route) is a SPECIFICATION ERROR: "
                     "_check_operation_order excludes any adjacent pair touching an unrecognized "
                     "op type from both `assessed` and `violations`, so this transformation could "
                     "never move the score on ANY version, buggy or not -- it flagged all 5 "
                     "versions (including current_HEAD and v3, both expected-clean), i.e. it has "
                     "zero specificity, not that it detected a real bug. Handled like Phase 15's "
                     "eps=-20 (finding 28): a specification error, not a validator regression. "
                     "The real bug/fix is specifically about assessed==0 (a route of ENTIRELY "
                     "unrecognized op types): 1.0 pre-fix, 0.5 post-fix. The corrected "
                     "transformation (replace ALL of a route's operations with two unrecognized-"
                     "type steps) is a direct exact-value check against that condition and is "
                     "stored under axiom_version_summary.monotonicity (unchanged from the prior "
                     "patch); the as-registered numbers are preserved unmodified above.",
        "as_registered": {
            "properly_flagged_bugs_of_6": n_flagged_registered,
            "n_violations_on_current_HEAD_all_axioms": n_viol_current_registered,
            "prediction_1_passes": n_flagged_registered >= 4,
            "prediction_2_passes": n_viol_current_registered == 0,
        },
        "corrected": {
            "properly_flagged_bugs_of_6": n_flagged_corrected,
            "n_violations_on_current_HEAD_all_axioms": n_viol_current_corrected,
            "prediction_1_passes": n_flagged_corrected >= 4,
            "prediction_2_passes": n_viol_current_corrected == 0,
        },
        "corrected_test_flags_historical_bug": (
            axiom_summary["monotonicity"]["v0_pre_scale_fix_2026-07-15"]["n_violations"] > 0
            and axiom_summary["monotonicity"]["v1_pre_hydrate_fix_2026-07-20"]["n_violations"] > 0
            and axiom_summary["monotonicity"]["v2_pre_atmtoken_monotonicity_fix_2026-07-21"]["n_violations"] > 0
            and axiom_summary["monotonicity"]["v3_pre_noneprop_fix_2026-07-22"]["n_violations"] == 0
            and axiom_summary["monotonicity"]["current_HEAD"]["n_violations"] == 0
        ),
    }
    # Using the STRICTER "properly flagged" definition throughout (not the original
    # script's looser "fired at least once at the latest buggy version" rule) --
    # recompute the top-level prediction fields to match, for both regimes.
    report["prediction_1_at_least_4_of_6"] = {
        "statement": "The suite flags at least 4 of the 6 named historical bugs at the versions "
                     "where they existed (properly: violations>0 at every expected-buggy version "
                     "AND violations==0 at every expected-clean version for that axiom).",
        "as_registered": {"n_flagged": n_flagged_registered, "n_total_named_bugs": 6,
                           "passes": n_flagged_registered >= 4},
        "corrected": {"n_flagged": n_flagged_corrected, "n_total_named_bugs": 6,
                      "passes": n_flagged_corrected >= 4},
    }
    report["prediction_2_zero_on_current"] = {
        "statement": "It flags none on the current version.",
        "as_registered": {"n_violations_on_current": n_viol_current_registered,
                           "passes": n_viol_current_registered == 0},
        "corrected": {"n_violations_on_current": n_viol_current_corrected,
                      "passes": n_viol_current_corrected == 0},
    }
    print("monotonicity as-registered:", report["monotonicity_deviation"]["as_registered"], flush=True)
    print("monotonicity corrected:   ", report["monotonicity_deviation"]["corrected"], flush=True)

    # ---- Item 3: boundary -- explicit per-version raw values -------------
    boundary_values = {}
    for v in VERSION_ORDER:
        d = json.loads((SCRATCH / f"ra_scored_{v}.json").read_text())
        case = d["results"].get("boundary_cu_metal", {})
        variant = case.get("cu_on_hull", {})
        bd = variant.get("breakdown", {})
        boundary_values[v] = {
            "target_stability": bd.get("target_stability"),
            "target_stability_gradeability": bd.get("target_stability_gradeability"),
        }
    all_correct = all(c["target_stability"] == 1.0 and c["target_stability_gradeability"] == "discrete"
                      for c in boundary_values.values())
    report["boundary_result"] = {
        "statement": "What happened, per version, for target=Cu (pure element, e_above_hull=0 by "
                     "definition): target_stability and its gradeability tag, read directly from "
                     "each version's own scored output (no rerun). A correct implementation must "
                     "return (1.0, \"discrete\") -- distinguishing a genuine 0.0 e_above_hull from "
                     "a missing PD entry (which would return 0.5, \"sentinel_no_entry\"). This was "
                     "true on EVERY version tested, including the earliest (v0, 2026-07-15) -- "
                     "consistent with the module docstring's claim that no historical regression "
                     "exists here within the range checked; this is a forward-looking correctness "
                     "check, not evidence of a fix commit.",
        "per_version": boundary_values,
        "all_versions_correct": all_correct,
    }
    print("boundary_result:", boundary_values, flush=True)

    # ---- Item 2a: report-card values for the atmosphere axioms -----------
    report["report_card_values"] = {
        "notation_atmosphere_synonym": "untested (no power) -- see atmosphere_test_power_gap_note",
        "token_identity": "untested (no power) -- see atmosphere_test_power_gap_note",
        "order": "pass (forward-looking only, no historical bug to catch)",
        "boundary": "pass (forward-looking only, no historical bug to catch)",
        "scale": "pass",
        "notation_hydrate": "pass",
        "monotonicity": "pass (corrected test; fails as registered -- see monotonicity_deviation)",
        "abstention": "pass",
    }

    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH} (items 1, 3, 2a patched)")

    # ---- Item 2b: post-hoc power fix --------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("POST-HOC POWER FIX: scanning for targets whose chempot envelope "
          "genuinely requires oxidizing or reducing conditions...", flush=True)
    with open(REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl", "rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_sharded_cache(REPO_ROOT / "data" / "cache" / "pd_index.json", REPO_ROOT)

    from pymatgen.core import Composition, Element  # noqa: E402
    O_EL = Element("O")

    def classify_target(target: str) -> str:
        try:
            comp = Composition(target)
            if O_EL not in comp.elements:
                return "not_oxide"
            all_chempots = thermo.composition_chempots(target)
            mu_O_ref = thermo.oxygen_reference_energy(target)
            if not all_chempots or mu_O_ref is None:
                return "ungradeable"
            o_mus_delta = [float(facet[O_EL]) - float(mu_O_ref) for facet in all_chempots.values() if O_EL in facet]
            if not o_mus_delta:
                return "ungradeable"
            mu_lo, mu_hi = min(o_mus_delta), max(o_mus_delta)
            if mu_lo > MU_O_OXIDIZING_REQUIRED:
                return "needs_oxidizing"
            if mu_hi < MU_O_REDUCING_REQUIRED:
                return "needs_reducing"
            return "flexible"
        except Exception:
            return "error"

    # Pool: every distinct target across both RS-SFT files, with at least one
    # parseable route that has a REAL HEATING operation (HEATING_OP_TYPES,
    # after _normalize_op_type) -- NOT just "any first operation" (the first
    # attempt used operations[0] unconditionally; for several routes that is
    # a MixingOperation, so `atms` stayed empty regardless of what we set
    # `heating_atmosphere` to). SECOND self-caught bug, also fixed here:
    # _check_chempot_atmosphere concatenates heating_atmosphere from EVERY
    # heating-type op in the route (calcine + sinter + cool, typically 2-3),
    # not just the first -- overwriting only one left the others' original
    # strings ("oxidizing", "inert" -- words _classify_atmosphere's token
    # sets do NOT contain) in `classes` as "unknown", which fails the
    # `all(c == "ox" ...)"/`all(c in {"red","inert"} ...)` check regardless
    # of what the first op was set to, forcing a uniform 0.0 on every
    # variant. Fixed by setting heating_atmosphere on ALL heating-type ops
    # in the route to the SAME test string, consistently, per variant.
    candidates: dict[str, dict] = {}
    for path in SOURCES:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                target = rec.get("target")
                if not target or target in candidates:
                    continue
                try:
                    route = parse_completion(rec["completion"], target)
                except Exception:
                    continue
                heat_indices = [i for i, op in enumerate(route.operations or [])
                                if SynthesisValidator._normalize_op_type(op.type) in HEATING_OP_TYPES]
                if not heat_indices:
                    continue
                ops_out = []
                for op in route.operations:
                    oc = op.conditions
                    ops_out.append({
                        "type": op.type,
                        "conditions": {
                            "heating_temperature": list(getattr(oc, "heating_temperature", []) or []),
                            "heating_time": list(getattr(oc, "heating_time", []) or []),
                            "heating_atmosphere": list(getattr(oc, "heating_atmosphere", []) or []),
                            "mixing_media": getattr(oc, "mixing_media", None),
                            "atmosphere": getattr(oc, "atmosphere", None),
                        },
                    })
                candidates[target] = {
                    "target": target,
                    "precursors": [{"formula": p.formula, "amount": getattr(p, "amount", 1.0)}
                                   for p in (route.precursors or [])],
                    "operations": ops_out,
                    "_heat_indices": heat_indices,
                }

    classifications = {t: classify_target(t) for t in candidates}
    counts: dict[str, int] = {}
    for c in classifications.values():
        counts[c] = counts.get(c, 0) + 1
    print(f"scanned {len(candidates)} distinct targets with >=1 operation-bearing route: {counts}", flush=True)

    qualifying = [t for t, c in classifications.items() if c in ("needs_oxidizing", "needs_reducing")]
    print(f"{len(qualifying)} targets qualify (needs_oxidizing or needs_reducing)", flush=True)

    if not qualifying:
        report["atmosphere_powerfix_result"] = {
            "statement": "[E, post-hoc power fix] Scanned every distinct target with an "
                         "operation-bearing route across both RS-SFT files; ZERO qualify as "
                         "needs_oxidizing or needs_reducing. The power gap could not be fixed "
                         "with this sample pool -- all candidate targets' chempot envelopes are "
                         "flexible. Reported honestly as an unresolved gap, not papered over.",
            "n_candidates_scanned": len(candidates),
            "classification_counts": counts,
        }
        OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
        print(f"\n-> {OUT_PATH} (power-fix: no qualifying targets found)")
        return

    def set_all_heat_atms(route_dict: dict, heat_idxs: list, value: list) -> dict:
        v = json.loads(json.dumps(route_dict))
        for idx in heat_idxs:
            v["operations"][idx]["conditions"]["heating_atmosphere"] = list(value)
        return v

    cases, meta = {}, {}
    for i, t in enumerate(qualifying):
        r = dict(candidates[t])
        heat_idxs = r.pop("_heat_indices")
        cid = f"notation_atm_synonym_powerfix_{i}"
        ar_variant = set_all_heat_atms(r, heat_idxs, ["Ar"])
        argon_variant = set_all_heat_atms(r, heat_idxs, ["argon"])
        cases[cid] = {"ar": ar_variant, "argon": argon_variant}
        meta[cid] = {"axiom": "notation_atmosphere_synonym_powerfix", "compare": "equal_channels",
                     "channels": ["chempot_atmosphere"], "target_classification": classifications[t]}

        cid2 = f"token_identity_powerfix_{i}"
        h2_variant = set_all_heat_atms(r, heat_idxs, ["H2"])
        h2o_variant = set_all_heat_atms(r, heat_idxs, ["H2O"])
        o2_variant = set_all_heat_atms(r, heat_idxs, ["O2"])
        cases[cid2] = {"h2_reducer": h2_variant, "h2o_trap": h2o_variant, "o2_oxidizer": o2_variant}
        meta[cid2] = {"axiom": "token_identity_powerfix", "compare": "trap_vs_reducer_oxidizer",
                      "channel": "chempot_atmosphere", "target_classification": classifications[t]}

    cases_path = SCRATCH / "ra_cases_atmosphere_powerfix.json"
    cases_path.write_text(json.dumps(cases))

    eval_by_version: dict[str, dict] = {}
    for version in VERSION_ORDER:
        print(f"=== scoring atmosphere power-fix cases on: {version} ===", flush=True)
        out_path = SCRATCH / f"ra_scored_powerfix_{version}.json"
        if version == "current_HEAD":
            scored = score_version(REPO_ROOT, cases_path, out_path)
        else:
            wt = setup_worktree(version, WORKTREE_REFS[version])
            try:
                scored = score_version(wt, cases_path, out_path)
            finally:
                teardown_worktree(version)
        weights = scored.get("weights", {})
        sentinel_tags = scored.get("sentinel_tags", [])
        results = scored.get("results", {})
        evals = {}
        for cid, variants_out in results.items():
            variants_out["_weights"] = weights
            variants_out["_sentinel_tags"] = sentinel_tags
            evals[cid] = evaluate_case(cid, meta[cid], variants_out)
        eval_by_version[version] = evals

    powerfix_summary = {"notation_atmosphere_synonym_powerfix": {}, "token_identity_powerfix": {}}
    for version in VERSION_ORDER:
        for axiom_key in powerfix_summary:
            prefix = "notation_atm_synonym_powerfix" if "notation" in axiom_key else "token_identity_powerfix"
            evals = {cid: e for cid, e in eval_by_version[version].items() if cid.startswith(prefix)}
            n_total = len(evals)
            n_skipped = sum(1 for e in evals.values() if "skipped" in e)
            n_scored = n_total - n_skipped
            n_violations = sum(1 for e in evals.values() if e.get("violation") is True)
            examples = [cid for cid, e in evals.items() if e.get("violation") is True][:3]
            powerfix_summary[axiom_key][version] = {
                "n_cases": n_total, "n_scored": n_scored, "n_skipped": n_skipped,
                "n_violations": n_violations,
                "violation_rate_of_scored": round(n_violations / n_scored, 3) if n_scored else None,
                "example_violating_cases": examples,
            }
        print(version, {k: powerfix_summary[k][version] for k in powerfix_summary}, flush=True)

    report["atmosphere_powerfix_result"] = {
        "statement": f"[E, post-hoc power fix] {len(qualifying)} targets found with a genuine "
                     "oxidizing/reducing chempot requirement (scanned via "
                     "thermo_checker.composition_chempots directly, same inputs "
                     "_check_chempot_atmosphere itself reads, independent of any route). Cases "
                     "rebuilt against these targets' own real routes and rerun across all 5 "
                     "versions. This does NOT replace the original 0-power run (preserved under "
                     "axiom_version_summary); it is reported separately as a labeled exploratory "
                     "follow-up.",
        "n_candidates_scanned": len(candidates),
        "classification_counts": counts,
        "n_qualifying_targets": len(qualifying),
        "qualifying_targets": qualifying,
        "per_version": powerfix_summary,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH} (power-fix complete)")


if __name__ == "__main__":
    main()
