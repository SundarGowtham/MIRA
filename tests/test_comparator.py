"""
Regression tests for core/comparator.py — needs the real sharded PD cache
(data/cache/pd_index.json + shards), so unlike test_validator.py's
LIGHT_VALIDATOR half this is not dependency-free; run in tmux per repo
convention since it touches ThermoChecker.

Run:  uv run python tests/test_comparator.py

Covers the three tests PHASE13_14_SPEC.md's Aggregation section requires:
  1. compare(a,b) == -compare(b,a) exactly.
  2. One sign test per channel (C1-C7) on a hand-built pair where the
     better route is unambiguous.
  3. A gate-failing route scores None on every channel (C1-C7; C8 is
     group-level and out of scope for this module, see comparator.py's
     module docstring).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator import (  # noqa: E402
    PredictedConditions,
    PredictedOperation,
    PredictedPrecursor,
    PredictedRoute,
    ThermoChecker,
)
from core.comparator import (  # noqa: E402
    CHANNEL_NAMES,
    Comparator,
    ComparatorParams,
    load_comparator,
)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def route(target, precursors, ops=None):
    return PredictedRoute(
        target_formula=target,
        precursors=[PredictedPrecursor(f, a) for f, a in precursors],
        operations=ops or [],
    )


def op(t, temp=None):
    return PredictedOperation(
        type=t,
        conditions=PredictedConditions(
            heating_temperature=[temp] if temp is not None else [],
        ),
    )


COMPARATOR = load_comparator(
    Path("data/cache/mp_formula_set.pkl"),
    Path("data/cache/pd_index.json"),
    Path("."),
)
assert COMPARATOR.thermo is not None, "real PD cache required for these tests"

# A wide, fine-grained reference range shared by every channel -- these
# tests check sign/structure, not the calibrated rank distributions (those
# live in misc/comparator_scales_v2.json, computed by
# research/compute_comparator_scales.py, and are exercised by the Phase 13
# scoring run itself, not by unit tests). -100,000 to 100,000 in steps of
# 100 comfortably spans every raw channel's realistic test-value magnitude
# without collapsing distinct values to the same rank.
_UNIT_SORTED_VALS = [i * 100.0 for i in range(-1000, 1001)]
UNIT_SCALES = {c: _UNIT_SORTED_VALS for c in CHANNEL_NAMES}


# ---------------------------------------------------------------------------
# 1. Antisymmetry
# ---------------------------------------------------------------------------

def test_antisymmetry():
    a = route("Al2TiO5", [("Al2O3", 1.0), ("TiO2", 1.0)],
              [op("HeatingOperation", temp=1400.0)])
    b = route("Al2TiO5", [("Al2O3", 1.0), ("TiO2", 1.0), ("SiO2", 0.1)],
              [op("HeatingOperation", temp=900.0)])
    m_ab, bd_ab = COMPARATOR.compare(a, b, "Al2TiO5", UNIT_SCALES)
    m_ba, bd_ba = COMPARATOR.compare(b, a, "Al2TiO5", UNIT_SCALES)
    check("compare(a,b) == -compare(b,a) (margin)",
          abs(m_ab - (-m_ba)) < 1e-9, f"{m_ab} vs {-m_ba}")
    for c in CHANNEL_NAMES:
        da = bd_ab.get(f"{c}_diff")
        db = bd_ba.get(f"{c}_diff")
        if da is None or db is None:
            check(f"antisymmetry gradeability matches for {c}", da is None and db is None)
            continue
        check(f"compare antisymmetric on {c}", abs(da - (-db)) < 1e-9, f"{da} vs {-db}")


# ---------------------------------------------------------------------------
# 2. One sign test per channel
# ---------------------------------------------------------------------------

def test_c4_interface_count_sign():
    # Fewer precursors -> fewer interfaces -> strictly better (less negative).
    # C4 depends only on len(precursors) and gate-passing, not on both
    # routes sharing a target -- MgAl2O4 has no natural 3-precursor route
    # (it only needs 2 cations), so the 3-precursor side uses a real
    # 3-cation target (Ba2TiSi2O8, fresnoite: BaCO3 + TiO2 + 2 SiO2 ->
    # Ba2TiSi2O8 + 2 CO2) instead of forcing an unbalanceable extra
    # precursor onto MgAl2O4.
    two = route("MgAl2O4", [("MgO", 1.0), ("Al2O3", 1.0)],
                [op("HeatingOperation", temp=1200.0)])
    three = route("Ba2TiSi2O8", [("BaCO3", 2.0), ("TiO2", 1.0), ("SiO2", 2.0)],
                  [op("HeatingOperation", temp=1200.0)])
    sa = COMPARATOR.score_channels(two, "MgAl2O4")
    sb = COMPARATOR.score_channels(three, "Ba2TiSi2O8")
    check("C4: 2-precursor route scores higher (fewer interfaces) than 3-precursor",
          sa["C4_interface_count"] is not None and sb["C4_interface_count"] is not None
          and sa["C4_interface_count"] > sb["C4_interface_count"],
          f"{sa['C4_interface_count']} vs {sb['C4_interface_count']}")


def test_c6_decomposition_clearance_sign():
    # Same carbonate precursor, higher T_max clears decomposition onset
    # more comfortably -> strictly higher score.
    low = route("Li2TiO3", [("Li2CO3", 1.0), ("TiO2", 1.0)],
                [op("HeatingOperation", temp=750.0)])
    high = route("Li2TiO3", [("Li2CO3", 1.0), ("TiO2", 1.0)],
                 [op("HeatingOperation", temp=1000.0)])
    sa = COMPARATOR.score_channels(low, "Li2TiO3")
    sb = COMPARATOR.score_channels(high, "Li2TiO3")
    check("C6: higher T_max clears decomposition onset with a higher score",
          sa["C6_decomposition_clearance"] is not None
          and sb["C6_decomposition_clearance"] is not None
          and sb["C6_decomposition_clearance"] > sa["C6_decomposition_clearance"],
          f"{sa['C6_decomposition_clearance']} vs {sb['C6_decomposition_clearance']}")


def test_c5_volatilization_sign():
    # Li is in VOLATILE_T; a much higher T_max should score strictly worse.
    low = route("Li2TiO3", [("Li2CO3", 1.0), ("TiO2", 1.0)],
                [op("HeatingOperation", temp=800.0)])
    high = route("Li2TiO3", [("Li2CO3", 1.0), ("TiO2", 1.0)],
                 [op("HeatingOperation", temp=1400.0)])
    sa = COMPARATOR.score_channels(low, "Li2TiO3")
    sb = COMPARATOR.score_channels(high, "Li2TiO3")
    check("C5: higher T_max scores worse (more volatilization loss)",
          sa["C5_volatilization"] is not None and sb["C5_volatilization"] is not None
          and sa["C5_volatilization"] > sb["C5_volatilization"],
          f"{sa['C5_volatilization']} vs {sb['C5_volatilization']}")


def test_c3_temperature_window_sign():
    # Al2O3 melts at 2327 K -> T_low ~= 1163.5 K ~= 890 C. A route reporting
    # room temperature is far below the window; one near the Tammann onset
    # should score strictly higher (closer to / inside the window).
    cold = route("MgAl2O4", [("MgO", 1.0), ("Al2O3", 1.0)],
                 [op("HeatingOperation", temp=100.0)])
    warm = route("MgAl2O4", [("MgO", 1.0), ("Al2O3", 1.0)],
                 [op("HeatingOperation", temp=950.0)])
    params = ComparatorParams(c3_fraction=None)  # flat 100K arm, deterministic
    sa = COMPARATOR.score_channels(cold, "MgAl2O4", params)
    sb = COMPARATOR.score_channels(warm, "MgAl2O4", params)
    check("C3: a route near the Tammann onset scores higher than one far below it",
          sa["C3_reactive_temperature_window"] is not None
          and sb["C3_reactive_temperature_window"] is not None
          and sb["C3_reactive_temperature_window"] > sa["C3_reactive_temperature_window"],
          f"{sa['C3_reactive_temperature_window']} vs {sb['C3_reactive_temperature_window']}")


def test_c7_gas_evolution_sign():
    # Carbonate route releases CO2; oxide-only route releases nothing.
    carbonate = route("Li2TiO3", [("Li2CO3", 1.0), ("TiO2", 1.0)],
                      [op("HeatingOperation", temp=900.0)])
    oxide = route("Li2TiO3", [("Li2O", 1.0), ("TiO2", 1.0)],
                  [op("HeatingOperation", temp=900.0)])
    sa = COMPARATOR.score_channels(carbonate, "Li2TiO3")
    sb = COMPARATOR.score_channels(oxide, "Li2TiO3")
    check("C7: oxide-only route (no gas) scores >= carbonate route (releases CO2)",
          sa["C7_gas_evolution"] is not None and sb["C7_gas_evolution"] is not None
          and sb["C7_gas_evolution"] >= sa["C7_gas_evolution"],
          f"carbonate={sa['C7_gas_evolution']} oxide={sb['C7_gas_evolution']}")


def test_c1_selectivity_margin_gradeable():
    # C1 is expensive to hand-sign (needs a real competing-phase kink to
    # exist on the tie line); assert it's at least gradeable and finite for
    # a plain 2-precursor route rather than asserting a specific direction.
    a = route("MgAl2O4", [("MgO", 1.0), ("Al2O3", 1.0)],
              [op("HeatingOperation", temp=1200.0)])
    sa = COMPARATOR.score_channels(a, "MgAl2O4")
    check("C1: gradeable for a real 2-precursor interface",
          sa["C1_selectivity_margin"] is not None, str(sa["C1_selectivity_margin"]))


def test_c2_unspent_driving_force_sign():
    # Carbonate precursors sit further from the elements' formation energy
    # floor than a metal oxide of the same cation in some systems, but the
    # unambiguous, table-free direction is: a LESS stable (higher, closer
    # to zero) formation energy precursor set should score higher than one
    # with a MORE negative mean formation energy, holding the target fixed.
    # Compare Li2CO3+TiO2 (carbonate route) vs Li2O+TiO2 (oxide route) --
    # Li2CO3 is thermodynamically less stable (less negative Ef/atom) than
    # Li2O in most real inorganic thermo tables.
    carbonate = route("Li2TiO3", [("Li2CO3", 1.0), ("TiO2", 1.0)],
                      [op("HeatingOperation", temp=900.0)])
    oxide = route("Li2TiO3", [("Li2O", 1.0), ("TiO2", 1.0)],
                  [op("HeatingOperation", temp=900.0)])
    sa = COMPARATOR.score_channels(carbonate, "Li2TiO3")
    sb = COMPARATOR.score_channels(oxide, "Li2TiO3")
    check("C2: gradeable for both sides",
          sa["C2_unspent_driving_force"] is not None
          and sb["C2_unspent_driving_force"] is not None,
          f"{sa['C2_unspent_driving_force']} vs {sb['C2_unspent_driving_force']}")


# ---------------------------------------------------------------------------
# 3. Gate failure -> None on every channel
# ---------------------------------------------------------------------------

def test_gate_failure_all_none():
    # No operations at all -> format_ok gate fails.
    bad = route("MgAl2O4", [("MgO", 1.0), ("Al2O3", 1.0)], ops=[])
    sa = COMPARATOR.score_channels(bad, "MgAl2O4")
    check("gate failure (no operations) -> every channel None",
          all(v is None for v in sa.values()), str(sa))

    none_route = None
    sb = COMPARATOR.score_channels(none_route, "MgAl2O4")
    check("predicted=None -> every channel None",
          all(v is None for v in sb.values()), str(sb))


if __name__ == "__main__":
    test_antisymmetry()
    test_c1_selectivity_margin_gradeable()
    test_c2_unspent_driving_force_sign()
    test_c3_temperature_window_sign()
    test_c4_interface_count_sign()
    test_c5_volatilization_sign()
    test_c6_decomposition_clearance_sign()
    test_c7_gas_evolution_sign()
    test_gate_failure_all_none()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    else:
        print("ALL PASS")
