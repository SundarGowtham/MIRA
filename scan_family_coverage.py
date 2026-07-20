"""
scan_family_coverage.py
--------------------------
Checks real representation of major materials families in the raw
Kononova corpus (17,616 targets) rather than guessing from a list.
Classification is composition-pattern matching (element sets + rough
stoichiometric ratios with tolerance for doping variation) - not exact
crystal-structure identification, since that would need the actual
structure, not just the formula. Treat a "match" as "this target's
composition is consistent with the family," not a certified structural
classification.

Cross-references against kononova_triage_results.json (if provided) for
gradeability tier, so you see not just "how many NMC-family targets exist"
but "how many are actually gradeable" - the thing that matters for
whether they're usable RL/SFT data, not just whether they're present.

Usage:
  uv run python scan_family_coverage.py \
      --synthesis data/raw/synthesis_clean.json \
      --triage-results kononova_triage_results.json \
      --examples-per-family 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def elements_of(comp) -> set[str]:
    return {str(el) for el in comp.elements}


def frac(comp, el: str) -> float:
    d = comp.as_dict()
    return d.get(el, 0.0)


def ratio_close(a: float, b: float, tol: float = 0.5) -> bool:
    """True if a/b (or b/a) is within [1-tol, 1+tol] of 1 - loose on
    purpose, since real doped/substituted compounds deviate a lot from
    idealized stoichiometry."""
    if a <= 0 or b <= 0:
        return False
    r = a / b
    return (1 - tol) <= r <= (1 + tol)


def make_families():
    """Each family: (name, description, classifier(comp, els) -> bool)."""
    F = []

    def nmc(c, e):
        tms = sum(1 for m in ("Ni", "Mn", "Co") if m in e)
        return "Li" in e and "O" in e and tms >= 2
    F.append(("NMC_cathode", "LiNixMnyCozO2 layered oxide cathode family", nmc))

    def nca(c, e):
        return {"Li", "Ni", "Co", "Al", "O"} <= e
    F.append(("NCA_cathode", "LiNixCoyAlzO2 cathode", nca))

    def lfp_lmp(c, e):
        return "Li" in e and "P" in e and "O" in e and ("Fe" in e or "Mn" in e) and "Ni" not in e and "Co" not in e
    F.append(("LFP_LMP_olivine", "LiFePO4 / LiMnPO4 olivine cathode", lfp_lmp))

    def lmo_spinel(c, e):
        return e == {"Li", "Mn", "O"}
    F.append(("LMO_spinel", "LiMn2O4 spinel cathode (Mn-only)", lmo_spinel))

    def lco(c, e):
        return e == {"Li", "Co", "O"}
    F.append(("LCO", "LiCoO2 (Co-only layered oxide)", lco))

    def lto(c, e):
        return e == {"Li", "Ti", "O"}
    F.append(("LTO_anode", "Li4Ti5O12-type anode (Ti-only)", lto))

    def na_ion(c, e):
        tms = {"V", "Fe", "Mn", "Ni", "Co", "Cr"}
        return "Na" in e and "Li" not in e and "O" in e and bool(tms & e)
    F.append(("Na_ion_cathode", "Na-ion layered/polyanionic cathode", na_ion))

    def llzo_garnet(c, e):
        return {"Li", "La", "Zr", "O"} <= e
    F.append(("LLZO_garnet", "Li7La3Zr2O12-type garnet solid electrolyte", llzo_garnet))

    def yig_yag_garnet(c, e):
        if {"Y", "Fe", "O"} <= e and "Al" not in e:
            return ratio_close(frac(c, "Fe"), frac(c, "Y") * 5 / 3, tol=0.6)
        if {"Y", "Al", "O"} <= e and "Fe" not in e:
            return ratio_close(frac(c, "Al"), frac(c, "Y") * 5 / 3, tol=0.6)
        return False
    F.append(("YIG_YAG_garnet", "Y3Fe5O12 / Y3Al5O12 garnet (magnetic/phosphor host)", yig_yag_garnet))

    def nasicon(c, e):
        a_site = "Na" in e or "Li" in e
        m_site = "Ti" in e or "Zr" in e
        return a_site and m_site and "P" in e and "O" in e
    F.append(("NASICON", "NASICON-type ion conductor (A-M-P-O)", nasicon))

    def batio3_pzt(c, e):
        if e == {"Ba", "Ti", "O"}:
            return ratio_close(frac(c, "Ti"), frac(c, "Ba"), tol=0.4)
        if {"Pb", "Zr", "Ti", "O"} <= e:
            return True
        return False
    F.append(("Ferroelectric_perovskite", "BaTiO3 / PZT ferroelectric perovskite", batio3_pzt))

    def bifeo3(c, e):
        return e == {"Bi", "Fe", "O"} and ratio_close(frac(c, "Bi"), frac(c, "Fe"), tol=0.4)
    F.append(("BiFeO3_multiferroic", "BiFeO3-type multiferroic", bifeo3))

    def cmr_manganite(c, e):
        a_rare = bool({"La", "Pr", "Nd", "Sm"} & e)
        a_alkaline = bool({"Sr", "Ca", "Ba"} & e)
        return a_rare and a_alkaline and "Mn" in e and "O" in e
    F.append(("CMR_manganite", "(La,Pr,Nd)(Sr,Ca,Ba)MnO3 colossal magnetoresistance", cmr_manganite))

    def sofc(c, e):
        ysz = {"Y", "Zr", "O"} <= e and len(e) <= 4
        lsgm = {"La", "Sr", "Ga", "Mg", "O"} <= e
        lscf = {"La", "Sr", "Co", "Fe", "O"} <= e
        gdc_sdc = ("Gd" in e or "Sm" in e) and "Ce" in e and "O" in e
        return ysz or lsgm or lscf or gdc_sdc
    F.append(("SOFC_material", "YSZ / LSGM / LSCF / GDC-SDC fuel-cell material", sofc))

    def high_tc_sc(c, e):
        ybco = {"Y", "Ba", "Cu", "O"} <= e
        bscco = {"Bi", "Sr", "Ca", "Cu", "O"} <= e
        lsco = {"La", "Sr", "Cu", "O"} <= e and "Y" not in e
        return ybco or bscco or lsco
    F.append(("High_Tc_superconductor", "YBCO / BSCCO / LSCO cuprate superconductor", high_tc_sc))

    def spinel_ferrite(c, e):
        m = {"Ni", "Co", "Mn", "Zn", "Cu", "Mg"} & e
        return bool(m) and "Fe" in e and "O" in e and len(e) <= 3
    F.append(("Spinel_ferrite", "MFe2O4 spinel ferrite (M=Ni,Co,Mn,Zn,Cu,Mg)", spinel_ferrite))

    def hexaferrite(c, e):
        a = {"Ba", "Sr"} & e
        return bool(a) and "Fe" in e and "O" in e and ratio_close(frac(c, "Fe"), sum(frac(c, x) for x in a) * 12, tol=0.5)
    F.append(("Hexaferrite", "(Ba,Sr)Fe12O19 hard magnetic hexaferrite", hexaferrite))

    def skutterudite(c, e):
        return "Co" in e and "Sb" in e and "O" not in e
    F.append(("Skutterudite_TE", "CoSb3-type skutterudite thermoelectric", skutterudite))

    def half_heusler(c, e):
        chalc = bool({"Sb", "Bi", "Sn"} & e)
        return len(e) == 3 and chalc and "O" not in e
    F.append(("Half_Heusler_TE", "Half-Heusler thermoelectric (no oxygen, 3 elements)", half_heusler))

    def bite_pbte_family(c, e):
        return ({"Bi", "Te"} <= e or {"Sb", "Te"} <= e or {"Pb", "Te"} <= e) and "O" not in e
    F.append(("Chalcogenide_TE", "Bi2Te3 / Sb2Te3 / PbTe thermoelectric", bite_pbte_family))

    return F


def is_fractional(formula: str, tol: float = 1e-6) -> bool:
    from pymatgen.core import Composition
    try:
        amounts = Composition(formula).as_dict().values()
    except Exception:
        return False
    return any(abs(a - round(a)) > tol for a in amounts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthesis", type=Path, required=True)
    ap.add_argument("--triage-results", type=Path, default=None)
    ap.add_argument("--examples-per-family", type=int, default=8)
    args = ap.parse_args()

    from pymatgen.core import Composition

    records = json.loads(args.synthesis.read_text())
    print(f"Loaded {len(records)} raw Kononova records.")

    tier_lookup: dict[str, str] = {}
    if args.triage_results:
        if not args.triage_results.exists():
            print(f"\nWARNING: --triage-results path does not exist: {args.triage_results}", file=sys.stderr)
            print("Proceeding WITHOUT gradeability cross-reference - every family's "
                  "gradeable/discrete/interp/ungrad columns will show 0, which is NOT "
                  "a real finding, it means the file wasn't found. Fix the path and rerun.",
                  file=sys.stderr)
        else:
            triage = json.loads(args.triage_results.read_text())
            for r in triage.get("records", []):
                if r.get("status") == "graded":
                    tier_lookup[r["target"]] = r.get("thermo_tier", "unknown")
            print(f"Loaded {len(tier_lookup)} tier assignments from {args.triage_results}")
            if len(tier_lookup) == 0:
                print("WARNING: triage-results file loaded but contains ZERO graded records. "
                      "Gradeability columns below will be meaningless (all zero) - "
                      "this is a data problem, not a real finding.", file=sys.stderr)

    families = make_families()
    matches: dict[str, list[str]] = defaultdict(list)
    n_parsed = 0
    n_parse_fail = 0

    for rec in records:
        target = rec.get("target_formula")
        if not target:
            continue
        try:
            comp = Composition(target)
        except Exception:
            n_parse_fail += 1
            continue
        n_parsed += 1
        els = elements_of(comp)
        for name, _desc, classifier in families:
            try:
                if classifier(comp, els):
                    matches[name].append(target)
            except Exception:
                continue

    print(f"\n{n_parsed} targets parsed, {n_parse_fail} failed to parse.\n")
    print("=" * 78)
    print(f"{'family':<26}{'n_matches':<11}{'gradeable':<11}{'discrete':<10}{'interp':<8}{'ungrad':<8}")
    print("=" * 78)

    for name, desc, _classifier in families:
        targets = matches[name]
        n = len(targets)
        if n == 0:
            print(f"{name:<26}{'0':<11}  <-- ZERO representation in the raw corpus")
            continue

        tier_counts = Counter(tier_lookup.get(t, "not_in_triage") for t in targets)
        n_gradeable = tier_counts.get("discrete", 0) + tier_counts.get("interpolated", 0)
        print(f"{name:<26}{n:<11}{n_gradeable:<11}{tier_counts.get('discrete',0):<10}"
              f"{tier_counts.get('interpolated',0):<8}{tier_counts.get('ungradeable',0):<8}")

    print("\n" + "=" * 78)
    print("families with ZERO or very low (<5) representation - genuine gaps:")
    for name, desc, _classifier in families:
        n = len(matches[name])
        if n < 5:
            print(f"  {name:<26} n={n:<4} {desc}")

    print("\nexample targets per family (first {} matches):".format(args.examples_per_family))
    for name, desc, _classifier in families:
        targets = matches[name]
        if not targets:
            continue
        sample = targets[:args.examples_per_family]
        print(f"\n  {name} ({desc}):")
        for t in sample:
            tier = tier_lookup.get(t, "?")
            print(f"    {t:<32} tier={tier}")


if __name__ == "__main__":
    main()