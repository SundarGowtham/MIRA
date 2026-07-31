"""
build_rl_dataset.py
-------------------
Builds data/rl/ — the dedicated RL prompt dataset — from the re-triaged
Kononova universe (misc/kononova_triage_results3.json).

Design (decided 2026-07-30, see KIMI_SESSION_WRITEUP.md + discussion):
  - Universe: unique targets from the current triage (8,588).
  - EXCLUDES every target in data/sft_v3/{train,val,test}.jsonl so RL
    improvement on these targets is unconfounded by imitation
    ("in-distribution novel", not true OOD — true OOD is eval-side).
  - NO p̂ pre-filtering: with continuous rewards the dead-group waste is
    ~20-30%, and the first RL run doubles as the on-policy probe
    (generation dump + frac_reward_zero_std monitoring). p̂ labels from
    the 125-target pilot probe are annotated where available.
  - Prompts regenerated with the CURRENT validator (SYSTEM_MSG +
    CLOSED_BOOK_USER + live PD stability data), pinning the prompt
    distribution to this validator/MP snapshot.

Outputs:
  data/rl/train.jsonl, data/rl/val.jsonl (200 held-out targets),
  data/rl/rl_train.jsonl -> train.jsonl / rl_val.jsonl -> val.jsonl
  (symlinks matching the trainer's <prefix>_{train,val}.jsonl convention),
  data/rl/manifest.json (provenance, composition, caveats).

Usage:
  PYTHONPATH=. uv run python data_curation/build_rl_dataset.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pymatgen.core import Composition  # noqa: E402

from core.reward import load_validator  # noqa: E402
from split_dataset import chemistry_class  # noqa: E402
from stratified_difficulty_eval import (  # noqa: E402
    SYSTEM_MSG,
    CLOSED_BOOK_USER,
    get_stability_data_sync,
)
from validator import VALIDATOR_VERSION  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--triage-results", type=Path,
                   default=Path("misc/kononova_triage_results3.json"))
    p.add_argument("--probe-results", type=Path,
                   default=Path("misc/support_probe_results.json"),
                   help="optional pilot probe for p-hat annotation")
    p.add_argument("--sft-dir", type=Path, default=Path("data/sft_v3"))
    p.add_argument("--out-dir", type=Path, default=Path("data/rl"))
    p.add_argument("--formula-set", type=Path,
                   default=Path("data/cache/mp_formula_set.pkl"))
    p.add_argument("--pd-index", type=Path,
                   default=Path("data/cache/pd_index.json"))
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--val-targets", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # --- universe: unique targets from triage, annotated with stratum ---
    triage = json.load(args.triage_results.open())["records"]
    universe: dict[str, dict] = {}
    for r in triage:
        if r.get("status") != "graded":
            continue
        t = r["target"]
        if t not in universe:
            universe[t] = {
                "target": t,
                "stratum": ("frac" if r.get("is_fractional") else "int")
                           + "x" + r.get("thermo_tier", "unknown"),
            }
    print(f"triage universe: {len(universe)} unique targets")

    # --- exclude every SFT target (unconfounded RL claim) ---
    sft_targets = set()
    for split in ("train", "val", "test"):
        p = args.sft_dir / f"{split}.jsonl"
        if p.exists():
            sft_targets |= {json.loads(l)["target"] for l in p.open() if l.strip()}
    n_overlap = len(set(universe) & sft_targets)
    universe = {t: r for t, r in universe.items() if t not in sft_targets}
    print(f"excluded {n_overlap} SFT targets -> {len(universe)} remain")

    # --- drop unparseable targets (typo-formula class) ---
    kept = {}
    for t, r in universe.items():
        try:
            Composition(t)
            kept[t] = r
        except Exception:
            pass
    print(f"dropped {len(universe) - len(kept)} unparseable targets -> {len(kept)}")
    universe = kept

    # --- optional p-hat annotation from the pilot probe ---
    p_hats: dict[str, dict] = {}
    if args.probe_results.exists():
        for r in json.load(args.probe_results.open())["per_target"]:
            p_hats[r["target"]] = {
                "p_hat_ge_0.65": r["p_hat_ge_0.65"],
                "p_hat_ge_0.9": r["p_hat_ge_0.9"],
            }
    n_annotated = len(set(universe) & set(p_hats))
    print(f"p-hat annotation available for {n_annotated} targets (pilot probe)")

    # --- split ---
    targets = sorted(universe)
    random.shuffle(targets)
    val_targets = set(targets[: args.val_targets])
    train_targets = targets[args.val_targets:]
    print(f"split: train={len(train_targets)} val={len(val_targets)}")

    # --- prompts with the CURRENT validator (pins prompt distribution) ---
    print("regenerating prompts with current validator (slow, ~1h for full set)...")
    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        sys.exit("FATAL: thermo_checker is None - check --pd-index path.")

    def to_record(t: str) -> dict:
        stability_text, _ = get_stability_data_sync(t, validator)
        user_msg = CLOSED_BOOK_USER.format(
            target=t, context="", stability_data=stability_text)
        rec = {
            "prompt": SYSTEM_MSG + "\n\n" + user_msg,
            "target": t,
            "stratum": universe[t]["stratum"],
            "chemistry_class": chemistry_class(t),
            "source": "kononova_triage_results3",
            "validator_version": VALIDATOR_VERSION,
        }
        rec.update(p_hats.get(t, {}))
        return rec

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(target_list, name):
        path = args.out_dir / name
        n = 0
        with path.open("w") as f:
            for t in target_list:
                f.write(json.dumps(to_record(t), ensure_ascii=False) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"  {name}: {n}/{len(target_list)}", flush=True)
        print(f"  wrote {n} -> {path}")

    write_jsonl(train_targets, "train.jsonl")
    write_jsonl(sorted(val_targets), "val.jsonl")

    for link, target_file in [("rl_train.jsonl", "train.jsonl"),
                              ("rl_val.jsonl", "val.jsonl")]:
        lp = args.out_dir / link
        lp.unlink(missing_ok=True)
        lp.symlink_to(target_file)

    manifest = {
        "validator_version": VALIDATOR_VERSION,
        "source_triage": str(args.triage_results),
        "prompt_format": "SYSTEM_MSG + CLOSED_BOOK_USER(live PD stability data)",
        "counts": {"train": len(train_targets), "val": len(val_targets)},
        "sft_overlap_excluded": n_overlap,
        "strata": dict(Counter(r["stratum"] for r in universe.values())),
        "chemistry": dict(Counter(chemistry_class(t) for t in universe)),
        "p_hat_annotation": {
            "targets_with_pilot_p_hat": n_annotated,
            "note": "pilot probe = 125 targets x 24 samples @ T=1.0; all other "
                    "targets have NO p-hat label by design (on-policy probe "
                    "during the first RL run instead of a 3-day offline probe).",
        },
        "caveats": [
            "Prompt PD context pinned to current MP snapshot + validator "
            f"version {VALIDATOR_VERSION}; regenerate if either changes.",
            "Dead-group waste (p~0 or p~1 targets) is accepted (~20-30% est. "
            "from pilot) in exchange for skipping the scaled offline probe; "
            "watch frac_reward_zero_std during training.",
            "Targets are in-distribution-novel (same Kononova distribution, "
            "unseen in SFT), NOT true OOD. True OOD = eval-side variants.",
        ],
    }
    with (args.out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote manifest -> {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
