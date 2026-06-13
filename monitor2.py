#!/usr/bin/env python3
"""
Analyze a JSONL file containing synthesis route data.
Computes statistics on validator_score and all numeric breakdown fields,
tracks thermodynamic_favorable == 0.5 (error case), shows last 50 entries,
average text lengths, and outliers (>2σ).
"""

import json
import sys
import statistics
from collections import defaultdict, deque

def analyze_jsonl(filepath):
    # Data containers
    last_50 = deque(maxlen=50)
    validator_scores = []          # list of (target, score)
    thinking_lengths = []
    reasoning_lengths = []
    thermodynamic_half_entries = []   # list of (target, line_no)
    
    # Breakdown stats: key -> list of (target, value)   (value is numeric)
    breakdown_data = defaultdict(list)

    line_no = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_no += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_no} invalid JSON: {e}", file=sys.stderr)
                continue

            target = record.get('target', '?')
            val_score = record.get('validator_score')
            if isinstance(val_score, (int, float)):
                validator_scores.append((target, val_score))

            # Lengths
            think_len = len(record.get('thinking', ''))
            reason_len = len(record.get('reasoning_raw', ''))
            thinking_lengths.append(think_len)
            reasoning_lengths.append(reason_len)

            # Breakdown fields (numeric only)
            breakdown = record.get('validator_breakdown', {})
            # List of expected numeric keys (including the temperature and dG)
            numeric_keys = [
                "stoichiometry", "charge_neutrality", "precursors_exist",
                "operation_order", "temperature_plausible", "thermodynamic_favorable",
                "thermodynamic_T_K", "thermodynamic_dG_eV_atom", "target_stability",
                "chempot_atmosphere", "target_match"
            ]
            for key in numeric_keys:
                if key in breakdown:
                    val = breakdown[key]
                    if isinstance(val, (int, float)):
                        breakdown_data[key].append((target, val))
                    # else skip non-numeric (should not happen)

            # Special tracking: thermodynamic_favorable == 0.5
            thermo_fav = breakdown.get('thermodynamic_favorable')
            if thermo_fav == 0.5:
                thermodynamic_half_entries.append((target, line_no))

            # Store summary for last 50 display
            last_50.append({
                'line': line_no,
                'target': target,
                'validator_score': val_score,
                'thermo_fav': thermo_fav,
                'think_len': think_len,
                'reason_len': reason_len
            })

    # --- Overall counts ---
    total_entries = len(thinking_lengths)
    if total_entries == 0:
        print("No valid entries found.")
        return

    # Validator scores (non-None)
    valid_scores = [score for (_, score) in validator_scores]
    if not valid_scores:
        print("No valid validator_score values found.")
        return

    # Stats for validator_score
    mean_score = statistics.mean(valid_scores)
    median_score = statistics.median(valid_scores)
    last_25_scores = [score for (_, score) in validator_scores[-25:]]
    rolling_25_mean = statistics.mean(last_25_scores) if last_25_scores else None
    if len(valid_scores) >= 2:
        stdev_score = statistics.stdev(valid_scores)
        outliers_score = [(target, score) for (target, score) in validator_scores
                          if abs(score - mean_score) > 2 * stdev_score]
    else:
        stdev_score = 0.0
        outliers_score = []

    # --- Breakdown statistics for each key ---
    breakdown_stats = {}
    for key, items in breakdown_data.items():
        values = [v for (_, v) in items]
        if not values:
            continue
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        last_25_vals = [v for (_, v) in items[-25:]]
        rolling_mean = statistics.mean(last_25_vals) if last_25_vals else None
        if len(values) >= 2:
            stdev_val = statistics.stdev(values)
            outliers_val = [(target, v) for (target, v) in items
                            if abs(v - mean_val) > 2 * stdev_val]
        else:
            stdev_val = 0.0
            outliers_val = []
        breakdown_stats[key] = {
            'mean': mean_val,
            'median': median_val,
            'rolling_mean_25': rolling_mean,
            'stdev': stdev_val,
            'outliers': outliers_val,
            'count': len(values)
        }

    # --- Print Report ---
    print("=" * 80)
    print(f"ANALYSIS REPORT for: {filepath}")
    print("=" * 80)

    # 1. Last 50 entries
    print("\n--- LAST 50 ENTRIES (most recent first) ---")
    for entry in reversed(last_50):
        print(f"Line {entry['line']:5d} | Target: {entry['target']:12s} | "
              f"val_score: {entry['validator_score']} | thermo_fav: {entry['thermo_fav']} | "
              f"think_len: {entry['think_len']:4d} | reason_len: {entry['reason_len']:4d}")

    # 2. Text lengths
    avg_think = statistics.mean(thinking_lengths) if thinking_lengths else 0
    avg_reason = statistics.mean(reasoning_lengths) if reasoning_lengths else 0
    print("\n--- TEXT LENGTHS ---")
    print(f"Average length of 'thinking': {avg_think:.1f} characters")
    print(f"Average length of 'reasoning_raw': {avg_reason:.1f} characters")

    # 3. Overall validator_score stats
    print("\n--- OVERALL VALIDATOR_SCORE STATISTICS ---")
    print(f"Total entries with score: {len(valid_scores)}")
    print(f"Overall mean: {mean_score:.4f}")
    print(f"Median: {median_score:.4f}")
    print(f"Standard deviation: {stdev_score:.4f}")
    if rolling_25_mean is not None:
        print(f"Rolling mean (last 25 entries): {rolling_25_mean:.4f}")
    if outliers_score:
        print("\nOutliers (|score - mean| > 2σ):")
        for target, score in outliers_score:
            print(f"  Target: {target:20s} | score: {score:.4f} | deviation: {abs(score - mean_score):.4f}")
    else:
        print("No outliers in validator_score.")

    # 4. Breakdown fields stats
    print("\n--- VALIDATOR BREAKDOWN FIELDS STATISTICS ---")
    # Order keys nicely
    ordered_keys = ["stoichiometry", "charge_neutrality", "precursors_exist", "operation_order",
                    "temperature_plausible", "thermodynamic_favorable", "thermodynamic_T_K",
                    "thermodynamic_dG_eV_atom", "target_stability", "chempot_atmosphere", "target_match"]
    for key in ordered_keys:
        if key not in breakdown_stats:
            continue
        stats = breakdown_stats[key]
        print(f"\n{key}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std dev: {stats['stdev']:.4f}")
        if stats['rolling_mean_25'] is not None:
            print(f"  Rolling mean (last 25): {stats['rolling_mean_25']:.4f}")
        if stats['outliers']:
            print(f"  Outliers (>2σ):")
            for target, val in stats['outliers']:
                print(f"    Target: {target:20s} | value: {val:.4f} | deviation: {abs(val - stats['mean']):.4f}")
        else:
            print("  No outliers.")

    # 5. Thermodynamic_favorable == 0.5 (error case)
    print("\n--- THERMODYNAMIC_FAVORABLE == 0.5 (error case) ---")
    if thermodynamic_half_entries:
        print(f"Count: {len(thermodynamic_half_entries)}")
        print("Targets (with line numbers):")
        for target, ln in thermodynamic_half_entries:
            print(f"  Line {ln:5d}: {target}")
    else:
        print("No entries found with thermodynamic_favorable == 0.5.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_synthesis.py <path_to_file.jsonl>")
        sys.exit(1)
    analyze_jsonl(sys.argv[1])