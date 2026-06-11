#!/usr/bin/env python3
"""
Analyze a JSONL file containing synthesis route data.
Computes statistics on validator_score, lengths, tracks thermodynamic_favorable=0.5,
and shows last 50 entries.
"""

import json
import sys
import statistics
from collections import deque

def analyze_jsonl(filepath):
    # Data containers
    last_50 = deque(maxlen=50)
    validator_scores = []      # list of (target, score)
    thinking_lengths = []
    thermodynamic_half_entries = []   # list of (target, line_no)

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
            thinking_lengths.append(think_len)

            # Thermodynamic_favorable == 0.5 check
            breakdown = record.get('validator_breakdown', {})
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

    # --- Basic counts ---
    total_entries = len(thinking_lengths)
    if total_entries == 0:
        print("No valid entries found.")
        return

    # Validator scores (non-None)
    valid_scores = [score for (_, score) in validator_scores]
    if not valid_scores:
        print("No valid validator_score values found.")
        return

    # Stats
    mean_score = statistics.mean(valid_scores)
    median_score = statistics.median(valid_scores)
    # Rolling mean of last 25 (most recent entries)
    last_25_scores = [score for (_, score) in validator_scores[-25:]]
    rolling_25_mean = statistics.mean(last_25_scores) if last_25_scores else None

    # Outliers (more than 2 std from mean)
    if len(valid_scores) >= 2:
        stdev = statistics.stdev(valid_scores)
        outliers = [(target, score) for (target, score) in validator_scores
                    if abs(score - mean_score) > 2 * stdev]
    else:
        stdev = 0.0
        outliers = []

    # --- Print Report ---
    print("=" * 80)
    print(f"ANALYSIS REPORT for: {filepath}")
    print("=" * 80)

    # Last 50 entries
    print("\n--- LAST 50 ENTRIES (most recent first) ---")
    # Display in reverse order (most recent last in deque, so reverse)
    for entry in reversed(last_50):
        print(f"Line {entry['line']:5d} | Target: {entry['target']:12s} | "
              f"val_score: {entry['validator_score']} | thermo_fav: {entry['thermo_fav']} | "
              f"think_len: {entry['think_len']:4d} | reason_len: {entry['reason_len']:4d}")

    # Average lengths
    avg_think = statistics.mean(thinking_lengths) if thinking_lengths else 0
    print("\n--- TEXT LENGTHS ---")
    print(f"Average length of 'thinking': {avg_think:.1f} characters")

    # Thermodynamic score
    print("\n--- VALIDATOR_SCORE STATISTICS ---")
    print(f"Total entries with score: {len(valid_scores)}")
    print(f"Overall mean: {mean_score:.4f}")
    print(f"Median: {median_score:.4f}")
    print(f"Standard deviation: {stdev:.4f}")
    if rolling_25_mean is not None:
        print(f"Rolling mean (last 25 entries): {rolling_25_mean:.4f}")

    # Outliers
    if outliers:
        print("\n--- OUTLIERS (|score - mean| > 2σ) ---")
        for target, score in outliers:
            print(f"  Target: {target:20s} | score: {score:.4f} | deviation: {abs(score - mean_score):.4f}")
    else:
        print("\nNo outliers found (none > 2σ).")

    # Track thermodynamic_favorable == 0.5
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