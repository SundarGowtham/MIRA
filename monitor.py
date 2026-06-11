import json
import os
import sys
import time
from collections import deque
import math

# --- CONFIGURATION ---
# Replace with your actual JSONL file path
FILE_PATH = "data/processed/synthesis_with_traces.jsonl" 
ROLLING_WINDOW_SIZE = 25
OUTLIER_STD_THRESHOLD = 2.0

# --- DATA STRUCTURES ---
# Track overall list for total metrics
all_validator_scores = []
all_breakdown_averages = []

# Track rolling window
rolling_validator_scores = deque(maxlen=ROLLING_WINDOW_SIZE)
rolling_breakdown_averages = deque(maxlen=ROLLING_WINDOW_SIZE)

# Error tracking (0.5 scores)
error_targets = set()

def calculate_stats(data_list):
    """Returns (mean, median, std_dev) for a list of numbers."""
    if not data_list:
        return 0.0, 0.0, 0.0
    
    n = len(data_list)
    mean = sum(data_list) / n
    
    sorted_data = sorted(data_list)
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[(n // 2) - 1] + sorted_data[n // 2]) / 2.0
        
    variance = sum((x - mean) ** 2 for x in data_list) / n
    std_dev = math.sqrt(variance)
    
    return mean, median, std_dev

def process_line(line):
    """Parses a single JSONL line and updates global metrics."""
    global all_validator_scores, all_breakdown_averages, error_targets
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        return None  # Skip malformed lines Safely

    target = data.get("target", "Unknown")
    v_score = data.get("validator_score")
    breakdown = data.get("validator_breakdown", {})

    if v_score is None or not breakdown:
        return None

    # Calculate the average of all sub-scores inside validator_breakdown
    breakdown_values = [float(v) for v in breakdown.values()]
    if not breakdown_values:
        return None
    b_avg = sum(breakdown_values) / len(breakdown_values)

    # Check for the specific 0.5 error condition inside breakdown
    if 0.5 in breakdown_values:
        error_targets.add(target)

    # Append to global histories
    all_validator_scores.append(v_score)
    all_breakdown_averages.append(b_avg)

    # Append to rolling queues
    rolling_validator_scores.append(v_score)
    rolling_breakdown_averages.append(b_avg)

    # Evaluate outliers against the current historical baseline
    v_mean, _, v_std = calculate_stats(all_validator_scores[:-1])
    b_mean, _, b_std = calculate_stats(all_breakdown_averages[:-1])

    is_outlier = False
    # Only flag outliers if we have enough baseline data to establish standard deviation
    if len(all_validator_scores) > 10:
        v_outlier = v_std > 0 and abs(v_score - v_mean) > (OUTLIER_STD_THRESHOLD * v_std)
        b_outlier = b_std > 0 and abs(b_avg - b_mean) > (OUTLIER_STD_THRESHOLD * b_std)
        is_outlier = v_outlier or b_outlier

    return {
        "target": target,
        "validator_score": v_score,
        "breakdown_avg": b_avg,
        "is_outlier": is_outlier
    }

def print_dashboard(recent_entries):
    """Clears terminal screen and prints refreshed real-time metrics."""
    # Compute stats
    v_mean, v_med, v_std = calculate_stats(all_validator_scores)
    b_mean, b_med, b_std = calculate_stats(all_breakdown_averages)
    
    r_v_mean, r_v_med, _ = calculate_stats(list(rolling_validator_scores))
    r_b_mean, r_b_med, _ = calculate_stats(list(rolling_breakdown_averages))

    # Clear terminal
    os.system('clear' if os.name != 'nt' else 'cls')

    print("=" * 70)
    print(f" REAL-TIME LOG MONITOR | Total Rows Processed: {len(all_validator_scores)}")
    print("=" * 70)
    
    print("\n--- STATISTICAL METRICS ---")
    print(f"Validator Score  -> Total Mean: {v_mean:.4f} | Rolling 25 Mean: {r_v_mean:.4f} | Median: {v_med:.4f} | Std Dev: {v_std:.4f}")
    print(f"Breakdown Avg    -> Total Mean: {b_mean:.4f} | Rolling 25 Mean: {r_b_mean:.4f} | Median: {b_med:.4f} | Std Dev: {b_std:.4f}")
    
    print("\n--- ERROR TRACKING (Targets with a 0.5 breakdown score) ---")
    if error_targets:
        print(f"Count: {len(error_targets)} targets flagged.")
        print(f"Targets: {', '.join(sorted(list(error_targets)))}")
    else:
        print("None detected.")

    print("\n--- RECENT ENTRIES (Last 50 Max, * Indicates Outlier) ---")
    print(f"{'Target':<15} | {'Validator Score':<15} | {'Breakdown Avg':<15} | Status")
    print("-" * 70)
    
    for entry in list(recent_entries)[-50:]:
        status = "* OUTLIER *" if entry["is_outlier"] else "Normal"
        print(f"{entry['target']:<15} | {entry['validator_score']:<15.4f} | {entry['breakdown_avg']:<15.4f} | {status}")

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Waiting for target file '{FILE_PATH}' to be created...")
        while not os.path.exists(FILE_PATH):
            time.sleep(1)

    print(f"Monitoring {FILE_PATH}...")
    
    recent_entries = deque(maxlen=50)
    
    with open(FILE_PATH, 'r') as f:
        # Catch up on any existing file contents first
        for line in f:
            res = process_line(line)
            if res:
                recent_entries.append(res)
        
        print_dashboard(recent_entries)

        # Loop forever watching for new appends
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)  # Rest briefly if no new rows arrived
                continue
            
            res = process_line(line)
            if res:
                recent_entries.append(res)
                print_dashboard(recent_entries)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)
