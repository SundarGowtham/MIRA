from __future__ import annotations
import argparse
import json, re
import os
import statistics
from pathlib import Path
from rich import print as rprint
from core.reward import parse_completion, ParseFailure, load_validator



def main():
    
    formula_set = Path("data/cache/mp_formula_set.pkl")
    pd_index = Path("data/cache/pd_index.json")
    project_root = os.getcwd()
    

    pd_index = pd_index if pd_index and pd_index.exists() else None
    validator = load_validator(formula_set, pd_index, project_root)


    with open('eval_results/eval_v3-rank16-seed42_test.json') as f:
        data = json.load(f)


    records = data["records"]

    for record in records[:1]:
        target = record["target"]
        completion = record["completion"]

        try:
            route = parse_completion(completion, target)
        except ParseFailure as e:
            print(f"  [parse_fail] idx={record.get('idx')} target={target}: {e}")
            continue

        # rprint(route)

        try:
            reward, breakdown = validator.val_debug(route, target)
        except Exception as e:
            rprint(str(e))
            rprint(f"  [validate_fail] idx={record.get('idx')} target={target}: {e}")
            continue

        # rprint(f"reward: {reward}")
        # rprint(f"reward: {breakdown}")




if __name__ == "__main__":
    main()