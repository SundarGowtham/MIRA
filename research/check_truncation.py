import json, sys

data = json.loads(open(sys.argv[1]).read())

for tier_name, results in data["results"].items():
    failures = [r for r in results if r["breakdown"].get("error_type") == "ParseFailure"]
    if not failures:
        continue
    print(f"\n=== {tier_name}: {len(failures)} ParseFailure records ===")
    for r in failures:
        completion = r.get("completion", "")
        n_chars = len(completion)
        tail = completion[-150:].replace("\n", "\\n")
        has_closing_brace = completion.rstrip().endswith("}")
        has_think_close = "</think>" in completion
        print(f"  {r['target']:<28} len={n_chars:<7} ends_with_}}={has_closing_brace}  "
              f"has_</think>={has_think_close}")
        print(f"    tail: ...{tail}")