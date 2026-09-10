import json, sys
data = json.loads(open(sys.argv[1]).read())
results = data["results"]["3_hard"]
n = len(results)
n_error = sum(1 for r in results if r["breakdown"].get("error") == 1.0)
n_real_zero = sum(1 for r in results if r["reward"] < 0.05 and r["breakdown"].get("error") != 1.0)
n_other = n - n_error - n_real_zero
print(f"n={n}")
print(f"parse/validate exceptions (mechanical failure): {n_error} ({n_error/n:.1%})")
print(f"real computed near-zero score (genuine bad chemistry): {n_real_zero} ({n_real_zero/n:.1%})")
print(f"everything else (partial-to-full credit): {n_other} ({n_other/n:.1%})")
print()
print("worst 5 by reward:")
for r in sorted(results, key=lambda r: r["reward"])[:5]:
    print(f"  {r['target']:<24} reward={r['reward']:.4f}  breakdown={r['breakdown']}")