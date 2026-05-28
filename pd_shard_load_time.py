import pickle, time
from pathlib import Path
import statistics

shards = sorted(Path("data/cache/pd_shards").glob("*.pkl"))
print(f"Total shards: {len(shards)}")

times = []
for i, p in enumerate(shards):
    t0 = time.time()
    with p.open("rb") as f:
        pickle.load(f)
    times.append(time.time() - t0)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(shards)}: median={statistics.median(times):.3f}s p95={sorted(times)[int(0.95*len(times))]:.3f}s")

print(f"\nFinal: median={statistics.median(times):.3f}s")
print(f"       p95={sorted(times)[int(0.95*len(times))]:.3f}s")
print(f"       max={max(times):.3f}s")
print(f"       total={sum(times):.1f}s")