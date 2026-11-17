#!/usr/bin/env python
"""
research/phase20_verify_samples.py -- accept a Phase 20 sampling stage's
output files, whether drawn on this machine or on a rented GPU.

For every output of the job (data_curation/phase20_sampling.py JOBS) checks:
  - the file exists and every target has exactly the expected number of rows
    (32, or 200 "reference" + 32 "resampling_rate"), no extra targets;
  - every row's "meta" matches the spec: temperature 0.9, top_p 0.95,
    max_tokens 8192, seed 42, backend vllm, bf16, base Qwen/Qwen3-8B, the
    expected adapter path, the vLLM version of this machine's mira-vllm env
    (0.23.0), and -- when the adapter exists locally -- its sha256.
Records the GPU name(s) found. Exit 0 = accepted (the chain then writes the
stage marker); exit 1 = rejected, with the reason.

Usage: python research/phase20_verify_samples.py <job>   (any env; no GPU)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data_curation import phase20_sampling as S  # noqa: E402

EXPECTED_VLLM = "0.23.0"


def verify(job: str) -> tuple[bool, dict]:
    report, ok = {"job": job, "files": {}}, True
    sha_cache = {}
    for j in S.JOBS[job]:
        path = S.OUT / j["out"]
        r = {"path": str(path.relative_to(REPO_ROOT)), "errors": []}
        report["files"][j["out"]] = r
        if not path.exists():
            r["errors"].append("missing")
            ok = False
            continue
        adir = REPO_ROOT / S.ADAPTERS[j["model"]]
        if adir.exists() and j["model"] not in sha_cache:
            sha_cache[j["model"]] = S.adapter_sha256(adir)
        expect = {"temperature": S.TEMPERATURE, "top_p": S.TOP_P, "max_tokens": S.MAX_NEW_TOKENS,
                  "seed": S.SEED, "backend": "vllm", "dtype": "bfloat16", "base_model": S.BASE_MODEL,
                  "adapter": S.ADAPTERS[j["model"]], "model": j["model"], "vllm_version": EXPECTED_VLLM}
        per_target, groups, gpus, shas, bad = Counter(), Counter(), Counter(), Counter(), Counter()
        n = 0
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            per_target[row["target"]] += 1
            if j["kind"] == "reference":
                groups[(row["target"], row.get("sample_group"))] += 1
            meta = row.get("meta") or {}
            for k, v in expect.items():
                if meta.get(k) != v:
                    bad[f"{k}={meta.get(k)!r} (want {v!r})"] += 1
            gpus[meta.get("gpu")] += 1
            shas[meta.get("adapter_sha256")] += 1
        tl = S.targets(j["set"])
        per = S.REF_N + S.REF_RESAMPLE_N if j["kind"] == "reference" else S.N32
        wrong = {t: per_target.get(t, 0) for t in tl if per_target.get(t, 0) != per}
        extra = sorted(set(per_target) - set(tl))
        if wrong:
            r["errors"].append(f"row counts wrong for {len(wrong)} targets: {dict(list(wrong.items())[:5])}")
        if extra:
            r["errors"].append(f"unexpected targets: {extra[:5]}")
        if j["kind"] == "reference":
            badg = [t for t in tl if groups[(t, "reference")] != S.REF_N
                    or groups[(t, "resampling_rate")] != S.REF_RESAMPLE_N]
            if badg:
                r["errors"].append(f"reference grouping wrong for {len(badg)} targets")
        if bad:
            r["errors"].append(f"meta mismatches: {dict(bad.most_common(5))}")
        if len(shas) != 1:
            r["errors"].append(f"{len(shas)} different adapter hashes in one file")
        if j["model"] in sha_cache and set(shas) != {sha_cache[j["model"]]}:
            r["errors"].append("adapter sha256 differs from the local adapter")
        r.update(n_rows=n, expected_rows=per * len(tl), gpus=dict(gpus),
                 adapter_sha256=list(shas), local_adapter_present=adir.exists())
        if r["errors"]:
            ok = False
    report["accepted"] = ok
    return ok, report


def main():
    job = sys.argv[1]
    ok, rep = verify(job)
    out = REPO_ROOT / "runs" / "phase20" / f"verify_{job}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    for name, r in rep["files"].items():
        print(f"  {name}: {'OK' if not r['errors'] else 'REJECT ' + '; '.join(r['errors'])}"
              f"  rows={r.get('n_rows')}/{r.get('expected_rows')} gpus={r.get('gpus')}")
    print(f"[verify {job}] {'ACCEPTED' if ok else 'REJECTED'} -> {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
