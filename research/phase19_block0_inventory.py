#!/usr/bin/env python
"""
research/phase19_block0_inventory.py -- Phase 19 Block 0 (CPU).
docs/phases/PHASE19_PREREG.md "Block 0".

1. Probe strings -> results/phase19/probe_strings.jsonl
   Probe set: results/phase17/probe_targets.json (20 ASTRAL clean + 20 held out).
   Counts: canonical recipes (precset_of) from results/phase18/recipes_by_model.json
   (Base and RS-SFT 232 samples/target, others 32). Pooled count = sum over the
   8 step-table models (S2-S7). Top 30 recipes by pooled count (ties broken by
   recipe key) among recipes that have at least one literal string; a recipe
   with no raw-text sample anywhere (it can occur only in the 4 ASTRAL
   structured files that store no completion text) is skipped and listed.
   String = the recipe's most frequent literal precursor-array text '[...]'
   across all raw completions of these targets (the 8 models plus the Phase 12 /
   E1 @100/@200 raw files, used as string sources only, never for counts).
2. Adapters: path, r/alpha, target modules, and layer coverage (all 36 blocks)
   read from the adapter safetensors keys.
3. Training data:
   - S2/S3/S4/S5's SFT training files + the TRUNCATION actually applied in
     training (TRL SFTConfig max_length=1024, truncation_mode keep_start).
   - Probe-target overlap with E3 / E3c training targets (P-19D-3).
   - S6/S7: first 50 groups (steps 0-24, 400 completions) from
     runs/<run>/generations.jsonl, GDPO advantages recomputed exactly as TRL
     1.8.0 normalize_then_sum computes them (per-reward z within group with
     nanstd + 1e-4, weighted nansum, batch z over the 16-completion generation
     batch, NaN -> 0). Written to results/phase19/rl_update_groups_<step>.jsonl.
   - Whether E1's breakdown includes the ranker objectives.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

from research.phase19_common import (  # noqa: E402
    OUT_DIR, PROBE_STRINGS_PATH, MODELS, STEPS, TOP_N, N_LAYERS, provenance, write_json, notify, adapters, rkey,
)

PROBE = json.loads((REPO_ROOT / "results/phase17/probe_targets.json").read_text())
RAW_EXTRA = {
    "astral": ["data/novelty/phase12_ckpt100_astral_n32.jsonl", "data/novelty/phase12_ckpt200_astral_n32.jsonl",
               "data/novelty/e1_ckpt100_astral_n32.jsonl", "data/novelty/e1_ckpt200_astral_n32.jsonl"],
    "heldout": [],
}
RL_RUNS = {"S6": ("runs/gdpo-qlora-gdpo-phase12-rssft-beta0", "checker"),
           "S7": ("runs/gdpo-qlora-gdpo-e1-ranker-rssft", "ranker")}
SFT_TRAIN = {"S2": "data/rs_sft/rs_sft_train.jsonl", "S3": "data/e3/e3_train.jsonl",
             "S4": "data/e3c/e3c_train.jsonl", "S5_init_corpus_sft": "data/sft_v3/sft_train.jsonl"}
N_GROUPS = 50
GEN = 8


def literal_array(completion: str):
    j = completion.rfind("</think>")
    ans = completion[j:] if j >= 0 else completion
    k = ans.find('"precursors"')
    if k < 0:
        return None
    m = re.match(r'"precursors"\s*:\s*', ans[k:])
    if m is None:
        return None
    i = k + m.end()
    if i >= len(ans) or ans[i] != "[":
        return None
    depth = 0
    for e in range(i, len(ans)):
        if ans[e] == "[":
            depth += 1
        elif ans[e] == "]":
            depth -= 1
            if depth == 0:
                return ans[i:e + 1]
    return None


def build_probe_strings():
    from core.reward import parse_completion
    from support_growth_by_checkpoint import precset_of
    from research.phase18_common import load_recipes, recipes, SOURCES
    cache = load_recipes()
    targets = [(t, "astral") for t in PROBE["probe_astral_20"]] + [(t, "heldout") for t in PROBE["probe_heldout_20"]]
    tset = {t: s for t, s in targets}
    # literal strings per (target, recipe)
    forms = defaultdict(lambda: defaultdict(Counter))
    files = set()
    for m in MODELS:
        for ts in ("astral", "heldout"):
            kind, path, _ = SOURCES[(m, ts)][0]
            if kind == "raw":
                files.add((str(path), ts))
    for ts, lst in RAW_EXTRA.items():
        for p in lst:
            files.add((str(REPO_ROOT / p), ts))
    n_scanned = Counter()
    for path, ts in sorted(files):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                t = r["target"]
                if tset.get(t) != ts:
                    continue
                try:
                    route = parse_completion(r["completion"], t)
                    ps = precset_of(route) if route is not None else None
                except Exception:
                    ps = None
                if not ps:
                    continue
                s = literal_array(r["completion"])
                if s is None:
                    continue
                forms[t][tuple(ps)][s] += 1
                n_scanned[Path(path).name] += 1
    rows, skipped, per_target = [], {}, {}
    for t, ts in targets:
        counts = {m: Counter(recipes(cache, m, ts, t)) for m in MODELS}
        n_samples = {m: len(recipes(cache, m, ts, t)) for m in MODELS}
        pooled = Counter()
        for m in MODELS:
            pooled.update(counts[m])
        order = sorted(pooled, key=lambda y: (-pooled[y], rkey(y)))
        chosen, sk = [], []
        for y in order:
            if len(chosen) == TOP_N:
                break
            if forms[t].get(y):
                chosen.append(y)
            else:
                sk.append({"recipe": list(y), "pooled_count": pooled[y]})
        skipped[t] = sk
        per_target[t] = {"n_distinct_recipes_pooled": len(pooled), "n_chosen": len(chosen), "n_skipped_no_string": len(sk),
                         "n_samples_parsed": n_samples}
        for rank, y in enumerate(chosen):
            fc = forms[t][y]
            s, sc = sorted(fc.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            rows.append({"target": t, "target_set": ts, "rank": rank, "recipe": list(y), "recipe_key": rkey(y),
                         "string": s, "string_count": sc, "n_distinct_strings": len(fc),
                         "pooled_count": pooled[y],
                         "counts": {m: counts[m].get(y, 0) for m in MODELS},
                         "sampled": {m: counts[m].get(y, 0) > 0 for m in MODELS},
                         "n_samples_parsed": n_samples})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROBE_STRINGS_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows, skipped, per_target, dict(n_scanned)


def adapter_inventory():
    from safetensors import safe_open
    out = {}
    ad = adapters()
    need = sorted({m for st in STEPS.values() for m in st if m != "base"})
    for name in need:
        p = REPO_ROOT / ad[name]
        cfg = json.loads((p / "adapter_config.json").read_text())
        layers, mods = set(), set()
        with safe_open(str(p / "adapter_model.safetensors"), "pt") as f:
            keys = list(f.keys())
        for k in keys:
            mm = re.search(r"layers\.(\d+)\.(\w+)\.(\w+)\.lora_[AB]", k)
            if mm:
                layers.add(int(mm.group(1)))
                mods.add(mm.group(3))
        out[name] = {"path": ad[name], "r": cfg["r"], "lora_alpha": cfg["lora_alpha"],
                     "target_modules": sorted(cfg["target_modules"]), "modules_in_weights": sorted(mods),
                     "n_lora_tensors": len(keys), "layers_covered": len(layers),
                     "covers_all_36": sorted(layers) == list(range(N_LAYERS))}
    return out


def sft_truncation():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    out = {}
    for step, f in SFT_TRAIN.items():
        rows = [json.loads(l) for l in open(REPO_ROOT / f) if l.strip()]
        P, TE, L = [], [], []
        for r in rows:
            txt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]},
                                           {"role": "assistant", "content": r["completion"]}], tokenize=False)
            p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                        add_generation_prompt=True)
            k = txt.rfind("</think>")
            P.append(len(tok(p).input_ids))
            TE.append(len(tok(txt[:k]).input_ids) if k > 0 else None)
            L.append(len(tok(txt).input_ids))
        te = [x for x in TE if x is not None]
        import statistics as s
        out[step] = {"file": f, "n_examples": len(rows), "prompt_tokens_median": s.median(P),
                     "think_end_token_median": s.median(te), "full_tokens_median": s.median(L),
                     "full_tokens_max": max(L), "n_full_le_1024": sum(l <= 1024 for l in L),
                     "n_answer_start_within_1024": sum(x < 1024 for x in te),
                     "targets": sorted({r["target"] for r in rows})}
    # trained tokens per example from trainer_state (num_tokens / (epoch * n))
    states = {"S2": "runs/sft-qlora-rs-sft-from-base/checkpoint-27", "S3": "runs/sft-qlora-e3-round1/checkpoint-48",
              "S4": "runs/sft-qlora-e3c-round1/checkpoint-48", "S5_init_corpus_sft": "runs/sft-qlora-sft-v3-2nd-rank16/checkpoint-117"}
    for k, d in states.items():
        st = json.loads((REPO_ROOT / d / "trainer_state.json").read_text())
        h = [x for x in st["log_history"] if "num_tokens" in x][0]
        out[k]["trained_tokens_per_example_from_trainer_state"] = h["num_tokens"] / (h["epoch"] * out[k]["n_examples"])
        out[k]["trainer_state"] = d
    return out


def rl_groups(step_label, run_dir, scorer):
    import torch
    from trl.trainer.utils import nanstd
    from validator import SynthesisValidator
    if scorer == "checker":
        from core.reward import RUN3_CHECKS as CHECKS
    else:
        from core.ranker import OBJECTIVE_NAMES as CHECKS
    CHECKS = list(CHECKS)
    weights = torch.tensor([0.2] + [1.0] * len(CHECKS))
    rows = []
    with open(REPO_ROOT / run_dir / "generations.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["step"] is None or r["step"] >= N_GROUPS // 2:
                if r["step"] is not None and r["step"] >= N_GROUPS // 2:
                    break
                continue
            rows.append(r)
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    steps = sorted(by_step)
    assert steps == list(range(N_GROUPS // 2)), f"{step_label}: steps {steps}"
    out, objectives_present = [], None
    for s in steps:
        batch = by_step[s]
        assert len(batch) == 2 * GEN, f"{step_label} step {s}: {len(batch)} rows (dedup in dump?)"
        # group = contiguous 8 rows (TRL lays out num_generations completions per prompt contiguously)
        groups = [batch[:GEN], batch[GEN:]]
        for g in groups:
            assert len({x["target"] for x in g}) == 1, f"{step_label} step {s}: mixed targets in a group"
        vals = []
        for r in batch:
            bd = r["breakdown"]
            if scorer == "checker":
                fmt = 0.0 if bd is None else 1.0
                ch = []
                for c in CHECKS:
                    if bd is None or bd.get(f"{c}_gradeability") in SynthesisValidator.SENTINEL_TAGS:
                        ch.append(float("nan"))
                    else:
                        v = bd.get(c)
                        ch.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan"))
            else:
                fmt = 0.0 if (bd is None or not bd.get("gate_format_ok")) else 1.0
                ch = []
                for c in CHECKS:
                    v = None if bd is None else bd.get(c)
                    ch.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan"))
                if bd is not None:
                    objectives_present = (objectives_present or set()) | {c for c in CHECKS if c in bd}
            vals.append([fmt] + ch)
        R = torch.tensor(vals, dtype=torch.float32)
        unscorable = torch.isnan(R).all(dim=1)
        grouped = R.view(-1, GEN, R.shape[1])
        mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
        std_k = nanstd(grouped, dim=1, keepdim=True)
        rk = ((grouped - mean_k) / (std_k + 1e-4)).view(-1, R.shape[1])
        rew = (rk * weights.unsqueeze(0)).nansum(dim=1)
        rew[unscorable] = torch.nan
        adv = (rew - torch.nanmean(rew)) / (nanstd(rew) + 1e-4)
        adv = torch.nan_to_num(adv, nan=0.0)
        for i, r in enumerate(batch):
            out.append({"step": s, "group": 2 * s + i // GEN, "target": r["target"], "completion": r["completion"],
                        "advantage": float(adv[i]), "rewards_per_func": [None if v != v else v for v in vals[i]]})
    path = OUT_DIR / f"rl_update_groups_{step_label}.jsonl"
    with path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    import numpy as np
    a = np.array([r["advantage"] for r in out])
    return {"run_dir": run_dir, "scorer": scorer, "reward_funcs": ["format_ok"] + CHECKS,
            "reward_weights": weights.tolist(), "n_groups": len(out) // GEN, "n_completions": len(out),
            "steps": [0, N_GROUPS // 2 - 1], "file": str(path.relative_to(REPO_ROOT)),
            "n_pos_adv": int((a > 0).sum()), "n_neg_adv": int((a < 0).sum()), "n_zero_adv": int((a == 0).sum()),
            "n_groups_all_zero_adv": int(sum(np.all(a[i:i + GEN] == 0) for i in range(0, len(a), GEN))),
            "breakdown_has_ranker_objectives": (sorted(objectives_present) if objectives_present is not None else None)}


def main():
    prov = provenance()
    rows, skipped, per_target, scanned = build_probe_strings()
    print(f"probe strings: {len(rows)} rows over {len(per_target)} targets", flush=True)
    ad = adapter_inventory()
    for k, v in ad.items():
        print(f"  adapter {k}: r={v['r']} alpha={v['lora_alpha']} layers={v['layers_covered']} mods={v['modules_in_weights']}")
    tr = sft_truncation()
    for k, v in tr.items():
        print(f"  SFT {k}: n={v['n_examples']} prompt={v['prompt_tokens_median']} think_end={v['think_end_token_median']} "
              f"trained/ex={v['trained_tokens_per_example_from_trainer_state']:.0f} answers_within_1024={v['n_answer_start_within_1024']}")
    probe_all = set(PROBE["probe_astral_20"]) | set(PROBE["probe_heldout_20"])
    overlap = {}
    for st, extra in (("S3", "data/e3/e3_val.jsonl"), ("S4", "data/e3c/e3c_val.jsonl")):
        tr_t = set(tr[st]["targets"])
        val_t = {json.loads(l)["target"] for l in open(REPO_ROOT / extra) if l.strip()}
        overlap[st] = {"probe_heldout_in_train": sorted(set(PROBE["probe_heldout_20"]) & tr_t),
                       "probe_heldout_in_val": sorted(set(PROBE["probe_heldout_20"]) & val_t),
                       "probe_astral_in_train": sorted(set(PROBE["probe_astral_20"]) & tr_t),
                       "probe_any_in_train_or_val": sorted(probe_all & (tr_t | val_t))}
    union_drop = sorted(set(overlap["S3"]["probe_heldout_in_train"]) | set(overlap["S4"]["probe_heldout_in_train"]))
    d3_targets = [t for t in PROBE["probe_heldout_20"] if t not in union_drop]
    print("  overlap:", overlap, "-> D-3 held-out targets:", len(d3_targets))
    rl = {}
    for st, (d, sc) in RL_RUNS.items():
        rl[st] = rl_groups(st, d, sc)
        print(f"  RL {st}: {rl[st]['n_groups']} groups, pos/neg/zero adv {rl[st]['n_pos_adv']}/{rl[st]['n_neg_adv']}/"
              f"{rl[st]['n_zero_adv']}, ranker objectives in breakdown: {rl[st]['breakdown_has_ranker_objectives']}")
    for st in rl.values():
        pass
    for t in tr.values():
        t.pop("targets")
    doc = {**prov, "block": "0",
           "probe_strings": {"file": str(PROBE_STRINGS_PATH.relative_to(REPO_ROOT)), "n_rows": len(rows),
                             "per_target": per_target, "skipped_no_literal_string": skipped,
                             "raw_files_scanned_rows": scanned,
                             "context": "chat-template prompt (add_generation_prompt=True) + '<think>\\n\\n</think>\\n\\n{\\n  \"precursors\": '",
                             "string_definition": "most frequent literal precursor-array text '[...]' among raw completions parsing to the recipe"},
           "adapters": ad,
           "sft_training_truncation": tr,
           "sft_truncation_note": ("experiments/sft.py passes max_length=h['max_seq_len']=1024 to TRL SFTConfig "
                                   "(truncation_mode keep_start); trained tokens per example from trainer_state ~1025 "
                                   "for every SFT run, so training saw the prompt + the opening of <think>, never the "
                                   "answer JSON. Block D's SFT update direction uses the sequences AS TRAINED (first 1024 "
                                   "tokens, loss on every token, as TRL's text-field SFT does)."),
           "d3_overlap": overlap, "d3_heldout_targets_after_drop": d3_targets,
           "rl_update_groups": rl,
           "D_sampled_definition": ("P-19D-3 positives: recipe with E3 count >= 1 and Base count == 0 among the "
                                    "probe strings of the retained held-out targets (fixed in Block 0, before D)."),
           }
    p = write_json("inventory.json", doc)
    print(f"-> {p}")
    notify(f"MIRA 19 Block 0 CPU done: {len(rows)} probe strings; D-3 targets {len(d3_targets)}; "
           f"E1 ranker objectives in log: {bool(rl['S7']['breakdown_has_ranker_objectives'])}")


if __name__ == "__main__":
    main()
