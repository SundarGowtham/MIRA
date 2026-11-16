#!/usr/bin/env python
"""
research/phase19_common.py -- shared constants, provenance, model holder and
scoring for Phase 19 (docs/phases/PHASE19_PREREG.md, commit 7412c23).

Every phase19_* script imports from here so the step table, probe strings,
context, provenance and scoring are identical across blocks.

PROVENANCE: the spec was committed (7412c23) BEFORE any Phase 19 computation.
provenance() asserts the live git blob of PHASE19_PREREG.md equals the frozen
blob and records the commit that introduced it as prereg_commit.

CONTEXT (answer-only, identical for every model; implementation choice fixed
in Block 0 before any GPU scoring): the chat-template-rendered closed-book
prompt (add_generation_prompt=True, exactly as every sampling script rendered
it) + an empty think block + the answer preamble that 100% of parsed samples
use ('{\\n  "precursors": '). The probe STRING is the literal precursor-array
text '[ ... ]' copied from real samples. l_m(y) = sum of log p over the tokens
whose character span overlaps the string (token-offset based, so a token that
merges the preamble's trailing space with '[' is counted once, identically
for every model).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

PREREG_PATH = REPO_ROOT / "docs" / "phases" / "PHASE19_PREREG.md"
PREREG_BLOB_SHA_FROZEN = "4df94b3aa65330ebe14be3025b9c3ca62c8c4d19"
SEED = 20261116
N_BOOT = 10000
OUT_DIR = REPO_ROOT / "results" / "phase19"
NTFY_TOPIC = "mira-g5x7k2-status"
BASE_MODEL = "Qwen/Qwen3-8B"
N_LAYERS = 36
TOP_N = 30

STEPS = {
    "S2": ("base", "rs_sft"),
    "S3": ("base", "e3"),
    "S4": ("base", "e3c"),
    "S5": ("corpus_sft", "gdpo_v3_ckpt300"),
    "S6": ("rs_sft", "phase12_ckpt300"),
    "S7": ("rs_sft", "e1_ckpt300"),
}
METHOD = {"S2": "SFT", "S3": "SFT", "S4": "SFT", "S5": "RL", "S6": "RL", "S7": "RL"}
PRIMARY_STEPS = ("S3", "S4", "S6", "S7")
SECONDARY_STEPS = ("S2", "S5")
MODELS = ["base", "corpus_sft", "rs_sft", "e3", "e3c", "gdpo_v3_ckpt300", "phase12_ckpt300", "e1_ckpt300"]

ANSWER_PREAMBLE = '<think>\n\n</think>\n\n{\n  "precursors": '
PROBE_STRINGS_PATH = OUT_DIR / "probe_strings.jsonl"
SCORES_PATH = OUT_DIR / "string_scores.jsonl"   # per (model, target, recipe) l_m(y), written by Block G


def notify(msg: str) -> None:
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode(), method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    blob = _git("hash-object", str(PREREG_PATH))
    if blob != PREREG_BLOB_SHA_FROZEN:
        raise SystemExit(f"PHASE19_PREREG.md changed after freeze: {blob} != {PREREG_BLOB_SHA_FROZEN}")
    commits = _git("log", "--format=%H", "--find-object=" + blob, "--", str(PREREG_PATH)).splitlines()
    if not commits:
        raise SystemExit("PHASE19_PREREG.md blob is not in any commit; the spec must be committed first")
    return {"prereg_commit": commits[-1], "prereg_blob_sha": blob,
            "repo_head_at_run": _git("rev-parse", "HEAD"), "seed": SEED}


def write_json(name: str, obj: dict) -> Path:
    """New files only: refuse to overwrite a git-tracked file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=REPO_ROOT,
                      capture_output=True).returncode == 0:
        raise SystemExit(f"refusing to overwrite tracked results file {path}")
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


def adapters() -> dict:
    from research.phase18_common import ADAPTERS
    return dict(ADAPTERS)


def base_model_path() -> str:
    snaps = sorted((Path("/models/huggingface/hub/models--Qwen--Qwen3-8B/snapshots")).glob("*"))
    return str(snaps[-1]) if snaps else BASE_MODEL


def closed_prompt(target: str) -> str:
    from stratified_difficulty_eval import SYSTEM_MSG
    return SYSTEM_MSG + "\n\nTarget: " + target + "\n\nProvide your synthesis route as a JSON object."


def render_prompt(tok, target: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": closed_prompt(target)}],
                                   tokenize=False, add_generation_prompt=True)


def context_text(tok, target: str) -> str:
    return render_prompt(tok, target) + ANSWER_PREAMBLE


def load_probe_strings() -> dict:
    """target -> list of rows (ordered by pooled count, desc)."""
    out = {}
    for line in PROBE_STRINGS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["target"], []).append(r)
    return out


def load_scores() -> dict:
    """(model, target) -> {recipe_key: l}; recipe_key = '|'.join(recipe)."""
    out = {}
    for line in SCORES_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault((r["model"], r["target"]), {})[r["recipe_key"]] = r["l"]
    return out


def rkey(recipe) -> str:
    return "|".join(recipe)


# ---------------------------------------------------------------------------
# GPU: one bf16 base with every adapter loaded; per-layer adapter selection
# ---------------------------------------------------------------------------
class MultiAdapterModel:
    def __init__(self, names: list[str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer
        ad = adapters()
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(base_model_path(), dtype=torch.bfloat16, device_map="cuda")
        names = list(names)
        self.model = PeftModel.from_pretrained(base, str(REPO_ROOT / ad[names[0]]), adapter_name=names[0])
        for n in names[1:]:
            self.model.load_adapter(str(REPO_ROOT / ad[n]), adapter_name=n)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.names = names
        self.lora = []  # (layer_idx, qualified name, module)
        for qn, m in self.model.named_modules():
            if isinstance(m, LoraLayer):
                parts = qn.split(".")
                li = int(parts[parts.index("layers") + 1])
                self.lora.append((li, qn, m))
        layers_cov = sorted({li for li, _, _ in self.lora})
        assert layers_cov == list(range(N_LAYERS)), f"LoRA modules do not cover all {N_LAYERS} blocks"
        self.layers = self.model.base_model.model.model.layers
        self.device = next(self.model.parameters()).device

    def set_per_layer(self, per_layer):
        """per_layer: list of length N_LAYERS; entry = adapter name or None (no adapter)."""
        for li, _qn, m in self.lora:
            a = per_layer[li]
            if a is None:
                m._disable_adapters = True
            else:
                m._disable_adapters = False
                m.set_adapter(a, inference_mode=True)

    def set_all(self, name):
        self.set_per_layer([name] * N_LAYERS)


def answer_token_mask(tok, ctx: str, full: str):
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc.input_ids
    mask = [end > len(ctx) for (_s, end) in enc.offset_mapping]
    return ids, mask


class Prepared:
    """Tokenized probe strings for one target: shared token prefix (no answer
    token in it) + per-string continuations; answer mask per continuation."""

    def __init__(self, tok, target: str, strings: list[str]):
        ctx = context_text(tok, target)
        seqs = [answer_token_mask(tok, ctx, ctx + s) for s in strings]
        P = min(len(x) for x, _ in seqs)
        first = seqs[0][0]
        for x, m in seqs:
            i = 0
            while i < P and x[i] == first[i]:
                i += 1
            P = min(P, i, m.index(True))
        self.P = P
        self.prefix = first[:P]
        self.conts = [x[P:] for x, _ in seqs]
        self.masks = [m[P:] for _, m in seqs]
        self.target = target


def _inner(mm):
    return mm.model.base_model.model.model, mm.model.base_model.model.lm_head


def score_prepared(mm: MultiAdapterModel, prep: Prepared, batch: int = 30, on_prefix=None, on_cont=None):
    """Exact teacher-forced l(y) per string. One prefix forward (KV cache),
    then batched continuations reusing a copy of the cache. Logits only at
    the positions that predict answer tokens. on_prefix()/on_cont(b0, n, Lc)
    are optional callbacks invoked right before each forward (used by F1 to
    stage per-position patches)."""
    import copy
    import torch
    body, head = _inner(mm)
    dev = mm.device
    pad = mm.tok.pad_token_id if mm.tok.pad_token_id is not None else mm.tok.eos_token_id
    with torch.no_grad():
        if on_prefix is not None:
            on_prefix()
        pre = torch.tensor([prep.prefix], device=dev)
        po = body(input_ids=pre, use_cache=True)
        cache = po.past_key_values
        h_last = po.last_hidden_state[:, -1:]
        out = []
        for b0 in range(0, len(prep.conts), batch):
            cs = prep.conts[b0:b0 + batch]
            ms = prep.masks[b0:b0 + batch]
            n, Lc = len(cs), max(len(c) for c in cs)
            ids = torch.full((n, Lc), pad, dtype=torch.long)
            att = torch.zeros((n, prep.P + Lc), dtype=torch.long)
            am = torch.zeros((n, Lc), dtype=torch.bool)
            att[:, :prep.P] = 1
            for i, (c, m) in enumerate(zip(cs, ms)):
                ids[i, :len(c)] = torch.tensor(c)
                att[i, prep.P:prep.P + len(c)] = 1
                am[i, :len(c)] = torch.tensor(m)
            ids, att, am = ids.to(dev), att.to(dev), am.to(dev)
            c2 = copy.deepcopy(cache)
            c2.batch_repeat_interleave(n)
            pos = (prep.P + torch.arange(Lc, device=dev)).unsqueeze(0).expand(n, -1)
            if on_cont is not None:
                on_cont(b0, n, Lc)
            o = body(input_ids=ids, attention_mask=att, past_key_values=c2, position_ids=pos, use_cache=True)
            hid = torch.cat([h_last.expand(n, 1, -1), o.last_hidden_state[:, :-1]], dim=1)  # predicts ids[:, j]
            sel = am.nonzero(as_tuple=False)
            logits = head(hid[sel[:, 0], sel[:, 1]]).float()
            lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[sel[:, 0], sel[:, 1]].unsqueeze(-1)).squeeze(-1)
            tot = torch.zeros(n, device=dev, dtype=torch.float32).index_add_(0, sel[:, 0], lp)
            out.extend(tot.tolist())
            del c2, o, hid, logits
        del cache
    return out


def score_full_reference(mm: MultiAdapterModel, target: str, strings: list[str]):
    """Uncached full-sequence scoring, one string at a time (smoke check of
    score_prepared's KV reuse)."""
    import torch
    body, head = _inner(mm)
    ctx = context_text(mm.tok, target)
    out = []
    with torch.no_grad():
        for s in strings:
            ids, m = answer_token_mask(mm.tok, ctx, ctx + s)
            x = torch.tensor([ids], device=mm.device)
            h = body(input_ids=x).last_hidden_state[0]
            idx = torch.tensor([j for j in range(1, len(ids)) if m[j]], device=mm.device)
            lp = torch.log_softmax(head(h[idx - 1]).float(), -1).gather(-1, x[0, idx].unsqueeze(-1)).sum()
            out.append(float(lp))
    return out


def recovery(delta_hybrid, delta) -> float:
    import numpy as np
    dh, d = np.asarray(delta_hybrid, float), np.asarray(delta, float)
    return float(1 - ((dh - d) ** 2).sum() / (d ** 2).sum())
