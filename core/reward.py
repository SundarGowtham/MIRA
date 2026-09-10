from __future__ import annotations
import json
import re
from pathlib import Path
from statistics import stdev

from validator import (
    SynthesisValidator, ThermoChecker,
    PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions,
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

# Model writes fraction literals ("amount": 5/12, "moles_per_formula_unit":
# 0.75 / 1) instead of decimals wherever it wants a non-terminating value --
# observed disproportionately on fractional/doped-composition targets, whose
# real amounts genuinely don't terminate in decimal (Phase 12 smoke gate,
# 2026-09-09: 18.2% parse-failure rate on the smoke run, concentrated on
# exactly that stratum). `a/b` is not valid JSON number syntax, so a single
# occurrence anywhere in the document fails json.loads for the whole object
# even when the rest -- including precursors/operations we actually need --
# is fine. Requires no quote directly after ':' so it can't fire inside a
# string value (a string field starts with '"', a number field doesn't).
FRACTION_LITERAL_RE = re.compile(
    r"(:\s*)(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)(?=\s*[,}\]])"
)


def _repair_fraction_literals(json_str: str) -> str:
    def _replace(m: re.Match) -> str:
        prefix, num, den = m.group(1), float(m.group(2)), float(m.group(3))
        if den == 0:
            return m.group(0)  # leave it -- let json.loads raise, don't divide by zero
        return f"{prefix}{num / den}"
    return FRACTION_LITERAL_RE.sub(_replace, json_str)


def _balanced_json_objects(text: str) -> list[str]:
    """Top-level {...} substrings via brace matching, ignoring braces inside
    strings. Ported from reward_geometry.py's analysis-only extractor --
    handles completions where the naive first-'{'-to-last-'}' span crosses
    multiple distinct top-level objects (observed failure mode: 'Extra data'
    JSONDecodeError, the model emits more than one brace-delimited chunk)."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:i + 1])
                    start = None
    return out


def _try_parse_json_object(candidate: str) -> dict | None:
    """Try straight, then fraction-literal-repaired, then //-comment-stripped
    (each independently, since the fixes address unrelated failure modes and
    stacking them unconditionally risks mangling an otherwise-valid string)."""
    for attempt in (candidate, _repair_fraction_literals(candidate),
                   re.sub(r"//[^\n]*", "", candidate),
                   re.sub(r"//[^\n]*", "", _repair_fraction_literals(candidate))):
        try:
            data = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


class ParseFailure(Exception):
    """Raised when a completion cannot be turned into a PredictedRoute.
    Kept distinct from validator-level scoring failures so we can count
    'the model's output was unparseable' separately from 'the model's
    output was parseable but chemically wrong' — collapsing these two
    was the root cause of the eval looking identical across six
    independently trained checkpoints (every run was silently falling
    back to an empty PredictedRoute and only the target-derived
    constraints, which don't depend on the model's output at all, were
    contributing to the score).
    """
    pass


def _coerce_float(value) -> float | None:
    """
    Coerce a field that should be numeric to float, defensively.

    Observed real failure: model emitted "time": "1 h" (string with units)
    instead of "time": 1.0 (number) in rank32-seed1337. Rather than crash
    on the float() call inside list comprehensions — which is what bit us
    once already — strip non-numeric trailing junk and parse the leading
    number. Returns None if nothing numeric can be recovered, letting the
    caller decide whether that's a missing field or a parse failure.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.match(r"\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", value)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def parse_completion(text: str, target_formula: str) -> PredictedRoute:
    """
    Parse a model completion into the validator's PredictedRoute schema.

    Real completion format (confirmed from data/sft/train.jsonl):
        <think> ... reasoning ... </think>
        {
          "precursors": [{"formula": "...", "amount": ...}, ...],
          "operations": [
            {"type": "...", "temperature_c": ..., "time_h": ...,
             "atmosphere": "...", "media": "..."},
            ...
          ],
          "thermodynamic_checks": [...]   # present in training data but NOT
                                           # part of PredictedRoute — the
                                           # validator recomputes thermo
                                           # feasibility itself from
                                           # target_formula, it never trusts
                                           # the model's self-report. We parse
                                           # it for potential future use
                                           # (e.g. cross-checking the model's
                                           # claimed oxidation states against
                                           # the validator's own) but do not
                                           # attach it to PredictedRoute.
        }

    Raises ParseFailure (caller decides how to score that) rather than
    silently returning an empty route, so parse failures are visible.
    """
    text_cleaned = THINK_RE.sub("", text, count=1).strip()
    text_cleaned = FENCE_RE.sub("", text_cleaned.strip())

    start_idx = text_cleaned.find("{")
    end_idx = text_cleaned.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ParseFailure(f"no JSON object found in completion (len={len(text)})")

    # Model-inserted // line comments and fraction-literal numbers
    # ("amount": 5/12) were observed in real completions -- both invalid
    # per JSON spec, but neither means the rest of the structure is
    # unreliable, so _try_parse_json_object recovers them rather than
    # counting a single occurrence as a hard parse failure (Phase 12 smoke
    # gate, 2026-09-09: 18.2% parse-failure rate, concentrated on
    # fractional-composition targets).
    json_str = text_cleaned[start_idx : end_idx + 1]
    data = _try_parse_json_object(json_str)

    if data is None:
        # The naive first-'{'-to-last-'}' span can cross multiple distinct
        # top-level objects (observed failure mode: 'Extra data'
        # JSONDecodeError -- the model emits more than one brace-delimited
        # chunk). Fall back to brace-matched candidates and take the
        # largest one that actually looks like a route, same logic as
        # reward_geometry.py's analysis-only extract_route_json.
        candidates = _balanced_json_objects(text_cleaned)
        route_like = []
        for c in candidates:
            obj = _try_parse_json_object(c)
            if obj is not None and ("precursors" in obj or "operations" in obj):
                route_like.append((len(c), obj))
        if route_like:
            data = max(route_like, key=lambda x: x[0])[1]

    if data is None:
        raise ParseFailure(f"JSON decode error: no parseable route object found "
                           f"(naive span + {len(_balanced_json_objects(text_cleaned))} "
                           f"balanced candidates all failed)")

    if not isinstance(data, dict):
        raise ParseFailure(f"parsed JSON is not an object (got {type(data).__name__})")

    precursors = []
    for p in data.get("precursors", []):
        if not isinstance(p, dict) or "formula" not in p:
            continue
        amount = _coerce_float(p.get("amount", 1.0))
        if amount is None:
            amount = 1.0
        try:
            precursors.append(PredictedPrecursor(
                formula=str(p["formula"]),
                amount=amount,
            ))
        except (TypeError, ValueError):
            continue

    operations = []
    for op in data.get("operations", []):
        if not isinstance(op, dict) or "type" not in op:
            continue

        # Case- and name-insensitive key lookup. Ground-truth survey of 162
        # real eval completions (see notes below) showed the model uses
        # several DIFFERENT key names for the same field, not just casing
        # variants of the trained schema:
        #   temperature: 733x   temperature_C: 7x   temperature_c: 3x
        #   (i.e. the model's dominant key, post-SFT, is "temperature" —
        #    NOT "temperature_c", which is what 100% of training data used)
        #   time_h (trained) vs time (drifted, paired with "temperature")
        # This is real schema drift away from the training format, not
        # noise — treat it as a finding, not just a bug to silently absorb.
        op_lower = {k.lower(): v for k, v in op.items()}

        temp_c = _coerce_float(op_lower.get("temperature_c")
                                or op_lower.get("temperature")
                                or op_lower.get("temperature_celsius"))
        time_h = _coerce_float(op_lower.get("time_h")
                                or op_lower.get("time")
                                or op_lower.get("time_hours"))
        atm = op_lower.get("atmosphere")

        operations.append(PredictedOperation(
            type=str(op["type"]),
            conditions=PredictedConditions(
                heating_temperature=[temp_c] if temp_c is not None else [],
                heating_time=[time_h] if time_h is not None else [],
                heating_atmosphere=[str(atm)] if atm is not None else [],
                mixing_media=str(op_lower["media"]) if op_lower.get("media") is not None else None,
                atmosphere=atm if atm in ("Ar", "N2", "vacuum", "air") else None,
            ),
        ))

    if not precursors and not operations:
        # JSON parsed, but had neither field populated — almost certainly
        # the wrong shape rather than a genuinely empty route. Treat as a
        # parse failure rather than a valid-but-empty route, so it's counted
        # and visible rather than silently scored as "no precursors, no ops".
        raise ParseFailure("parsed JSON contained no usable precursors or operations")

    return PredictedRoute(
        target_formula=target_formula,
        precursors=precursors,
        operations=operations,
    )


def load_validator(formula_set_path: Path, pd_index_path: Path | None = None,
                   project_root: Path | None = None):
    """
    formula_set_path: data/cache/mp_formula_set.pkl
    pd_index_path:    data/cache/pd_index.json  (maps chemsys -> shard filename,
                       e.g. "Al-O-Zn" -> "pd_shards/Al-O-Zn.pkl")
    project_root:      directory pd_index.json's relative shard paths are
                       resolved against (i.e. data/cache/), since
                       ThermoChecker._get_pd does project_root / pd_index[chemsys]

    NOTE: thermo data is sharded per chemical system (data/cache/pd_shards/*.pkl),
    not one bulk pickle. ThermoChecker.from_sharded_cache only loads pd_index.json
    up front; individual shards are lazy-loaded on first access by chemsys.
    """
    import pickle
    with formula_set_path.open("rb") as f:
        formula_set = pickle.load(f)

    thermo = None
    if pd_index_path and pd_index_path.exists():
        root = project_root or pd_index_path.parent
        thermo = ThermoChecker.from_sharded_cache(pd_index_path, root)

    return SynthesisValidator(formula_set, thermo_checker=thermo)


def make_reward_fn(validator: SynthesisValidator, verbose: bool = False):
    """
    Returns reward_fn(completions, target_formula, **kwargs) -> list[float].

    Tracks parse failures on the function object itself (reward_fn.parse_stats)
    so callers (training loop, eval harness) can surface a parse-failure rate
    instead of it being invisible inside a blanket try/except, which is what
    produced six independently-trained checkpoints scoring identically: every
    completion was silently falling back to an empty route and only the
    target-derived constraints (which never depend on the model's output)
    were contributing to the score.
    """
    stats = {"n_total": 0, "n_parse_failed": 0, "n_validate_failed": 0}

    def reward_fn(completions, target_formula, **kwargs):
        rewards = []
        for completion, target in zip(completions, target_formula):
            stats["n_total"] += 1
            try:
                route = parse_completion(completion, target)
            except Exception as e:
                # Broad on purpose: ParseFailure is the expected path, but a
                # malformed operations entry can raise TypeError/ValueError
                # from dataclass construction, and that must not propagate
                # into TRL's reward computation and kill a training run.
                stats["n_parse_failed"] += 1
                if verbose:
                    print(f"[parse_fail] target={target}: {e}")
                rewards.append(0.0)
                continue

            try:
                r, _ = validator.validate(route, target)
                adjusted = max(r - 0.30, 0.0)
                rewards.append(adjusted)
            except Exception as e:
                stats["n_validate_failed"] += 1
                if verbose:
                    print(f"[validate_fail] target={target}: {e}")
                rewards.append(0.0)
        return rewards

    reward_fn.parse_stats = stats
    return reward_fn


# Phase 12 (2026-09-09): a stratum's cumulative parse-failure rate crossing
# this bar, after enough samples to not be early-training noise, means the
# non-random attrition the smoke gate found is growing rather than holding
# -- exactly the case that would silently bias which routes a run can even
# score. MIN_N=20 chosen so one bad group of 8 can't trip it alone.
PARSE_FAIL_ALERT_THRESHOLD = 0.30
PARSE_FAIL_ALERT_MIN_N = 20


def _fire_parse_fail_alert(stratum: str, rate: float, n_total: int) -> None:
    """Best-effort notification, fired once per stratum per run. Two
    channels, independently wrapped: wandb.alert() (shows on the run page;
    reaches Slack/email only if the account has that configured) and the
    same ntfy.sh topic every tmux launcher in this repo already pings
    (guaranteed delivery regardless of wandb account settings). Neither
    failure mode may raise -- a broken notification must never crash
    training."""
    title = f"parse_fail_rate/{stratum} = {rate:.0%} (n={n_total})"
    try:
        import wandb
        if wandb.run is not None:
            wandb.alert(title="MIRA parse-failure bias growing", text=title,
                       level=wandb.AlertLevel.WARN)
    except Exception:
        pass
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://ntfy.sh/mira-g5x7k2-status",
            data=f"MIRA parse-fail alert: {title}".encode(),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


# Run-3 reward vector (gdpo_v4_next_steps_claude_recommendation.md §5, step 4).
# The five live-or-revivable channels. Dropped:
#   target_stability, target_match — prompt-determined / constant, so zero
#     within-group variance by construction under any group-relative method;
#   charge_neutrality — insensitive to the variation the policy actually
#     produces (100% zero-std groups across runs 1-2);
#   precursors_exist, temperature_plausible — saturated (means ~0.998/0.999).
RUN3_CHECKS = (
    "amount_accuracy",
    "thermodynamic_favorable",
    "stoichiometry",
    "chempot_atmosphere",
    "operation_order",
)


def make_check_reward_fns(validator: SynthesisValidator, dump_path: str | None = None,
                          checks: tuple[str, ...] = RUN3_CHECKS,
                          format_weight: float = 0.2,
                          target_strata: dict[str, str] | None = None):
    """
    Per-check reward functions for multi-reward (GDPO) training via TRL's
    multi_objective_aggregation.

    Design (see CLAUDE_RESPONSE_TO_WRITEUP.md §1):
      - ONE shared parse+validate per completion, cached. TRL calls every
        reward func with the same batch; without the cache each completion
        would be validated once per check (9-10x).
      - Can't-compute checks return None, NOT 0.0: TRL converts None ->
        NaN (grpo_trainer.py:1518-1519), excludes it from that check's
        group mean/std, and drops it from the nansum aggregation. This is
        None-propagation — the RL-side twin of validate()'s sentinel
        exclusion. Returning 0.0 would reinstall the sentinel bug one
        level up and inject spurious z-scores under normalize_then_sum.
      - Parse failure yields None for every chemistry check (masked by
        TRL's unscorable_mask -> advantage 0) and 0.0 for format_ok:
        format pressure without poisoning the chemistry channels.
      - dump_path appends (step, target, completion, breakdown) as JSONL:
        the generation archive for RS-SFT seeding, verified-pair DPO,
        on-policy p-hat re-estimation, and the JEPA-surrogate monitor.
      - checks: run-3 vector by default (RUN3_CHECKS, uniform weights —
        the validator's scalar weights were tuned for a weighted sum of
        [0,1] scores and mean something different after per-channel
        z-normalization). A missing requested check raises — silently
        dropping channels is exactly this codebase's historical failure
        mode.
      - Each check fn also reports its within-group std via TRL's
        log_metric hook (wandb: within_group_std/<check>). Groups are
        keyed on target_formula, robust to batch ordering. This is the
        per-channel dead-channel diagnostic that the aggregate
        frac_reward_zero_std hid in runs 1-2.
      - If `target_strata` is given (target_formula -> stratum, e.g. from
        data/rl_run3's own `stratum` field), `bank()` also logs
        parse_fail_rate/<stratum> per batch. Phase 12 smoke gate
        (2026-09-09) found an 18.2% parse-failure rate concentrated on the
        fractional/doped stratum specifically -- non-random attrition
        correlated with the outcome variable the run is meant to measure.
        This makes that bias visible while training, not reconstructed
        from generations.jsonl after the fact.

    Returns (funcs, names, weights) for GRPOTrainer(reward_funcs=funcs)
    and GRPOConfig(reward_weights=weights): uniform 1.0 per retained check,
    plus a small weight on format_ok.
    """
    missing = [c for c in checks if c not in validator.weights]
    if missing:
        raise ValueError(
            f"requested reward checks {missing} not present in validator "
            f"(has {sorted(validator.weights)}) — refusing to run with a "
            f"silently truncated reward vector")

    cache: dict[tuple[str, str], dict | None] = {}
    dump_fp = open(dump_path, "a", buffering=1) if dump_path else None
    strata_counts: dict[str, list[int]] = {}  # stratum -> [n_total, n_failed]
    alerted_strata: set[str] = set()

    def bank(completions, target_formula, trainer_state=None, log_metric=None, **kwargs):
        out = []
        new = []
        for c, t in zip(completions, target_formula):
            key = (c, t)
            if key not in cache:
                try:
                    route = parse_completion(c, t)
                    cache[key] = validator.validate(route, t)[1]
                except Exception:
                    cache[key] = None
                new.append((c, t, cache[key]))
            out.append(cache[key])
        # dump only newly-validated completions — bank() is called once per
        # reward func per batch, and we want each generation archived ONCE.
        if dump_fp is not None and new:
            step = getattr(trainer_state, "global_step", None)
            for c, t, bd in new:
                dump_fp.write(json.dumps(
                    {"step": step, "target": t, "completion": c, "breakdown": bd},
                    default=str) + "\n")
        if target_strata is not None and new:
            for _c, t, bd in new:
                stratum = target_strata.get(t, "unknown")
                counts = strata_counts.setdefault(stratum, [0, 0])
                counts[0] += 1
                if bd is None:
                    counts[1] += 1
            for stratum, (n_total, n_failed) in strata_counts.items():
                rate = n_failed / n_total
                if log_metric is not None:
                    log_metric(f"parse_fail_rate/{stratum}", rate)
                if (stratum not in alerted_strata and n_total >= PARSE_FAIL_ALERT_MIN_N
                        and rate > PARSE_FAIL_ALERT_THRESHOLD):
                    alerted_strata.add(stratum)
                    _fire_parse_fail_alert(stratum, rate, n_total)
        return out

    checks = list(checks)

    def format_ok(completions, target_formula, **kwargs):
        bds = bank(completions, target_formula, **kwargs)
        return [0.0 if bd is None else 1.0 for bd in bds]
    format_ok.__name__ = "format_ok"

    def make_fn(check: str):
        def fn(completions, target_formula, log_metric=None, **kwargs):
            bds = bank(completions, target_formula, **kwargs)
            vals = []
            for bd in bds:
                if bd is None:
                    vals.append(None)
                elif bd.get(f"{check}_gradeability") in SynthesisValidator.SENTINEL_TAGS:
                    vals.append(None)
                else:
                    v = bd.get(check)
                    vals.append(float(v) if isinstance(v, (int, float)) else None)
            if log_metric is not None:
                # Per-check within-group std, keyed on target (TRL lays each
                # prompt's num_generations completions contiguously, but the
                # target key is robust to any ordering). This is the live
                # dead-channel diagnostic; frac_reward_zero_std aggregates
                # across channels and read ~0 while 8/10 were dead.
                groups: dict[str, list[float]] = {}
                for v, t in zip(vals, target_formula):
                    if v is not None:
                        groups.setdefault(t, []).append(v)
                stds = [stdev(g) for g in groups.values() if len(g) >= 2]
                if stds:
                    log_metric(f"within_group_std/{check}",
                               sum(stds) / len(stds))
            return vals
        fn.__name__ = f"check_{check}"
        return fn

    funcs = [format_ok] + [make_fn(c) for c in checks]
    names = ["format_ok"] + checks
    weights = [format_weight] + [1.0] * len(checks)
    return funcs, names, weights
