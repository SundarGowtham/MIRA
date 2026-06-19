from __future__ import annotations
import json
import re
from pathlib import Path

from validator import (
    SynthesisValidator, ThermoChecker,
    PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions,
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


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

    json_str = text_cleaned[start_idx : end_idx + 1]
    # Strip trailing // line comments. Observed in real completions (e.g. a
    # model-inserted "// Target is metastable, slightly above hull?" inside
    # an otherwise well-formed object) — invalid per JSON spec, but a single
    # comment doesn't mean the rest of the structure is unreliable, so we
    # recover it rather than counting it as a hard parse failure.
    json_str = re.sub(r"//[^\n]*", "", json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseFailure(f"JSON decode error: {e}") from e

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
                mixing_media=op_lower.get("media"),
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
            except ParseFailure as e:
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