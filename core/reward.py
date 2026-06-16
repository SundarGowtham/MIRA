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

    try:
        data = json.loads(text_cleaned[start_idx : end_idx + 1])
    except json.JSONDecodeError as e:
        raise ParseFailure(f"JSON decode error: {e}") from e

    if not isinstance(data, dict):
        raise ParseFailure(f"parsed JSON is not an object (got {type(data).__name__})")

    precursors = []
    for p in data.get("precursors", []):
        if not isinstance(p, dict) or "formula" not in p:
            continue
        try:
            precursors.append(PredictedPrecursor(
                formula=str(p["formula"]),
                amount=float(p.get("amount", 1.0)),
            ))
        except (TypeError, ValueError):
            continue

    operations = []
    for op in data.get("operations", []):
        if not isinstance(op, dict) or "type" not in op:
            continue

        # Case-insensitive key lookup: we've now hit one casing mismatch
        # (temperature_c vs temperature_C) that silently dropped real data
        # rather than raising, because dict.get() doesn't fail loudly on a
        # near-miss key. Rather than patch this one key, normalize all
        # operation-level keys to lowercase once, so the same failure class
        # doesn't quietly recur for time_h/atmosphere/media if a future
        # generation batch varies casing on those instead.
        op_lower = {k.lower(): v for k, v in op.items()}

        temp_c = op_lower.get("temperature_c")
        time_h = op_lower.get("time_h")
        atm = op_lower.get("atmosphere")

        operations.append(PredictedOperation(
            type=str(op["type"]),
            conditions=PredictedConditions(
                heating_temperature=[float(temp_c)] if temp_c is not None else [],
                heating_time=[float(time_h)] if time_h is not None else [],
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


def load_validator(formula_set_path: Path, pd_cache_path: Path | None = None):
    import pickle
    with formula_set_path.open("rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_cache(pd_cache_path) if pd_cache_path and pd_cache_path.exists() else None
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