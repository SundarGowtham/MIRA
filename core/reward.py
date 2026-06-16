from __future__ import annotations
import pickle
import re
from pathlib import Path
from typing import List
import json

from validator import (
    SynthesisValidator, ThermoChecker,
    PredictedRoute, PredictedPrecursor, PredictedOperation, PredictedConditions,
)

PRECURSORS_RE = re.compile(r"<precursors>(.*?)</precursors>", re.DOTALL)
OPERATIONS_RE = re.compile(r"<operations>(.*?)</operations>", re.DOTALL)
TEMP_RE       = re.compile(r"T=([0-9.]+)")
TIME_RE       = re.compile(r"t=([0-9.]+)")
ATM_RE        = re.compile(r"atm=([^,|]+)")

def parse_completion(text: str, target_formula: str) -> PredictedRoute:
    precursors = []
    operations = []
    thermodynamic_checks = []

    # 1. Strip the reasoning block
    # re.DOTALL ensures the '.' matches newline characters inside the <think> tags
    text_cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 2. Isolate the JSON string (in case there's trailing/leading text)
    start_idx = text_cleaned.find('{')
    end_idx = text_cleaned.rfind('}')

    if start_idx == -1 or end_idx == -1:
        return PredictedRoute(target_formula=target_formula, precursors=[], operations=[])

    



def parse_completion_old(text: str, target_formula: str) -> PredictedRoute:
    """Parse model output back into validator schema. Defaults to empty on failure."""
    precursors = []
    operations = []

    pm = PRECURSORS_RE.search(text)
    if pm:
        for line in pm.group(1).strip().splitlines():
            line = line.strip().lstrip("-").strip()
            if not line or "|" not in line:
                continue
            try:
                formula, amount = [p.strip() for p in line.split("|", 1)]
                precursors.append(PredictedPrecursor(formula=formula, amount=float(amount)))
            except (ValueError, IndexError):
                continue

    om = OPERATIONS_RE.search(text)
    if om:
        for line in om.group(1).strip().splitlines():
            line = re.sub(r"^\d+\.\s*", "", line.strip())
            if not line or "|" not in line:
                continue
            op_type, _, cond_str = line.partition("|")
            cond_str = cond_str.strip()
            temps = [float(t) for t in TEMP_RE.findall(cond_str)]
            times = [float(t) for t in TIME_RE.findall(cond_str)]
            atm_match = ATM_RE.search(cond_str)
            atm = [a.strip() for a in atm_match.group(1).split(",")] if atm_match else []
            operations.append(PredictedOperation(
                type=op_type.strip(),
                conditions=PredictedConditions(
                    heating_temperature=temps,
                    heating_time=times,
                    heating_atmosphere=atm,
                ),
            ))

    return PredictedRoute(
        target_formula=target_formula,
        precursors=precursors,
        operations=operations,
    )


def load_validator(formula_set_path: Path, pd_cache_path: Path | None = None):
    with formula_set_path.open("rb") as f:
        formula_set = pickle.load(f)
    thermo = ThermoChecker.from_cache(pd_cache_path) if pd_cache_path and pd_cache_path.exists() else None
    return SynthesisValidator(formula_set, thermo_checker=thermo)


def make_reward_fn(validator: SynthesisValidator):

    def reward_fn(completions, target_formula, **kwargs):
        rewards = []
        for completion, target in zip(completions, target_formula):
            try:
                route = parse_completion(completion, target)
                r, _ = validator.validate(route, target)

                adjusted = r - 0.30
                # rewards.append(r)
                rewards.append(max(adjusted, 0.0))
            except Exception:
                rewards.append(0.0)
        return rewards
    return reward_fn