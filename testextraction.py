# # from core.reward import parse_completion
# # from rich import print as rprint


# completion = """
# <think>\nWe are asked to design a solid-state synthesis route for KSr4B3O9. The target is a potassium strontium borate. We need to output a JSON with precursors, operations, and thermodynamic checks.\n\nFirst, we need to determine stoichiometry: K, Sr, B, O. Target formula: KSr4B3O9. Oxidation states: K is +1, Sr is +2, B is +3, O is -2. Check charge balance: 1*(+1) + 4*(+2) + 3*(+3) + 9*(-2) = 1+8+9-18 = 0. So balanced.\n\nPrecursors: We need sources of K, Sr, B, O. Common solid-state precursors: K2CO3 (or KNO3, KCl), SrCO3 (or SrO), B2O3. But B2O3 is a liquid at high temperatures, so we might use it as a powder. We need to balance equation. Let's write target: KSr4B3O9. We can use K2CO3, SrCO3, B2O3. Let's balance:\n\nK2CO3 -> 2 K, 1 C, 3 O\nSrCO3 -> 1 Sr, 1 C, 3 O\nB2O3 -> 2 B, 3 O\n\nWe need 1 K, 4 Sr, 3 B, 9 O. Let's set coefficients:\nK2CO3: x\nSrCO3: y\nB2O3: z\n\nFrom K: 2x = 1 => x = 0.5\nFrom Sr: y = 4\nFrom B: 2z = 3 => z = 1.5\nFrom O: 3x + 3y + 3z = 3*(0.5 + 4 + 1.5) = 3*6 = 18 O, but target has 9 O. Wait, the carbon and oxygen from precursors will be released as CO2 or other gases. The target only has O from borate. So the O balance is 3x + 3y + 3z = 9 O (from precursors) plus 3x + 3y + 3z O from CO2? Actually, the carbon in CO3^2- will be released as CO2. So the oxygen in the product comes from the oxides: the K, Sr, B are from precursors, but the oxygen in the product is all from the precursors (CO3^2- and B2O3). However, the carbon is released. So the equation is: 0.5 K2CO3 + 4 SrCO3 + 1.5 B2O3 -> KSr4B3O9 + 0.5 CO2 + 4 CO2 + 1.5 B2O3? Wait, B2O3 is already an oxide, so we just add it. The carbonates decompose to oxides and CO2. So the reaction: 0.5 K2CO3 + 4 SrCO3 + 1.5 B2O3 -> KSr4B3O9 + 0.5 CO2 + 4 CO2. Total CO2: 0.5 + 4 = 4.5 moles. That's fine. So the precursors are K2CO3, SrCO3, B2O3 in amounts 0.5, 4, 1.5 moles. We can scale to whole numbers: multiply by 2: K2CO3, SrCO3, B2O3 in 1, 8, 3 moles. Then target: 2 KSr4B3O9? Actually, 1 K2CO3 gives 2 K, 8 SrCO3 gives 8 Sr, 3 B2O3 gives 6 B, so 2 K, 8 Sr, 6 B. That's 2 KSr4B3O9 (2 K, 8 Sr, 6 B). So scaling up by 2, we get 2 KSr4B3O9. So we can use 1 K2CO3, 8 SrCO3, 3 B2O3. That's a reasonable choice. So precursors: [{\"formula\": \"K2CO3\", \"amount\": 1.0}, {\"formula\": \"SrCO3\", \"amount\": 8.0}, {\"formula\": \"B2O3\", \"amount\": 3.0}].\n\nAlternatively, we could use SrO and K2CO3 and B2O3, but SrCO3 is common. B2O3 is a liquid at high temperature, so we might need to use it as a powder or pre-melt. But solid-state route often uses solid precursors. We can use B2O3 as a powder.\n\nOperations: Typical solid-state synthesis: mix, dry (if needed), press, calcine, sinter, cool. Since we're using carbonates, we need to decompose them, so heating to ~800-1000\u00b0C. Borates often require high temperatures, maybe 1100\u00b0C. We can do a two-step: calcine at 800\u00b0C for 12h to decompose carbonates, then regrind, press, sinter at 1100\u00b0C for 24h, then cool. Atmosphere: air, since no redox needed. We need to specify temperatures and times. We'll design: mix (ball mill or mortar), dry (if any moisture), press (pelletize), calcine at 800\u00b0C for 12h, cool to room temperature, then sinter at 1100\u00b0C for 24h, then cool. Or we can do a single heating step. I'll go with two steps: calcine and sinter. Also, we might need a cool step to avoid rapid cooling causing cracking. We can include a cool operation at room temperature after sintering. We'll specify temperatures: calcine 800\u00b0C, sinter 1100\u00b0C, cool 25\u00b0C. Atmosphere: air. Time: calcine 12h, sinter 24h, cool 1h? Actually, cooling can be slow, but we'll just say cool for 1h. We need to output the operations array.\n\nThermodynamic checks: We need 3-6 claims. Since no phase diagram data is provided, we must infer based on general knowledge. The system is K-Sr-B-O. We can claim oxidation states: K is +1, Sr +2, B +3, O -2, all stable. No mixed valence. We can claim competing phases: maybe SrB6O11 (strontium borate), K2B4O9 (potassium tetraborate), SrO, K2CO3, SrCO3, B2O3, etc. But without data, we can't give formation energies. We can claim stoichiometric constraints: K2CO3, SrCO3, B2O3 are stable precursors. We can also claim that the target is stable? But we don't have data. However, we can say that the target is a known phase? I don't know. But we must output thermodynamic checks. We can claim oxidation state for K, Sr, B, O. We can claim competing phase like SrB6O11 (but no energy). We can claim stoichiometric constraint: O2 consumption? None, because we're in air, oxidation is not needed. We can claim a hull stability for the target? But we don't have formation energy. The prompt says: \"Reference only formulas that appear in the SYSTEM STABLE PHASES or DANGEROUS METASTABLE SIDE-PHASES sections of the prompt; do not invent phase names.\" Since no phase data is given, we cannot reference any specific phases. So we must rely on generic claims that don't require numeric values? But the schema requires numeric values for some fields like e_above_hull, requires_atmosphere, etc. For oxidation state claim, we can provide e_above_hull? Not needed. For competing phase claim, we can set e_above_hull to 0? But it's not provided. The prompt says: \"For numeric values, you must reference only values that appear in the prompt; do not invent numbers.\" Since no values are given, we can't provide numbers. However, the schema requires some numeric fields. We might have to omit those fields that require numbers if they are not in the prompt? But the schema says \"required\": [\"type\", \"formula\", \"temperature_range_C\", \"context\"]. For oxidation_state claim, we can provide 
# e_above_hull as 0? That's not justified. The prompt says: \"You must output 3-6 thermodynamic claims total, covering oxidation state, stoichiometric constraint, and/or competing phase types. For numeric values, you must reference only values that appear in the 
# prompt; do not invent numbers.\" Since no numbers are given, we cannot provide any numeric claims that require numbers. So we must only use claims that don't need numbers? The oxidation_state claim requires e_above_hull? Actually, the oxidation_state claim schema: \"oxidation_state\": {\"type\": \"str\", \"description\": \"Formula of the compound or element that has a specific oxidation state (e.g. 'Fe2O3' for Fe3+, 'FeO' for Fe2+).\"}, but the required fields are \"type\", \"formula\", \"temperature_range_C\", \"context\". \"e_above_hull\" is optional? The schema for \"competing_phase\" includes \"e_above_hull\" as optional. For oxidation_state, it doesn't have \"e_above_hull\". So we can just provide oxidation_state claim with formula and context. For stoichiometric constraint, it requires \"species\" and \"moles_per_formula_unit\". We could claim CO2 release: from decomposition of carbonates. We can say species CO2, moles_per_formula_unit 4.5 (for 1 mole of target, we get 4.5 moles of CO2). But we need to specify \"type\": \"stoichiometric_constraint\", \"species\": \"CO2\", \"moles_per_formula_unit\": 4.5, \"role\": \"released\". That's fine. The temperature_range_C could be the temperature at which the reaction occurs, e.g. 800-1100\u00b0C. We can set it to \"800-1100\". Atmosphere: \"open\" or \"air\". The prompt says: \"requires_atmosphere\" is optional. We can omit it. So we can do:\n\n1. oxidation_state: {\"type\": \"oxidation_state\", \"formula\": \"K2CO3\", \"temperature_range_C\": \"800-1100\", \"context\": \"K is +1 in K2CO3, stable.\"}\n2. oxidation_state: {\"type\": \"oxidation_state\", \"formula\": \"SrCO3\", \"temperature_range_C\": \"800-1100\", \"context\": \"Sr is +2 in SrCO3, stable.\"}\n3. oxidation_state: {\"type\": \"oxidation_state\", \"formula\": \"B2O3\", \"temperature_range_C\": \"800-1100\", \"context\": \"B is +3 in B2O3, stable.\"}\n4. stoichiometric_constraint: {\"type\": \"stoichiometric_constraint\", \"species\": \"CO2\", \"moles_per_formula_unit\": 4.5, \"role\": \"released\", \"temperature_range_C\": \"800-1100\", \"requires_atmosphere\": \"open\"}\n5. competing_phase: {\"type\": \"competing_phase\", \"formula\": \"SrB6O11\", \"temperature_range_C\": \"1000-1200\", \"e_above_hull\": 0.02} (but we don't know this energy, so we cannot include it). We can omit \"e_above_hull\" if it's optional. The schema for competing_phase says \"e_above_hull\" is optional. So we can include it without value? Probably better to omit. We can just provide formula, temperature_range_C, and maybe a context. But the schema says \"context\" is required. 
# We can put a generic context like \"likely forms at high temperature\". But we need to reference formulas from the prompt. Since no formulas are given, we can't reference any. So we can only use formulas we invent that are reasonable, like SrB6O11. However, the prompt says \"Reference only formulas that appear in the SYSTEM STABLE PHASES or DANGEROUS METASTABLE SIDE-PHASES sections of the prompt; do not invent phase names.\" Since there are no such sections, we cannot reference any formulas. So we cannot include any competing_phase claim with a formula. So we have to stick to oxidation_state and stoichiometric_constraint. We need 3-6 claims. We can do 4 claims: 3 oxidation_state and 1 stoichiometric_constraint. That meets the requirement.\n\nNow, we need to output the JSON with precursors, operations, thermodynamic_checks.\n\nPrecursors: we'll use the scaled amounts: 1 K2CO3, 8 SrCO3, 3 B2O3. We can set amounts in moles? The schema says \"amount\" is a float, likely in moles. So we can set 1.0, 8.0, 3.0.\n\nOperations: mix, dry, press, calcine, cool, sinter. We'll define each step. We'll include \"type\": \"mix\" with media \"mortar and pestle\", temperature 25, time 1h. Then dry? We could dry the mixture at 120\u00b0C for 1h to remove moisture. Then press into pellets. Then calcine at 800\u00b0C for 12h in air. Then cool to room temperature (25\u00b0C) for 1h.
#  Then sinter at 1100\u00b0C for 24h in air. Then cool to room temperature again. We can include a \"cool\" operation after sintering. 
#  We'll specify temperatures and times. Atmosphere for calcine and sinter: \"air\". We can also include a \"grind\" step after calcine? Not necessary. We'll stick to mix, dry, press, calcine, cool, sinter, cool. We'll output operations array with these steps.\n\nNow, we need to ensure that all operations are correctly formatted. The schema expects \"type\" (e.g., \"mix\", \"dry\", \"press\", \"calcine\", \"sinter\", \"cool\"), \"media\" (optional for mix, maybe \"air\"), \"temperature_C\" (float), \"time_h\" (float). For calcine, we'll set temperature 800, time 12. For sinter, temperature 1100, time 24. For cool, temperature 25, time 1. We'll include \"atmosphere\" field only for calcine and sinter? The schema says \"atmosphere\" is optional. For calcine, we can set \"atmosphere\": \"air\". For sinter, same. For mix, we 
#  can set \"atmosphere\": \"air\" if we're grinding in air. But it's not required. We'll include it. So operations:\n\n[\n  {\"type\": \"mix\", 
#  \"media\": \"mortar and pestle\", \"temperature_C\": 25, \"time_h\": 1.0, \"atmosphere\": \"air\"},\n  {\"type\": \"dry\", \"media\": \"oven\", 
#  \"temperature_C\": 120, \"time_h\": 1.0, \"atmosphere\": \"air\"},\n  {\"type\": \"press\", \"media\": \"pellet press\", \"temperature_C\": 25, 
#  \"time_h\": 0.1, \"atmosphere\": \"air\"},\n  {\"type\": \"calcine\", \"media\": \"alumina crucible\", \"temperature_C\": 800, \"time_h\": 12.0, 
#  \"atmosphere\": \"air\"},\n  {\"type\": \"cool\", \"media\": \"furnace\", \"temperature_C\": 25, \"time_h\": 1.0, \"atmosphere\": \"air\"},\n  {\"type\": \"sinter\", \"media\": \"alumina crucible\", \"temperature_C\": 1100, \"time_h\": 24.0, \"atmosphere\": \"air\"},\n  {\"type\": \"cool\", \"media\": \"furnace\", \"temperature_C\": 25, \"time_h\": 1.0, \"atmosphere\": \"air\"}\n]\n\nNow, we need to output the JSON with these sections.\n</think>\n\n{\n  \"precursors\": [\n    {\"formula\": \"K2CO3\", \"amount\": 1.0},\n    {\"formula\": \"SrCO3\", \"amount\": 8.0},\n    {\"formula\": \"B2O3\", \"amount\": 3.0}\n  ],\n  \"operations\": [\n    {\"type\": \"mix\", \"media\": \"mortar and pestle\", \"temperature_C\": 25, \"time_h\": 1.0, \"atmosphere\": \"air\"},\n    {\"type\": \"dry\", \"media\": \"oven\", \"temperature_C\": 120, \"time_h\": 1.0, \"atmosphere\": \"air\"},\n    {\"type\": \"press\", \"media\": \"pellet press\", \"temperature_C\": 25, \"time_h\": 0.1, \"atmosphere\": \"air\"},\n    {\"type\": \"calcine\", \"media\": \"alumina crucible\", \"temperature_C\": 800, \"time_h\": 12.0, \"atmosphere\": \"air\"},\n    {\"type\": \"cool\", \"media\": \"furnace\", \"temperature_C\": 25, \"time_h\": 1.0, \"atmosphere\": \"air\"},\n    {\"type\": \"sinter\", \"media\": \"alumina crucible\", \"temperature_C\": 1100, \"time_h\": 24.0, \"atmosphere\": \"air\"},\n    {\"type\": \"cool\", \"media\": \"furnace\", \"temperature_C\": 25, \"time_h\": 1.0, \"atmosphere\": \"air\"}\n  ],\n  \"thermodynamic_checks\": [\n    {\"type\": \"oxidation_state\", \"formula\": \"K2CO3\", \"temperature_range_C\": \"800-1100\", \"context\": \"K is +1 in K2CO3, stable.\"},\n    {\"type\": \"oxidation_state\", \"formula\": \"SrCO3\", \"temperature_range_C\": \"800-1100\", \"context\": \"Sr is +2 in SrCO3, stable.\"},\n    {\"type\": \"oxidation_state\", \"formula\": \"B2O3\", \"temperature_range_C\": \"800-1100\", \"context\": \"B is +3 in B2O3, stable.\"},\n    {\"type\": \"stoichiometric_constraint\", \"species\": \"CO2\", \"moles_per_formula_unit\": 4.5, \"role\": \"released\", \"temperature_range_C\": \"800-1100\", \"requires_atmosphere\": \"open\"}\n  ]\n}
# """

# target = "KSr4B3O9"
# # res = parse_completion(completion, target)
# # rprint(res)



import json, re
from collections import Counter

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

key_variants = Counter()
n_examples = 0
n_with_ops = 0
n_failed_json = 0
sample_ops = []


all_op_keys = Counter()

with open('eval_results/eval_v3-rank16-seed42_test.json') as f:
    data = json.load(f)

# Figure out the actual structure first
print("Top-level type:", type(data))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    # try common nesting patterns
    examples = data.get('records')
elif isinstance(data, list):
    examples = data
print("n examples found:", len(examples) if examples else 0)

if examples:
    for ex in examples[:200]:
        completion = ex.get('completion') or ex.get('output') or ex.get('generation') or ''
        n_examples += 1
        text_cleaned = THINK_RE.sub("", completion, count=1).strip()
        text_cleaned = FENCE_RE.sub("", text_cleaned.strip())
        start_idx = text_cleaned.find("{")
        end_idx = text_cleaned.rfind("}")
        if start_idx == -1 or end_idx == -1:
            n_failed_json += 1
            continue
        try:
            parsed = json.loads(text_cleaned[start_idx:end_idx+1])
        except json.JSONDecodeError:
            n_failed_json += 1
            continue
        ops = parsed.get('operations', [])
        if ops:
            n_with_ops += 1
            for op in ops:
                if isinstance(op, dict):
                    for k in op.keys():
                        if 'temp' in k.lower() or 'firing' in k.lower():
                            key_variants[k] += 1

                all_op_keys.update(op.keys())
            if len(sample_ops) < 3:
                sample_ops.append(ops[0])

print()
print("n_examples checked:", n_examples)
print("n_failed_json:", n_failed_json)
print("n_with_ops:", n_with_ops)
print()
print("Temperature-like key variants found:", key_variants)
print()
print("Sample first-operation dicts:")
for op in sample_ops:
    print(json.dumps(op, indent=2))
print("-"*20)
print("ALL OP KEYS:")
for op in all_op_keys:
    print(op)
