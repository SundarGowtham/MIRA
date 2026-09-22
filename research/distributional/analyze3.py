import json, re
from collections import Counter
M = {'base':'astral_gen_n32_base.json','sft':'astral_gen_n32_sft.json',
     'rs_sft':'astral_gen_n32_rs_sft.json','gdpo':'astral_gen_n32_gdpo_phase12_beta0.json'}
D = {k: {r['target']: r for r in json.load(open(v))['results']} for k,v in M.items()}
T=list(D['base'])
bare = {'Li2O','Na2O','K2O','Rb2O','Cs2O'}           # bare alkali oxides
carb = {'Li2CO3','Na2CO3','K2CO3','Rb2CO3','Cs2CO3','BaCO3','SrCO3','CaCO3','MgCO3'}
print("== precursor-type share (% of parsed samples containing ≥1) ==")
print(f"{'model':7s} {'bare alkali ox':>15s} {'carbonate':>10s} {'P2O5':>6s} {'B2O3':>6s} {'H3BO3':>6s}")
for m in M:
    s=[x for t in T for x in D[m][t]['samples'] if x.get('precursors')]
    f=lambda S: 100*sum(any(p in S for p in x['precursors']) for x in s)/len(s)
    print(f"{m:7s} {f(bare):14.1f}% {f(carb):9.1f}% {f({'P2O5','P4O10'}):5.1f}% {f({'B2O3'}):5.1f}% {f({'BH3O3','H3BO3'}):5.1f}%")
print()
print("== SFT superset matches ==")
for t in T:
    for x in D['sft'][t]['samples']:
        if 'SUPERSET' in x['match']: print(t, x['match'], x['precursors'], 'predicted=',D['sft'][t]['predicted'])
