import json, re
from collections import Counter
M = {'base':'astral_gen_n32_base.json','sft':'astral_gen_n32_sft.json',
     'rs_sft':'astral_gen_n32_rs_sft.json','gdpo':'astral_gen_n32_gdpo_phase12_beta0.json'}
D = {k: {r['target']: r for r in json.load(open(v))['results']} for k,v in M.items()}
T=list(D['base'])
def fam(t):
    if 'P' in re.sub(r'Pb|Pr|Pd|Pt|Pm|Po|Pu','',t): return 'phosphate'
    if 'B' in re.sub(r'Ba|Bi|Be|Br|Bk','',t): return 'borate'
    return 'other oxide'
def key(ps): return tuple(sorted(ps))

print("== match-category share of all samples, per model ==")
for m in M:
    c=Counter(x['match'] for t in T for x in D[m][t]['samples']); n=sum(c.values())
    print(f"{m:7s} n={n:4d}  " + "  ".join(f"{k}={c[k]:4d} ({100*c[k]/n:4.1f}%)" for k in ['PREDICTED','TRADITIONAL','OTHER']) + f"  other_labels={ {k:v for k,v in c.items() if k not in ('PREDICTED','TRADITIONAL','OTHER')} }")
print()
print("== same, by family ==")
for f in ['phosphate','borate','other oxide']:
    for m in M:
        c=Counter(x['match'] for t in T if fam(t)==f for x in D[m][t]['samples']); n=sum(c.values())
        print(f"{f:11s} {m:7s} " + "  ".join(f"{k[:4]}={100*c[k]/n:4.1f}%" for k in ['PREDICTED','TRADITIONAL','OTHER']))
    print()
print("== ammonium precursor (N+H in formula) usage on PHOSPHATE targets, % of samples ==")
def ammon(p):
    return bool(re.search(r'N',p)) and bool(re.search(r'H',p)) and 'NO3' not in p
for m in M:
    s=[x for t in T if fam(t)=='phosphate' for x in D[m][t]['samples']]
    a=sum(any(ammon(p) for p in x['precursors']) for x in s)
    h3=sum(any(p in ('PH3O4','H3PO4') for p in x['precursors']) for x in s)
    print(f"{m:7s} ammonium-P {100*a/len(s):5.1f}%   H3PO4 {100*h3/len(s):5.1f}%   (n={len(s)})")
print()
print("== OTHER routes: how many (target,set) pairs are NEW vs base (never proposed by base)? ==")
bset={t:{key(x['precursors']) for x in D['base'][t]['samples']} for t in T}
for m in ['sft','rs_sft','gdpo']:
    oth=[(t,key(x['precursors'])) for t in T for x in D[m][t]['samples'] if x['match']=='OTHER']
    new=[p for p in oth if p[1] not in bset[p[0]]]
    print(f"{m:7s} OTHER samples={len(oth):4d}  of which set never seen in base={len(new):4d} ({100*len(new)/max(1,len(oth)):4.1f}%)  distinct new pairs={len(set(new))}")
print()
print("== GDPO most common OTHER sets (top 12) ==")
c=Counter((t,' + '.join(key(x['precursors']))) for t in T for x in D['gdpo'][t]['samples'] if x['match']=='OTHER')
for (t,s),n in c.most_common(12): print(f"{n:3d}  {t:14s} {s}   {'NEW vs base' if tuple(s.split(' + ')) not in bset[t] else ''}")
print()
print("== sampling noise: P(observe 0 of 32 | true rate p) ==")
for p in [1/64,1/32,2/32,3/32,5/32]:
    print(f"true p={p:.3f}  P(0/32)={(1-p)**32:.2f}")
print()
print("== mean validator reward and max-T by category, gdpo ==")
for cat in ['PREDICTED','TRADITIONAL','OTHER']:
    s=[x for t in T for x in D['gdpo'][t]['samples'] if x['match']==cat]
    r=[x['reward'] for x in s if x.get('reward') is not None]; mt=[x['max_T'] for x in s if x.get('max_T')]
    print(f"{cat:11s} n={len(s):4d} reward={sum(r)/len(r):.3f} maxT={sum(mt)/len(mt):.0f}")
