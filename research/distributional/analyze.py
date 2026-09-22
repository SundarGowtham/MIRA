import json, re
from collections import Counter, defaultdict
M = {'base':'astral_gen_n32_base.json','sft':'astral_gen_n32_sft.json',
     'rs_sft':'astral_gen_n32_rs_sft.json','gdpo':'astral_gen_n32_gdpo_phase12_beta0.json'}
D = {k: {r['target']: r for r in json.load(open(v))['results']} for k,v in M.items()}
targets = list(D['base'].keys())

def fam(t):
    if 'P' in re.sub(r'Pb|Pr|Pd|Pt|Pm|Po|Pu','',t): return 'phosphate'
    if 'B' in re.sub(r'Ba|Bi|Be|Br|Bk','',t): return 'borate'
    return 'other oxide'

def key(ps): return tuple(sorted(ps))
rows=[]
for t in targets:
    row={'t':t,'fam':fam(t)}
    for m in M:
        s=D[m][t]['samples']
        row[m+'_n']=len(s)
        row[m+'_pred']=sum(x['match']=='PREDICTED' for x in s)
        row[m+'_trad']=sum(x['match']=='TRADITIONAL' for x in s)
        row[m+'_sets']=Counter(key(x['precursors']) for x in s if x.get('precursors'))
    rows.append(row)

# 1. which targets get ANY predicted hit, per model
hit={m:{r['t'] for r in rows if r[m+'_pred']>0} for m in M}
print("== targets with >=1 PREDICTED sample ==")
for m in M: print(f"{m:7s} {len(hit[m]):2d}  {sorted(hit[m])}")
print()
print("base ∩ rs_sft:", len(hit['base']&hit['rs_sft']), sorted(hit['base']&hit['rs_sft']))
print("base only (lost by rs_sft):", sorted(hit['base']-hit['rs_sft']))
print("rs_sft only (not in base):", sorted(hit['rs_sft']-hit['base']))
print("gdpo only vs rs_sft:", sorted(hit['gdpo']-hit['rs_sft']))
print("gdpo gains in base?:", sorted((hit['gdpo']-hit['rs_sft'])&hit['base']))
print("union all models:", len(set().union(*hit.values())))
print()
# 2. per-target predicted frequency table
print("== predicted-hit count out of 32, per target (only targets any model hits) ==")
print(f"{'target':16s} {'fam':11s} base  sft rsft gdpo")
for r in sorted(rows,key=lambda r:(r['fam'],r['t'])):
    if any(r[m+'_pred'] for m in M):
        print(f"{r['t']:16s} {r['fam']:11s} {r['base_pred']:4d} {r['sft_pred']:4d} {r['rs_sft_pred']:4d} {r['gdpo_pred']:4d}")
print()
# 3. family breakdown
print("== by family: #targets with predicted hit / #targets, and total predicted samples ==")
for f in ['phosphate','borate','other oxide']:
    rs=[r for r in rows if r['fam']==f]
    line=f"{f:11s} n={len(rs):2d} | "
    for m in M:
        line+=f"{m}: {sum(r[m+'_pred']>0 for r in rs)} tgt/{sum(r[m+'_pred'] for r in rs):3d} smp  "
    print(line)
    line=f"{'':11s}  trad  | "
    for m in M:
        line+=f"{m}: {sum(r[m+'_trad']>0 for r in rs)} tgt/{sum(r[m+'_trad'] for r in rs):3d} smp  "
    print(line)
print()
# 4. solution-space size: distinct precursor sets per target
print("== distinct precursor sets per target (mean over 35) and total unique (target,set) pairs ==")
for m in M:
    ds=[len(r[m+'_sets']) for r in rows]
    print(f"{m:7s} mean distinct/target={sum(ds)/len(ds):.2f}  total unique pairs={sum(ds)}")
print()
# 5. overlap of solution sets between models (Jaccard per target, averaged)
def jac(a,b):
    a,b=set(a),set(b); return len(a&b)/len(a|b) if a|b else 1.0
for a,b in [('base','sft'),('base','rs_sft'),('rs_sft','gdpo'),('base','gdpo'),('sft','gdpo')]:
    js=[jac(r[a+'_sets'],r[b+'_sets']) for r in rows]
    print(f"Jaccard {a:6s} vs {b:6s}: mean {sum(js)/len(js):.2f}")
json.dump([{k:(dict((' + '.join(kk),vv) for kk,vv in v.items()) if isinstance(v,Counter) else v) for k,v in r.items()} for r in rows], open('per_target.json','w'), indent=1)
