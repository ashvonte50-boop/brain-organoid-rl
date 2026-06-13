import pickle, os
base = r'C:\Users\Admin\brain-organoid-rl\ablation_results'

print('PILOT data (PILOT_*.pkl) - which showed ablation effects:')
for cname in ['FULL','ABLATE_M1','ABLATE_M5','ABLATE_M10']:
    p = os.path.join(base, f'PILOT_{cname}.pkl')
    if not os.path.exists(p):
        print(f'  PILOT_{cname}.pkl: MISSING'); continue
    with open(p,'rb') as f: d = pickle.load(f)
    print(f'  PILOT_{cname}: n={len(d)}')
    for i, s in enumerate(d):
        nat = s.get('natural', {})
        dai = nat.get('dai_core', None)
        rs  = nat.get('real_schema', None)
        print(f'    seed{i}: DAI={dai}  RS={rs}')

print()
print('10-seed data (VAL10_*.pkl) - which showed NO effect:')
for cname in ['FULL','ABLATE_M1','ABLATE_M5','ABLATE_M10']:
    p = os.path.join(base, f'VAL10_{cname}.pkl')
    if not os.path.exists(p):
        print(f'  VAL10_{cname}.pkl: MISSING'); continue
    with open(p,'rb') as f: d = pickle.load(f)
    print(f'  VAL10_{cname}: n={len(d)}')
    for i, s in enumerate(d[:3]):
        nat = s.get('natural', {})
        dai = nat.get('dai_core', None)
        rs  = nat.get('real_schema', None)
        print(f'    seed{i}: DAI={dai}  RS={rs}')
