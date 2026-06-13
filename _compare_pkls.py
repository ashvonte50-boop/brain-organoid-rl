import pickle, os
base = r'C:\Users\Admin\brain-organoid-rl\ablation_results'
print('10-seed PKLs - seed 0 (seed=42) values:')
for cname in ['FULL','ABLATE_M5','ABLATE_M1','ABLATE_M10']:
    p = os.path.join(base, f'VAL10_{cname}_seed0.pkl')
    if not os.path.exists(p):
        print(f'  {cname}: MISSING')
        continue
    with open(p,'rb') as f: d = pickle.load(f)
    nat = d.get('natural', {})
    dai = nat.get('dai_core', 0)
    rs  = nat.get('real_schema', 0)
    ret = nat.get('retention_A', 0)
    print(f'  {cname:<12}: DAI={dai:.4f}  RS={rs:.4f}  Ret_A={ret:.4f}')

print()
print('Diagnostic seed=42 in-process (just now):')
print('  FULL        : DAI=0.9349  RS=0.4289  Ret_A=0.3427')
print('  ABLATE_M5   : DAI=0.9368  RS=0.4301  Ret_A=0.3570')
print('  ABLATE_M1   : DAI=0.9364  RS=0.4413  Ret_A=0.3479')
print('  ABLATE_M10  : DAI=0.9412  RS=0.4490  Ret_A=0.3482')
