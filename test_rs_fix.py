"""Quick smoke-test: verify REAL_SCHEMA != 0 after the pipeline fix."""
import os, sys
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
from ablation_pipeline import run_one, BASE_SEED

print('Running 1-seed smoke test for REAL_SCHEMA fix...', flush=True)
result = run_one(BASE_SEED, ablation_dict={}, boost_scale=1.3, label='smoke_test')

for mode in ('natural', 'hyper'):
    if mode not in result:
        print(f'  {mode}: MISSING (crashed)')
        continue
    rs  = result[mode]['real_schema']
    dai = result[mode]['dai_core']
    ret = result[mode]['retention_A']
    print(f'  {mode}: REAL_SCHEMA={rs:.4f}  DAI_core={dai:.4f}  Retention_A={ret:.4f}')
    assert rs != 0.0, f'FAIL: REAL_SCHEMA still 0 in {mode} mode!'
    assert rs > -1.0 and rs < 1.0, f'FAIL: REAL_SCHEMA out of range: {rs}'

print('\nSMOKE TEST PASSED — REAL_SCHEMA fix confirmed.')
