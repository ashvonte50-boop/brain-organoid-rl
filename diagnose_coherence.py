"""
COHERENCE DIAGNOSTIC — ~5 min
Runs one seed and captures smooth_coh_last per replay event.
Answers: does coherence ever exceed COH_THR=0.50 during replay?
"""
import os, sys, warnings
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import torch

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE

SEED = 42
COH_THR = getattr(ccf, 'REPLAY_COHERENCE_THR', 0.5)
coh_values = []

orig = ccf._replay_one_event
def _wrapper(net, assembly, tags=None, **kw):
    result = orig(net, assembly, tags=tags, **kw)
    if isinstance(result, dict):
        coh = result.get('smooth_coh_last', None)
        if coh is not None:
            coh_values.append(float(coh))
    return result
ccf._replay_one_event = _wrapper

torch.manual_seed(SEED); np.random.seed(SEED)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
print(f'Running seed={SEED} ... (~5 min)', flush=True)
ccf.run_sequential_experiment(True, True, assemblies, SEED, ablation={})
ccf._replay_one_event = orig

print(f'\n{"="*55}')
print(f'COHERENCE DIAGNOSTIC  (COH_THR = {COH_THR})')
print(f'{"="*55}')
if not coh_values:
    print('smooth_coh_last NOT found in result dict.')
    print('Cannot read coherence from return value.')
else:
    arr = np.array(coh_values)
    print(f'Replay events captured : {len(arr)}')
    print(f'Coherence mean         : {arr.mean():.4f}')
    print(f'Coherence max          : {arr.max():.4f}')
    print(f'Coherence min          : {arr.min():.4f}')
    print(f'Events > COH_THR ({COH_THR}): {(arr > COH_THR).sum()} / {len(arr)}')
    print(f'Events > 0.30          : {(arr > 0.30).sum()} / {len(arr)}')
    print(f'Events > 0.10          : {(arr > 0.10).sum()} / {len(arr)}')
    print()
    pct_above = 100 * (arr > COH_THR).mean()
    if pct_above == 0:
        print(f'VERDICT: Coherence NEVER exceeds {COH_THR}.')
        print(f'=> M1–M10 gate condition is NEVER satisfied.')
        print(f'=> All mechanisms are dormant. Ablating them has zero effect.')
        print(f'=> Fix: lower REPLAY_COHERENCE_THR or use production-mode replay params.')
    else:
        print(f'VERDICT: Coherence exceeds {COH_THR} in {pct_above:.1f}% of events.')
        print(f'=> Mechanisms CAN fire. Ablation should produce measurable differences.')
    print()
    # Histogram
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print('Distribution:')
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = int(((arr >= lo) & (arr < hi)).sum())
        bar = '█' * count
        marker = ' ← COH_THR' if abs(lo - 0.5) < 0.05 else ''
        print(f'  [{lo:.1f}–{hi:.1f}): {count:3d}  {bar}{marker}')

print(f'\nCCF constants:')
for c in ['REPLAY_COHERENCE_THR','STDP_GATE_BIAS','STDP_GATE_SLOPE',
          'REPLAY_NOISE', 'N_REPLAY_EVENTS','DEV_MODE']:
    print(f'  {c:<30} = {getattr(ccf, c, "NOT FOUND")}')
