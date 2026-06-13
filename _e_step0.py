"""Step 0 characterization: verify M1 data + time a single FULL run + measure W_encode gradient."""
import os, sys, time
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')

# ---- 1. Verify M1 data for E2 reuse ----
m1 = r'C:\Users\Admin\brain-organoid-rl\m1_results\m1_task105_20seeds.csv'
df = pd.read_csv(m1)
print('=== M1 DATA (E2 first-20-seeds reuse check) ===', flush=True)
print('rows:', len(df), '| seeds:', len(df.seed.unique()), '| conds:', sorted(df.condition.unique()))
cells = df.groupby(['seed','condition']).size()
print('complete (seed x cond) cells:', cells.shape[0], 'of 60 expected (20 seeds x 3 cond)')
print('rows per cell (should be 4):', sorted(cells.unique()))
print('seeds:', sorted(df.seed.unique()), flush=True)

# ---- 2. Time a single FULL run + measure natural W_encode gradient ----
import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _CENTROID_LOG, _last_net

print('\n=== TIMING + W_encode GRADIENT (seed=42, FULL) ===', flush=True)
seed = 42
ccf.torch.manual_seed(seed); ccf.np.random.seed(seed)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
core_set = set(int(x) for x in core_mask.tolist())
ne = 750

# Hook to measure fast-weight W magnitude of each memory's UNIQUE block right after
# its own encoding (before later memories / replay touch it).
W_encode = {}
_net_ref = [None]
_orig_train = ccf.train_one_memory
_train_call = [0]
def _measure_train(net, assembly, **kw):
    _net_ref[0] = net
    j = _train_call[0]
    r = _orig_train(net, assembly, **kw)
    # measure unique block fast-weight mean immediately post-encode
    asm = assemblies[j]
    uniq = [int(x) for x in asm if int(x) not in core_set and int(x) < ne]
    with torch.no_grad():
        Wf = net.W.detach().cpu().numpy()
    W_encode[j] = float(Wf[np.ix_(uniq, uniq)].mean())
    _train_call[0] += 1
    return r
ccf.train_one_memory = _measure_train

t0 = time.time()
try:
    ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
finally:
    ccf.train_one_memory = _orig_train
runtime = time.time() - t0
net = _net_ref[0]

print(f'\nSingle seed runtime: {runtime:.1f} seconds', flush=True)
print('W_encode (fast-weight unique-block mean, post-encode, BEFORE replay):')
for j in range(4):
    print(f'  M{j}: {W_encode.get(j, float("nan")):.5f}')
grad_ok = W_encode.get(0,0) >= W_encode.get(3,0)
print(f'Gradient M0>=M3? {grad_ok}  (M0={W_encode.get(0,0):.5f}, M3={W_encode.get(3,0):.5f})')

# Also final retention + W_slow per memory for sanity
print('\nFinal retention + W_slow (post-experiment):')
with torch.no_grad():
    WS = net.W_slow.detach().cpu().numpy()
for j in range(4):
    asm = assemblies[j]
    uniq = [int(x) for x in asm if int(x) not in core_set and int(x) < ne]
    ret = float(ccf.probe_memory(net, asm)['isyn_score'])
    ws = float(WS[np.ix_(uniq, uniq)].mean())
    print(f'  M{j}: retention={ret:.4f}  W_slow_uniq={ws:.4f}')

print('\n=== STEP 0 DONE ===', flush=True)
