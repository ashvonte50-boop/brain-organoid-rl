"""
TASK 5 RUNNER — Causal Role of Wcc
===================================
5 seeds, 1 training run each (4 conditions branched post-hoc). ~45 min.
"""
import os, sys, time, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task5'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [42, 1042, 2042, 3042, 4042]


def run_seed(seed):
    out = os.path.join(OUT_DIR, f'T5_seed{seed}.pkl')
    if os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        print(f'  seed={seed} [cached]', flush=True)
        return r
    log = out.replace('.pkl', '.log')
    cmd = [sys.executable, 'task5_worker.py', str(seed), '--prefix', 'T5']
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR, stdout=lf, stderr=subprocess.STDOUT)
    el = int(time.time() - t0)
    if proc.returncode == 0 and os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        c = r['conditions']
        print(f'  seed={seed} {el:4d}s  '
              f'FULL_Ret={c["FULL"]["retention_mean"]:.4f}  '
              f'DESTROY_Ret={c["WCC_DESTROY"]["retention_mean"]:.4f}  '
              f'ENHANCE_Ret={c["WCC_ENHANCE"]["retention_mean"]:.4f}', flush=True)
        return r
    print(f'  seed={seed} FAILED (exit={proc.returncode})', flush=True)
    return None


if __name__ == '__main__':
    print('TASK 5: CAUSAL ROLE OF Wcc', flush=True)
    print(f'Seeds={SEEDS}  (4 conditions branched post-hoc per seed)', flush=True)
    t0 = time.time()
    data = {}
    for s in SEEDS:
        r = run_seed(s)
        if r: data[s] = r
    with open(os.path.join(OUT_DIR, 'task5_combined.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min → run task5_analyze.py', flush=True)
