"""
TASK 3 RUNNER — Schema Formation Dynamics
==========================================
2 conditions × 5 seeds = 10 runs (~100 min)
"""
import os, sys, time, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task3'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042]
CONDITIONS = [
    ('FULL',      True,  1.3),
    ('NO_REPLAY', False, 1.3),
]


def run_one(label, use_replay, boost, seed):
    out = os.path.join(OUT_DIR, f'T3_{label}_seed{seed}.pkl')
    if os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        traj = r.get('trajectory', [])
        final_s1 = traj[-1]['S1'] if traj else 0
        print(f'  {label:<12s} seed={seed} [cached] '
              f'checkpoints={len(traj)} S1_final={final_s1:.4f} '
              f'Ret={r["retention_mean"]:.4f}', flush=True)
        return r
    log = out.replace('.pkl', '.log')
    cmd = [sys.executable, 'task3_worker.py', label, '0', str(seed),
           '--prefix', 'T3',
           '--use_replay', '1' if use_replay else '0',
           '--boost_scale', str(boost)]
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = int(time.time() - t0)
    if proc.returncode == 0 and os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        traj = r.get('trajectory', [])
        final_s1 = traj[-1]['S1'] if traj else 0
        print(f'  {label:<12s} seed={seed} {elapsed:4d}s '
              f'checkpoints={len(traj)} S1_final={final_s1:.4f} '
              f'Ret={r["retention_mean"]:.4f}', flush=True)
        return r
    print(f'  {label:<12s} seed={seed} FAILED (exit={proc.returncode})', flush=True)
    return None


if __name__ == '__main__':
    print('TASK 3: SCHEMA FORMATION DYNAMICS', flush=True)
    print(f'Conditions: {[c[0] for c in CONDITIONS]}  Seeds: {SEEDS}', flush=True)
    print(f'Total runs: {len(CONDITIONS)*len(SEEDS)}  (~{len(CONDITIONS)*len(SEEDS)*10} min)', flush=True)

    t0 = time.time()
    data = {}
    for label, use_replay, boost in CONDITIONS:
        print(f'\n--- {label}  use_replay={use_replay} ---', flush=True)
        for seed in SEEDS:
            r = run_one(label, use_replay, boost, seed)
            if r: data[(label, seed)] = r

    with open(os.path.join(OUT_DIR, 'task3_combined.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min  →  run task3_analyze.py', flush=True)
