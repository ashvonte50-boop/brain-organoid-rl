"""
TASK 4 RUNNER — Mechanism Discovery
====================================
2 conditions × 3 seeds = 6 instrumented runs (~75 min)
"""
import os, sys, time, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task4'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042]
CONDITIONS = [('FULL', True), ('NO_REPLAY', False)]


def run_one(label, use_replay, seed):
    out = os.path.join(OUT_DIR, f'T4_{label}_seed{seed}.pkl')
    if os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        print(f'  {label:<12s} seed={seed} [cached] events={r["replay_events"]} '
              f'coinc_cc={r["coinc_cc"]:.3f} Ret={r["retention_mean"]:.4f}', flush=True)
        return r
    log = out.replace('.pkl', '.log')
    cmd = [sys.executable, 'task4_worker.py', label, str(seed),
           '--prefix', 'T4', '--use_replay', '1' if use_replay else '0',
           '--boost_scale', '1.3']
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR, stdout=lf, stderr=subprocess.STDOUT)
    el = int(time.time() - t0)
    if proc.returncode == 0 and os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        print(f'  {label:<12s} seed={seed} {el:4d}s events={r["replay_events"]} '
              f'coinc_cc={r["coinc_cc"]:.3f} coinc_uu={r["coinc_uu"]:.3f} '
              f'pot_cc={r["stdp"]["pot_cc"]:.2f} Ret={r["retention_mean"]:.4f}', flush=True)
        return r
    print(f'  {label:<12s} seed={seed} FAILED (exit={proc.returncode})', flush=True)
    return None


if __name__ == '__main__':
    print('TASK 4: MECHANISM DISCOVERY', flush=True)
    print(f'Conditions={[c[0] for c in CONDITIONS]} Seeds={SEEDS}', flush=True)
    print(f'Total: {len(CONDITIONS)*len(SEEDS)} runs (~75 min)', flush=True)
    t0 = time.time()
    data = {}
    for label, ur in CONDITIONS:
        print(f'\n--- {label} ---', flush=True)
        for s in SEEDS:
            r = run_one(label, ur, s)
            if r: data[(label, s)] = r
    with open(os.path.join(OUT_DIR, 'task4_combined.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min → run task4_analyze.py', flush=True)
