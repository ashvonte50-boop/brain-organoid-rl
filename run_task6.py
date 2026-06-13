"""
TASK 6 RUNNER — True Replay-Protected Memory Substrate
=======================================================
3 seeds, 1 training run each (6 interventions branched post-hoc). ~42 min.
"""
import os, sys, time, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task6'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [42, 1042, 2042]


def run_seed(seed):
    out = os.path.join(OUT_DIR, f'T6_seed{seed}.pkl')
    if os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        print(f'  seed={seed} [cached]', flush=True)
        return r
    log = out.replace('.pkl', '.log')
    cmd = [sys.executable, 'task6_worker.py', str(seed), '--prefix', 'T6']
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR, stdout=lf, stderr=subprocess.STDOUT)
    el = int(time.time() - t0)
    if proc.returncode == 0 and os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        c = r['conditions']
        print(f'  seed={seed} {el:4d}s  '
              f'CTRL={c["CONTROL"]["retention_mean"]:.4f}  '
              f'-WUC={c["DESTROY_WUC"]["retention_mean"]:.4f}  '
              f'-WUU={c["DESTROY_WUU"]["retention_mean"]:.4f}  '
              f'-WUC_WUU={c["DESTROY_WUC_WUU"]["retention_mean"]:.4f}  '
              f'-ALL={c["DESTROY_ALL"]["retention_mean"]:.4f}', flush=True)
        return r
    try:
        with open(log, encoding='utf-8') as lf:
            lines = lf.readlines()
        print(f'  seed={seed} FAILED (exit={proc.returncode}) — last log lines:', flush=True)
        for ln in lines[-30:]:
            print('    ' + ln.rstrip(), flush=True)
    except Exception:
        print(f'  seed={seed} FAILED (exit={proc.returncode})', flush=True)
    return None


if __name__ == '__main__':
    print('TASK 6: TRUE REPLAY-PROTECTED MEMORY SUBSTRATE', flush=True)
    print(f'Seeds={SEEDS}  (6 interventions branched post-hoc per seed)', flush=True)
    t0 = time.time()
    data = {}
    for s in SEEDS:
        r = run_seed(s)
        if r: data[s] = r
    with open(os.path.join(OUT_DIR, 'task6_combined.pkl'), 'wb') as f:
        pickle.dump(data, f)
    elapsed = (time.time() - t0) / 60
    print(f'\nDone in {elapsed:.1f} min — {len(data)}/{len(SEEDS)} seeds succeeded', flush=True)
    print('Run task6_analyze.py to generate figures and report.', flush=True)
