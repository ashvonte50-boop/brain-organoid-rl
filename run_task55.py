"""
TASK 5.5 RUNNER — Formation-time Causal Test of Wcc
=====================================================
2 seeds, 4 conditions each trained from scratch.  ~30-40 min.
"""
import os, sys, time, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task55'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [42, 1042]


def run_seed(seed):
    out = os.path.join(OUT_DIR, f'T55_seed{seed}.pkl')
    if os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        print(f'  seed={seed} [cached]', flush=True)
        return r
    log = out.replace('.pkl', '.log')
    cmd = [sys.executable, 'task55_worker.py', str(seed), '--prefix', 'T55']
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR, stdout=lf, stderr=subprocess.STDOUT)
    el = int(time.time() - t0)
    if proc.returncode == 0 and os.path.exists(out):
        with open(out, 'rb') as f: r = pickle.load(f)
        c = r['conditions']
        print(f'  seed={seed} {el:4d}s  '
              f'FULL={c["FULL"]["retention_mean"]:.4f}  '
              f'FROZEN={c["WCC_FROZEN"]["retention_mean"]:.4f}  '
              f'ZERO={c["WCC_CLAMPED_ZERO"]["retention_mean"]:.4f}  '
              f'NO_STDP={c["WCC_NO_STDP"]["retention_mean"]:.4f}', flush=True)
        return r
    # Print last 30 lines of log on failure
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
    print('TASK 5.5: FORMATION-TIME CAUSAL TEST OF Wcc', flush=True)
    print(f'Seeds={SEEDS}  (4 conditions trained independently per seed)', flush=True)
    t0 = time.time()
    data = {}
    for s in SEEDS:
        r = run_seed(s)
        if r: data[s] = r
    with open(os.path.join(OUT_DIR, 'task55_combined.pkl'), 'wb') as f:
        pickle.dump(data, f)
    elapsed = (time.time() - t0) / 60
    print(f'\nDone in {elapsed:.1f} min — {len(data)}/{len(SEEDS)} seeds succeeded', flush=True)
    print('Run task55_analyze.py to generate figures and report.', flush=True)
