"""
TASK 2: REPLAY NECESSITY TEST — 40 runs (~5.3 hr)
==================================================
Determines whether replay is necessary for schema formation.

Conditions:
  FULL              use_replay=True,  boost_scale=1.3
  FULL_NO_MB        use_replay=True,  boost_scale=1.0
  NO_REPLAY         use_replay=False, boost_scale=1.3 (unused: wrapper never fires)
  NO_REPLAY_NO_MB   use_replay=False, boost_scale=1.0 (unused)

Seeds: 42, 1042, 2042, ..., 9042 (10 total)

Verification: replay_events must be exactly 0 for NO_REPLAY conditions.
"""
import os, sys, time, json, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task2'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]

CONDITIONS = [
    # (label,              use_replay, boost_scale)
    ('FULL',               True,       1.3),
    ('FULL_NO_MB',         True,       1.0),
    ('NO_REPLAY',          False,      1.3),
    ('NO_REPLAY_NO_MB',    False,      1.0),
]


def run_one(label, use_replay, boost_scale, seed):
    out_path = os.path.join(OUT_DIR, f'T2_{label}_seed{seed}.pkl')
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            r = pickle.load(f)
        print(f'  {label:<18s} seed={seed}  [cached]  '
              f'RS={r["real_schema"]:.4f}  Ret={r["retention_mean"]:.4f}  '
              f'rep={r["replay_events"]}', flush=True)
        return r

    log = out_path.replace('.pkl', '.log')
    cmd = [sys.executable, 'task2_worker.py',
           label, '0', str(seed), '{}',
           '--prefix',      'T2',
           '--use_replay',  '1' if use_replay else '0',
           '--boost_scale', str(boost_scale)]
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}

    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR,
                              stdout=lf, stderr=subprocess.STDOUT)
    elapsed = int(time.time() - t0)

    if proc.returncode == 0 and os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            r = pickle.load(f)
        print(f'  {label:<18s} seed={seed}  {elapsed:4d}s  '
              f'RS={r["real_schema"]:.4f}  Ret={r["retention_mean"]:.4f}  '
              f'rep={r["replay_events"]}', flush=True)
        return r
    else:
        print(f'  {label:<18s} seed={seed}  FAILED (exit={proc.returncode}, log={log})',
              flush=True)
        return None


if __name__ == '__main__':
    print('TASK 2: REPLAY NECESSITY TEST', flush=True)
    print(f'Conditions: {[c[0] for c in CONDITIONS]}', flush=True)
    print(f'Seeds:      {SEEDS}', flush=True)
    print(f'Total runs: {len(CONDITIONS) * len(SEEDS)}  '
          f'(~{len(CONDITIONS)*len(SEEDS)*8} min)', flush=True)
    print(flush=True)

    t0 = time.time()
    data = {}
    for label, use_replay, boost_scale in CONDITIONS:
        print(f'\n--- {label}  use_replay={use_replay}  boost={boost_scale} ---',
              flush=True)
        for seed in SEEDS:
            r = run_one(label, use_replay, boost_scale, seed)
            if r is not None:
                data[(label, seed)] = r

    # Save combined snapshot
    combined_path = os.path.join(OUT_DIR, 'task2_combined.pkl')
    with open(combined_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved combined -> {combined_path}', flush=True)
    print(f'\nAll runs complete in {(time.time()-t0)/60:.1f} min', flush=True)
    print('Now run: python task2_analyze.py', flush=True)
