"""
TASK 2.5: CORE NECESSITY TEST — 30 runs (~4 hr)
================================================
Conditions:
  FULL             (sanity-check baseline; matches Task 2 FULL exactly)
  NO_CORE_STIM    (primary necessity test: core never directly stimulated)
  HALF_STIM        (confound control: total injected current matched to NO_CORE_STIM)

Seeds: 42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042 (10 each)
Total: 30 runs.

Verification:
  FULL must reproduce Task 2 FULL (RS ~0.50, Ret ~0.286)
  NO_CORE_STIM should collapse RS (if hypothesis holds)
  HALF_STIM must still produce RS (rules out "less stim = less schema" confound)
"""
import os, sys, time, json, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task25'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]

CONDITIONS = [
    # (label,         intervention,    use_replay, boost_scale)
    ('FULL',          'FULL',          1, 1.3),
    ('NO_CORE_STIM',  'NO_CORE_STIM',  1, 1.3),
    ('HALF_STIM',     'HALF_STIM',     1, 1.3),
]


def run_one(label, intervention, use_replay, boost_scale, seed):
    out_path = os.path.join(OUT_DIR, f'T25_{label}_seed{seed}.pkl')
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            r = pickle.load(f)
        print(f'  {label:<14s} seed={seed}  [cached]  '
              f'RS={r["real_schema"]:.4f}  RS_perm={r["real_schema_permuted"]:.4f}  '
              f'Ret={r["retention_mean"]:.4f}  rep={r["replay_events"]}', flush=True)
        return r

    log = out_path.replace('.pkl', '.log')
    cmd = [sys.executable, 'task25_worker.py',
           label, '0', str(seed), '{}',
           '--prefix',       'T25',
           '--intervention', intervention,
           '--use_replay',   str(use_replay),
           '--boost_scale',  str(boost_scale)]
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}

    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR,
                              stdout=lf, stderr=subprocess.STDOUT)
    elapsed = int(time.time() - t0)

    if proc.returncode == 0 and os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            r = pickle.load(f)
        print(f'  {label:<14s} seed={seed}  {elapsed:4d}s  '
              f'RS={r["real_schema"]:.4f}  RS_perm={r["real_schema_permuted"]:.4f}  '
              f'Ret={r["retention_mean"]:.4f}  rep={r["replay_events"]}', flush=True)
        return r
    else:
        print(f'  {label:<14s} seed={seed}  FAILED (exit={proc.returncode}, log={log})',
              flush=True)
        return None


if __name__ == '__main__':
    print('TASK 2.5: CORE NECESSITY TEST', flush=True)
    print(f'Conditions: {[c[0] for c in CONDITIONS]}', flush=True)
    print(f'Seeds:      {SEEDS}', flush=True)
    print(f'Total runs: {len(CONDITIONS) * len(SEEDS)}  '
          f'(~{len(CONDITIONS)*len(SEEDS)*8} min)', flush=True)
    print(flush=True)

    t0 = time.time()
    data = {}
    for label, intervention, use_replay, boost_scale in CONDITIONS:
        print(f'\n--- {label}  intervention={intervention}  '
              f'use_replay={use_replay}  boost={boost_scale} ---', flush=True)
        for seed in SEEDS:
            r = run_one(label, intervention, use_replay, boost_scale, seed)
            if r is not None:
                data[(label, seed)] = r

    combined_path = os.path.join(OUT_DIR, 'task25_combined.pkl')
    with open(combined_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved combined -> {combined_path}', flush=True)
    print(f'\nAll runs complete in {(time.time()-t0)/60:.1f} min', flush=True)
    print('Now run: python task25_analyze.py', flush=True)
