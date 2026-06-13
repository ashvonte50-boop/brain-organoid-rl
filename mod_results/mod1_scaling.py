#!/usr/bin/env python3
"""
MOD-1: Scaling test — subprocess-per-run architecture.

Each (N, seed, condition) is run in a FRESH subprocess via mod1_worker.py.
This avoids all module-level singleton contamination (SynapticTags,
CorticalAutoencoder, structural mask cache, etc.) that caused prior failures.

Ladder: N = 1000, 1500, 2000
Per N: 3 seeds * 2 conditions = 6 runs.
Append-safe: completed rows are skipped on resume.
"""
import os, sys, time, json, subprocess
import pandas as pd
import psutil

OUT     = r'C:\Users\Admin\brain-organoid-rl\mod_results'
RESULTS = os.path.join(OUT, 'mod1_scaling_results.csv')
WORKER  = os.path.join(OUT, 'mod1_worker.py')
PYTHON  = sys.executable

LADDER = [1000, 1500, 2000]
SEEDS  = [42, 1042, 2042]

os.makedirs(OUT, exist_ok=True)


def is_done(N, seed, cond):
    if not os.path.exists(RESULTS):
        return False
    df = pd.read_csv(RESULTS)
    return ((df.N == N) & (df.seed == seed) & (df.condition == cond)).any()


def save_row(row):
    pd.DataFrame([row]).to_csv(
        RESULTS, mode='a', index=False,
        header=not os.path.exists(RESULTS))


def run_one(N, seed, use_replay):
    """Spawn a fresh subprocess for one run; parse JSON result."""
    cmd = [PYTHON, WORKER, str(N), str(seed), str(use_replay)]
    t0  = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    stdout = proc.stdout
    stderr = proc.stderr

    # Print worker output so it appears in our log
    for line in stdout.splitlines():
        print(f"    [worker] {line}", flush=True)
    if stderr.strip():
        print(f"    [worker stderr] {stderr[:500]}", flush=True)

    # Parse JSON result from last RESULT_JSON line
    result_line = None
    for line in reversed(stdout.splitlines()):
        if line.startswith('RESULT_JSON:'):
            result_line = line[len('RESULT_JSON:'):]
            break

    if result_line is None:
        return {
            'ok': False, 'mean_retention': None,
            'M0': None, 'M1': None, 'M2': None, 'M3': None,
            'time_sec': elapsed, 'mem_peak_MB': 0, 'available_RAM_GB': 0,
            'error': f'No RESULT_JSON in output. returncode={proc.returncode}. stderr={stderr[:300]}',
        }

    try:
        return json.loads(result_line)
    except Exception as e:
        return {
            'ok': False, 'mean_retention': None,
            'M0': None, 'M1': None, 'M2': None, 'M3': None,
            'time_sec': elapsed, 'mem_peak_MB': 0, 'available_RAM_GB': 0,
            'error': f'JSON parse error: {e}. line={result_line[:200]}',
        }


print("=" * 60, flush=True)
print("MOD-1: SCALING TEST (subprocess-per-run)", flush=True)
print("=" * 60, flush=True)
print(f"Ladder: {LADDER}", flush=True)
print(f"Seeds:  {SEEDS}", flush=True)
print(f"RAM available: {psutil.virtual_memory().available/1e9:.2f} GB", flush=True)
print(f"Worker: {WORKER}", flush=True)

N_MAX_REACHED = 0

for N in LADDER:
    print(f"\n=== N = {N} ===", flush=True)
    success_at_N = 0
    fail_at_N    = 0

    for seed in SEEDS:
        for cond_name, use_replay in [('FULL', True), ('NO_REPLAY', False)]:

            if is_done(N, seed, cond_name):
                print(f"  SKIP  N={N} seed={seed} {cond_name}", flush=True)
                success_at_N += 1
                continue

            avail = psutil.virtual_memory().available / 1e9
            if avail < 0.3:
                print(f"  RAM LOW ({avail:.2f} GB) — stopping", flush=True)
                fail_at_N += 99
                break

            print(f"  RUN   N={N} seed={seed} {cond_name} ...", flush=True)
            t0  = time.time()
            res = run_one(N, seed, use_replay)
            elapsed = time.time() - t0

            save_row({
                'N': N, 'seed': seed, 'condition': cond_name,
                'time_sec':         res['time_sec'],
                'mem_peak_MB':      res['mem_peak_MB'],
                'available_RAM_GB': res['available_RAM_GB'],
                'mean_retention':   res['mean_retention'],
                'M0_retention':     res['M0'],
                'M1_retention':     res['M1'],
                'M2_retention':     res['M2'],
                'M3_retention':     res['M3'],
                'ok':               res['ok'],
                'error':            res['error'],
            })

            if res['ok']:
                ret = res['mean_retention']
                print(f"  OK    N={N} seed={seed} {cond_name}: "
                      f"ret={ret:.4f}  ({elapsed:.0f}s)", flush=True)
                success_at_N += 1
            else:
                print(f"  FAIL  N={N} seed={seed} {cond_name}: "
                      f"{res['error'][:200]}", flush=True)
                fail_at_N += 1

    print(f"  N={N} summary: {success_at_N} OK, {fail_at_N} FAIL", flush=True)
    if success_at_N > 0:
        N_MAX_REACHED = N
    if fail_at_N >= len(SEEDS) * 2 * 0.5:
        print(f"  Majority failure at N={N}; stopping ladder.", flush=True)
        break

print(f"\n=== VERDICT ===", flush=True)
print(f"N_MAX_REACHED = {N_MAX_REACHED}", flush=True)

with open(os.path.join(OUT, 'mod1_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"MOD-1 N_MAX_REACHED = {N_MAX_REACHED}\n")
    if os.path.exists(RESULTS):
        df = pd.read_csv(RESULTS)
        for Nv in sorted(df['N'].unique()):
            sub = df[df.N == Nv]
            ok  = sub[sub.ok == True]
            if len(ok) > 0:
                full = ok[ok.condition == 'FULL']['mean_retention'].mean()
                nr   = ok[ok.condition == 'NO_REPLAY']['mean_retention'].mean()
                f.write(f"  N={Nv}: FULL={full:.3f}, NO_REPLAY={nr:.3f}, "
                        f"effect={full-nr:+.3f}, n_ok={len(ok)}\n")

print("[MOD-1] DONE", flush=True)
