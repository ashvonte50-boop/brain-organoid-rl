"""Run Task 7 worker across 3 seeds, then analyze."""
import subprocess, sys, os, time

SEEDS = [42, 1042, 2042]
WORKER = os.path.join(os.path.dirname(__file__), 'task7_worker.py')
ANALYZE = os.path.join(os.path.dirname(__file__), 'task7_analyze.py')

for seed in SEEDS:
    print(f'\n{"="*60}')
    print(f'Running seed {seed}')
    print('='*60)
    t0 = time.time()
    r = subprocess.run([sys.executable, WORKER, str(seed)], check=False)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f'WORKER FAILED for seed {seed} (exit {r.returncode})')
    else:
        print(f'seed {seed} done in {elapsed/60:.1f} min')

print('\nRunning analysis...')
subprocess.run([sys.executable, ANALYZE], check=False)
print('Task 7 complete.')
