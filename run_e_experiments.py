"""
Master launcher for the 3 decisive experiments E1-E3.
Order: E2 (most direct, reuses Task 10.5) -> E1 (decisive 2x2) -> E3 (largest, new code).
Each script has fault-tolerant resume logic -- safe to restart.
"""
import subprocess, sys, os, time

TASKS = [
    ('E2', 'e2_task105_30seeds.py'),
    ('E1', 'e1_boost_adequate_seed.py'),
    ('E3', 'e3_learned_schema.py'),
]
BASE_DIR = r'C:\Users\Admin\brain-organoid-rl'
LOG_DIR  = os.path.join(BASE_DIR, 'e_task_logs')
os.makedirs(LOG_DIR, exist_ok=True)

print('=== E1-E3 MASTER LAUNCHER ===', flush=True)
print(f'Order: {[t[0] for t in TASKS]}', flush=True)

for task_name, script in TASKS:
    log_path = os.path.join(LOG_DIR, f'{task_name.lower()}_stdout.log')
    print(f'\n{"="*60}', flush=True)
    print(f'[LAUNCHER] Starting {task_name}: {script}', flush=True)
    t0 = time.time()
    with open(log_path, 'w') as log_f:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, script)],
            cwd=BASE_DIR, stdout=log_f, stderr=subprocess.STDOUT,
            env={**os.environ, 'DEV_MODE': '1'},
        )
    elapsed = time.time() - t0
    status = 'COMPLETED' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'[LAUNCHER] {task_name} {status} in {elapsed/3600:.2f} hrs', flush=True)

print(f'\n[LAUNCHER] ALL E-EXPERIMENTS DONE', flush=True)
