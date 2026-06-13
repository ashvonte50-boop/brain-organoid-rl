"""
Master launcher for M1-M5 tasks.
Runs in sequence: M1 -> M3 -> M4 -> M2 -> M5
Each has fault-tolerant resume logic -- safe to restart.
"""
import subprocess, sys, os, time

TASKS = [
    ('M1', 'm1_task105_20seeds.py'),
    ('M3', 'm3_dual_metrics.py'),
    ('M4', 'm4_null_model.py'),
    ('M2', 'm2_attractor_diagnostics.py'),
    ('M5', 'm5_encoding_order.py'),
]

BASE_DIR = r'C:\Users\Admin\brain-organoid-rl'
LOG_DIR  = os.path.join(BASE_DIR, 'm_task_logs')
os.makedirs(LOG_DIR, exist_ok=True)

print('=== M1-M5 MASTER LAUNCHER ===', flush=True)
print(f'Order: {[t[0] for t in TASKS]}', flush=True)
print(f'Logs: {LOG_DIR}', flush=True)

for task_name, script in TASKS:
    log_path = os.path.join(LOG_DIR, f'{task_name.lower()}_stdout.log')
    print(f'\n{"="*60}', flush=True)
    print(f'[LAUNCHER] Starting {task_name}: {script}', flush=True)
    print(f'[LAUNCHER] Log: {log_path}', flush=True)
    t0 = time.time()

    with open(log_path, 'w') as log_f:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, script)],
            cwd=BASE_DIR,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**os.environ, 'DEV_MODE': '1'},
        )

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f'[LAUNCHER] {task_name} COMPLETED in {elapsed/3600:.2f} hrs (exit 0)', flush=True)
    else:
        print(f'[LAUNCHER] {task_name} FAILED (exit {result.returncode}) after {elapsed/3600:.2f} hrs', flush=True)
        print(f'[LAUNCHER] Check log: {log_path}', flush=True)
        # Continue to next task despite failure

print(f'\n[LAUNCHER] ALL TASKS DONE', flush=True)
