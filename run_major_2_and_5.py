#!/usr/bin/env python3
"""Run MAJOR-2 then MAJOR-5 sequentially. Called after MAJOR-1 completed."""
import os, sys, time, subprocess

BASE = r'C:\Users\Admin\brain-organoid-rl'
SCRIPTS = [
    ('MAJOR-2', 'major2_behavioral_readout.py'),
    ('MAJOR-5', 'major5_benna_fusi.py'),
]

results = {}
t_global = time.time()

for name, script in SCRIPTS:
    path = os.path.join(BASE, script)
    print(f"\n{'='*60}\nSTARTING {name}\n{'='*60}\n", flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable, path], cwd=BASE)
    elapsed = time.time() - t0
    status = 'PASS' if rc == 0 else f'FAIL(rc={rc})'
    results[name] = (status, elapsed)
    print(f"\n[{name}] {status} in {elapsed/3600:.2f}h\n", flush=True)

total = time.time() - t_global
print(f"\n{'='*60}\nFINAL STATUS\n{'='*60}")
for name, (status, elapsed) in results.items():
    print(f"  {name}: {status} in {elapsed/3600:.2f}h")
print(f"  TOTAL: {total/3600:.2f}h")
