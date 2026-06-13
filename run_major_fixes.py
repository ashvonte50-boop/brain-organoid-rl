#!/usr/bin/env python3
"""
Master launcher for all 5 MAJOR fixes.
Runs in order: MAJOR-3 → MAJOR-4 → MAJOR-1 → MAJOR-2 → MAJOR-5
"""
import os, sys, time, subprocess

SCRIPTS = [
    ('MAJOR-3', 'major3_wslow_panel.py',        'W_slow heatmap panel (no sims)'),
    ('MAJOR-4', 'major4_narrative_reframe.py',   'Narrative reframe (no sims)'),
    ('MAJOR-1', 'major1_scheduling_test.py',     'Scheduling artifact test (45 runs)'),
    ('MAJOR-2', 'major2_behavioral_readout.py',  'Behavioral readout (30 runs)'),
    ('MAJOR-5', 'major5_benna_fusi.py',          'Benna-Fusi cascade (20 runs)'),
]

BASE = r'C:\Users\Admin\brain-organoid-rl'
t_global = time.time()

print("=" * 70)
print("MASTER LAUNCHER: ALL 5 MAJOR FIXES")
print("=" * 70)

results = {}

for name, script, desc in SCRIPTS:
    path = os.path.join(BASE, script)
    print(f"\n{'='*70}")
    print(f"STARTING {name}: {desc}")
    print(f"Script: {path}")
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    try:
        rc = subprocess.call([sys.executable, path], cwd=BASE)
        elapsed = time.time() - t0
        status = 'PASS' if rc == 0 else f'FAIL (rc={rc})'
        results[name] = (status, elapsed)
        print(f"\n[{name}] {status} in {elapsed/3600:.2f}h")
    except Exception as e:
        elapsed = time.time() - t0
        results[name] = (f'ERROR: {e}', elapsed)
        print(f"\n[{name}] ERROR: {e} after {elapsed/3600:.2f}h")

total = time.time() - t_global

print("\n" + "=" * 70)
print("MAJOR FIX STATUS TABLE")
print("=" * 70)
print(f"{'Task':<12} {'Status':<20} {'Time':<12}")
print("-" * 44)
for name, script, desc in SCRIPTS:
    status, elapsed = results.get(name, ('NOT RUN', 0))
    print(f"{name:<12} {status:<20} {elapsed/3600:.2f}h")
print("-" * 44)
print(f"{'TOTAL':<12} {'':<20} {total/3600:.2f}h")
print("=" * 70)
