#!/usr/bin/env python3
"""Sequentially run all remaining MOD experiments.

Order: MOD-1 (scaling) -> MOD-5 sweep (bio range) -> MOD-4 (iSTDP) -> MOD-3 (15-seed
learned schema). Each step writes its own log; pipeline continues even on rc!=0
so partial CSVs survive crashes.
"""
import os, sys, time, subprocess

BASE = r'C:\Users\Admin\brain-organoid-rl'
SCRIPTS = [
    ('MOD-1',   os.path.join('mod_results', 'mod1_scaling.py'),
     os.path.join(BASE, 'mod_results', 'mod1.log')),
    ('MOD-5sw', os.path.join('mod_results', 'mod5_bio_sweep.py'),
     os.path.join(BASE, 'mod_results', 'mod5.log')),
    ('MOD-4',   os.path.join('mod_results', 'mod4_istdp.py'),
     os.path.join(BASE, 'mod_results', 'mod4.log')),
    ('MOD-3',   os.path.join('mod_results', 'mod3_learned_schema_15seeds.py'),
     os.path.join(BASE, 'mod_results', 'mod3.log')),
]

results = {}
t_global = time.time()
for name, script, log in SCRIPTS:
    print(f"\n{'='*60}\nSTARTING {name}\n{'='*60}\n", flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        rc = subprocess.call([sys.executable, os.path.join(BASE, script)],
                             cwd=BASE, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = 'PASS' if rc == 0 else f'FAIL(rc={rc})'
    results[name] = (status, elapsed)
    print(f"[{name}] {status} in {elapsed/3600:.2f}h", flush=True)

print(f"\n{'='*60}\nFINAL STATUS\n{'='*60}", flush=True)
for name, (status, elapsed) in results.items():
    print(f"  {name}: {status} in {elapsed/3600:.2f}h", flush=True)
print(f"  TOTAL: {(time.time()-t_global)/3600:.2f}h", flush=True)
