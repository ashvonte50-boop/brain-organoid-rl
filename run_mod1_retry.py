#!/usr/bin/env python3
"""Re-run MOD-1 scaling with patched patch_N (N_HC/N_CTX fix).
Skips already-completed N=1000 rows via is_done check.
Appends to mod1_scaling_results.csv; overwrites mod1_retry.log.
"""
import os, sys, subprocess

BASE = r'C:\Users\Admin\brain-organoid-rl'
LOG  = os.path.join(BASE, 'mod_results', 'mod1_retry.log')

print("Starting MOD-1 retry (N_HC/N_CTX fix applied)...", flush=True)
with open(LOG, 'w', encoding='utf-8') as lf:
    rc = subprocess.call(
        [sys.executable, os.path.join(BASE, 'mod_results', 'mod1_scaling.py')],
        cwd=BASE, stdout=lf, stderr=subprocess.STDOUT)
status = 'PASS' if rc == 0 else f'FAIL(rc={rc})'
print(f"[MOD-1-retry] {status}", flush=True)
