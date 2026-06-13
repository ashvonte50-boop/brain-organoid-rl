"""
MINIMAL REPLICATION — 9 runs
COH_THR: [0.00, 0.08, 0.50] x seeds: [42, 1042, 2042]
"""
import os, sys, time, json, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import ttest_ind

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\replication'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

COH_THRS = [0.00, 0.08, 0.50]
SEEDS    = [42, 1042, 2042]


def chk(coh, seed):
    return os.path.join(OUT_DIR, f'coh{coh:.2f}_seed{seed}.pkl')


def run_one(coh, seed):
    p = chk(coh, seed)
    if os.path.exists(p):
        with open(p,'rb') as f: return pickle.load(f)
    log = p.replace('.pkl','.log')
    cmd = [sys.executable, 'ablation_single_seed.py',
           'FULL', '0', str(seed), '{}',
           '--prefix', f'REP_coh{coh:.2f}_s{seed}',
           '--coh_thr', str(coh)]
    env = {**os.environ, 'DEV_MODE':'1', 'PYTHONIOENCODING':'utf-8'}
    t0  = time.time()
    with open(log,'w') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = int(time.time()-t0)
    worker_chk = os.path.join(r'C:\Users\Admin\brain-organoid-rl\ablation_results',
                              f'REP_coh{coh:.2f}_s{seed}_FULL_seed0.pkl')
    if proc.returncode == 0 and os.path.exists(worker_chk):
        with open(worker_chk,'rb') as f: res = pickle.load(f)
        with open(p,'wb') as f: pickle.dump(res, f)
        n = res.get('natural',{})
        print(f'  coh={coh:.2f} seed={seed}  {elapsed}s  '
              f'DAI={n.get("dai_core",0):.4f}  '
              f'RS={n.get("real_schema",0):.4f}  '
              f'Ret={n.get("retention_mean",0):.4f}', flush=True)
        return res
    print(f'  coh={coh:.2f} seed={seed}  FAILED', flush=True)
    return None


def summarise(data):
    print(f'\n{"="*65}', flush=True)
    print('REPLICATION SUMMARY', flush=True)
    print(f'{"="*65}', flush=True)

    for metric, label in [('dai_core','DAI_core'),('real_schema','REAL_SCHEMA'),('retention_mean','Retention')]:
        print(f'\n{label}:', flush=True)
        print(f'  {"COH_THR":>8}  {"mean":>8}  {"sem":>8}  {"values"}', flush=True)
        groups = {}
        for coh in COH_THRS:
            vals = [data[(coh,s)].get('natural',{}).get(metric,0)
                    for s in SEEDS if (coh,s) in data and data[(coh,s)]
                    and 'natural' in data[(coh,s)]]
            if vals:
                m = np.mean(vals); se = np.std(vals,ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0
                groups[coh] = vals
                print(f'  {coh:>8.2f}  {m:>8.4f}  {se:>8.4f}  {[round(v,4) for v in vals]}', flush=True)

        # t-tests
        if 0.0 in groups and 0.5 in groups and len(groups[0.0])>1:
            t,p = ttest_ind(groups[0.0], groups[0.5], equal_var=False)
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            delta = np.mean(groups[0.0]) - np.mean(groups[0.5])
            print(f'  coh=0.00 vs 0.50: delta={delta:+.4f}  p={p:.4f}  {sig}', flush=True)

    # Verdict
    dai_low = [data[(0.0,s)].get('natural',{}).get('dai_core',0)
               for s in SEEDS if (0.0,s) in data and data[(0.0,s)]]
    dai_high= [data[(0.5,s)].get('natural',{}).get('dai_core',0)
               for s in SEEDS if (0.5,s) in data and data[(0.5,s)]]
    ret_low = [data[(0.0,s)].get('natural',{}).get('retention_mean',0)
               for s in SEEDS if (0.0,s) in data and data[(0.0,s)]]

    print(f'\n{"="*65}', flush=True)
    if dai_low and dai_high:
        delta = np.mean(dai_low) - np.mean(dai_high)
        ret_gone = all(r < 0.01 for r in ret_low)
        if abs(delta) > 0.02 and ret_gone:
            print('REPLICATION: CONFIRMED', flush=True)
            print(f'Pattern holds across 3 seeds.', flush=True)
            print(f'ΔDAI(0.00 vs 0.50) = {delta:+.4f}', flush=True)
            print(f'Retention collapses to ~0 when mechanisms active.', flush=True)
            print(f'=> START WRITING THE PAPER.', flush=True)
        else:
            print('REPLICATION: WEAK/FAILED', flush=True)
            print(f'ΔDAI = {delta:+.4f} (need >0.02)', flush=True)
            print(f'Retention gone: {ret_gone}', flush=True)
            print(f'=> Do NOT proceed. Investigate further.', flush=True)


if __name__ == '__main__':
    print('MINIMAL REPLICATION — 9 runs (~1.2 hr)', flush=True)
    print(f'COH_THRS={COH_THRS}  SEEDS={SEEDS}', flush=True)
    t0 = time.time()
    data = {}
    for coh in COH_THRS:
        print(f'\n--- COH_THR={coh} ---', flush=True)
        for seed in SEEDS:
            res = run_one(coh, seed)
            if res: data[(coh,seed)] = res
    summarise(data)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min', flush=True)
