"""
MB NECESSITY/SUFFICIENCY ABLATION — 15 runs (~2 hr)
====================================================

The decisive experiment. Determines whether MB (the core-boost applied per replay
event in the analysis wrapper) is the actual source of schema formation, or merely
amplifies an already-existing replay/coherence/plasticity-driven schema mechanism.

Conditions (5):
  FULL          boost_scale=1.3  ablation={}                                    -- baseline
  FULL_NO_MB    boost_scale=1.0  ablation={}                                    -- MB necessity
  MB_ONLY       boost_scale=1.3  ablation=ALL_OFF                               -- MB sufficiency
  FULL_NO_M2    boost_scale=1.3  ablation={'cross_ltd':False}                   -- isolate M2's role
  FULL_NO_MB_NO_M2 boost_scale=1.0 ablation={'cross_ltd':False}                 -- joint control

Seeds: 42, 1042, 2042
Total: 5 x 3 = 15 runs at default COH_THR=0.50

Interpretation:
  FULL_NO_MB collapses (RS->0)        => MB necessary
  MB_ONLY produces RS                 => MB sufficient
  Both                                 => MB is THE mechanism; rewrite theory
  FULL_NO_MB preserves RS             => replay/coherence/plasticity build schema
"""
import os, sys, time, json, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\mb_ablation'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042]

# All M1-M10 keys (from ablation_pipeline.MECHANISMS).  MB is wrapper-level
# (boost_scale=1.0), so it does not appear here.
ALL_OFF = {
    'overlap_penalty':   False,   # M1
    'cross_ltd':         False,   # M2
    'overlap_priority':  False,   # M3
    'pers_competition':  False,   # M4
    'drift':             False,   # M5
    'fatigue':           False,   # M6
    'hetero_tag':        False,   # M7
    'decorrelation':     False,   # M8
    'wta':               False,   # M9
    'reconsol':          False,   # M10
}

CONDITIONS = [
    # (label,             boost_scale, ablation_dict)
    ('FULL',              1.3, {}),
    ('FULL_NO_MB',        1.0, {}),
    ('MB_ONLY',           1.3, ALL_OFF),
    ('FULL_NO_M2',        1.3, {'cross_ltd': False}),
    ('FULL_NO_MB_NO_M2',  1.0, {'cross_ltd': False}),
]


def chk(label, seed):
    return os.path.join(OUT_DIR, f'{label}_seed{seed}.pkl')


def run_one(label, boost_scale, ablation_dict, seed):
    p = chk(label, seed)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    log = p.replace('.pkl', '.log')
    cmd = [sys.executable, 'ablation_single_seed.py',
           'FULL', '0', str(seed), json.dumps(ablation_dict),
           '--prefix',      f'MBA_{label}_s{seed}',
           '--boost_scale', str(boost_scale)]
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    t0  = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = int(time.time() - t0)

    worker_chk = os.path.join(r'C:\Users\Admin\brain-organoid-rl\ablation_results',
                              f'MBA_{label}_s{seed}_FULL_seed0.pkl')
    if proc.returncode == 0 and os.path.exists(worker_chk):
        with open(worker_chk, 'rb') as f:
            res = pickle.load(f)
        with open(p, 'wb') as f:
            pickle.dump(res, f)
        n = res.get('natural', {})
        print(f'  {label:<18s} seed={seed}  {elapsed}s  '
              f'RS={n.get("real_schema",0):.4f}  '
              f'DAI={n.get("dai_core",0):.4f}  '
              f'Ret={n.get("retention_mean",0):.4f}', flush=True)
        return res
    print(f'  {label:<18s} seed={seed}  FAILED (exit={proc.returncode})', flush=True)
    return None


def summarise(data):
    print(f'\n{"="*80}', flush=True)
    print('MB ABLATION RESULTS', flush=True)
    print(f'{"="*80}', flush=True)

    for metric, mlabel in [('real_schema', 'REAL_SCHEMA'),
                           ('dai_core',    'DAI_core'),
                           ('retention_mean', 'Retention')]:
        print(f'\n{mlabel}:', flush=True)
        print(f'  {"Condition":<18s}  {"mean":>8s}  {"sem":>8s}  values', flush=True)
        groups = {}
        for label, _, _ in CONDITIONS:
            vals = [data[(label, s)].get('natural', {}).get(metric, 0)
                    for s in SEEDS
                    if (label, s) in data and data[(label, s)]
                    and 'natural' in data[(label, s)]]
            if vals:
                m  = np.mean(vals)
                se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                groups[label] = vals
                vstr = '  '.join(f'{v:.4f}' for v in vals)
                print(f'  {label:<18s}  {m:>8.4f}  {se:>8.4f}  [{vstr}]', flush=True)

        # Key pairwise contrasts
        if 'FULL' in groups and 'FULL_NO_MB' in groups:
            t, p = ttest_ind(groups['FULL'], groups['FULL_NO_MB'], equal_var=False)
            delta = np.mean(groups['FULL']) - np.mean(groups['FULL_NO_MB'])
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'  FULL vs FULL_NO_MB      delta={delta:+.4f}  p={p:.4f}  {sig}'
                  '  [MB necessity]', flush=True)
        if 'MB_ONLY' in groups and 'FULL_NO_MB' in groups:
            t, p = ttest_ind(groups['MB_ONLY'], groups['FULL_NO_MB'], equal_var=False)
            delta = np.mean(groups['MB_ONLY']) - np.mean(groups['FULL_NO_MB'])
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'  MB_ONLY vs FULL_NO_MB   delta={delta:+.4f}  p={p:.4f}  {sig}'
                  '  [MB sufficiency]', flush=True)
        if 'FULL' in groups and 'MB_ONLY' in groups:
            t, p = ttest_ind(groups['FULL'], groups['MB_ONLY'], equal_var=False)
            delta = np.mean(groups['FULL']) - np.mean(groups['MB_ONLY'])
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'  FULL vs MB_ONLY         delta={delta:+.4f}  p={p:.4f}  {sig}'
                  '  [M1-M10 contribution]', flush=True)
        if 'FULL' in groups and 'FULL_NO_M2' in groups:
            t, p = ttest_ind(groups['FULL'], groups['FULL_NO_M2'], equal_var=False)
            delta = np.mean(groups['FULL']) - np.mean(groups['FULL_NO_M2'])
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'  FULL vs FULL_NO_M2      delta={delta:+.4f}  p={p:.4f}  {sig}'
                  '  [M2 specific effect at THR=0.50]', flush=True)


def verdict(data):
    """Auto-print interpretation."""
    def mean(label, metric):
        vals = [data[(label, s)].get('natural', {}).get(metric, 0)
                for s in SEEDS
                if (label, s) in data and data[(label, s)]
                and 'natural' in data[(label, s)]]
        return np.mean(vals) if vals else float('nan')

    full_rs       = mean('FULL',       'real_schema')
    full_no_mb_rs = mean('FULL_NO_MB', 'real_schema')
    mb_only_rs    = mean('MB_ONLY',    'real_schema')

    print(f'\n{"="*80}', flush=True)
    print('VERDICT', flush=True)
    print(f'{"="*80}', flush=True)
    print(f'  FULL RS         = {full_rs:.4f}', flush=True)
    print(f'  FULL_NO_MB RS   = {full_no_mb_rs:.4f}', flush=True)
    print(f'  MB_ONLY RS      = {mb_only_rs:.4f}', flush=True)
    print(flush=True)

    full_collapsed = abs(full_no_mb_rs) < 0.05
    mb_sufficient  = mb_only_rs > 0.10 and mb_only_rs > 0.5 * full_rs
    full_no_mb_high = full_no_mb_rs > 0.10 and full_no_mb_rs > 0.5 * full_rs

    if full_collapsed and mb_sufficient:
        print('  RESULT: MB IS THE MECHANISM', flush=True)
        print('  FULL_NO_MB collapses AND MB_ONLY produces schema.', flush=True)
        print('  Schema formation is driven entirely by the core-boost step.', flush=True)
        print('  Coherence-gated replay does not build schema; it only protects it.', flush=True)
        print('  => Theory must be rewritten around MB as the formation mechanism.', flush=True)
    elif full_no_mb_high:
        print('  RESULT: REPLAY/COHERENCE/PLASTICITY BUILDS SCHEMA INDEPENDENTLY', flush=True)
        print('  Schema persists without MB; MB merely amplifies an existing mechanism.', flush=True)
        print('  => Coherence-gated replay abstraction is the mechanism. Paper claim survives.', flush=True)
    elif full_collapsed and not mb_sufficient:
        print('  RESULT: MB NECESSARY BUT NOT SUFFICIENT', flush=True)
        print('  Without MB, schema collapses. But MB alone does not produce it either.', flush=True)
        print('  => MB and M1-M10 work together; neither builds schema alone.', flush=True)
    elif (not full_collapsed) and mb_sufficient:
        print('  RESULT: MB SUFFICIENT BUT NOT NECESSARY', flush=True)
        print('  Schema forms in both FULL_NO_MB and MB_ONLY.', flush=True)
        print('  => There are TWO paths to schema; both replay and MB are sufficient.', flush=True)
    else:
        print('  RESULT: INCONCLUSIVE — review numbers manually', flush=True)


if __name__ == '__main__':
    print('MB NECESSITY/SUFFICIENCY ABLATION', flush=True)
    print(f'Conditions: {[c[0] for c in CONDITIONS]}', flush=True)
    print(f'Seeds:      {SEEDS}', flush=True)
    print(f'Total runs: {len(CONDITIONS) * len(SEEDS)} (~{len(CONDITIONS)*len(SEEDS)*8} min)', flush=True)
    print(flush=True)

    t0 = time.time()
    data = {}
    for label, boost_scale, ablation_dict in CONDITIONS:
        print(f'\n--- {label}  boost={boost_scale}  abl={ablation_dict} ---', flush=True)
        for seed in SEEDS:
            res = run_one(label, boost_scale, ablation_dict, seed)
            if res:
                data[(label, seed)] = res

    summarise(data)
    verdict(data)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min', flush=True)
