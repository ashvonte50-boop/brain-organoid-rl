"""
PILOT ABLATION — Phase 1 screening
===================================
Runs 6 conditions × 3 seeds to identify high-impact mechanisms cheaply.

Conditions tested:
  FULL  — baseline
  -M1   — Overlap-sensitive coherence
  -M2   — Cross-assembly LTD
  -M5   — Directional drift
  -M7   — Heterosynaptic LTD tag
  -M10  — Reconsolidation window

After this run, inspect the summary table and pick the top 3–5 mechanisms
that show the largest ΔDAI_core and/or ΔREAL_SCHEMA.  Those are the ones
worth the expensive 10-seed full ablation.

Usage:
  python ablation_pilot.py
  python ablation_pilot.py --seeds 5     # optional: bump to 5 for more confidence
"""
import os, sys, time, argparse, pickle
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

from ablation_pipeline import (
    run_one, aggregate, cohen_d, print_summary,
    _save_condition, _save_all, _write_csv,
    BASE_SEED, MECHANISMS, OUT_DIR,
)
from ablation_figures import generate_all_figures

import numpy as np
from scipy.stats import ttest_ind

# ── Pilot conditions (high-prior mechanisms + full model baseline) ─────────────
PILOT_CONDITIONS = {
    'FULL':      {},                           # all mechanisms on
    'ABLATE_M1': {'overlap_penalty':  False},
    'ABLATE_M2': {'cross_ltd':        False},
    'ABLATE_M5': {'drift':            False},
    'ABLATE_M7': {'hetero_tag':       False},
    'ABLATE_M10':{'reconsol':         False},
}

PILOT_LABELS = {
    'FULL':       'Full model',
    'ABLATE_M1':  '-M1: Overlap Coherence',
    'ABLATE_M2':  '-M2: Cross-Assembly LTD',
    'ABLATE_M5':  '-M5: Directional Drift',
    'ABLATE_M7':  '-M7: Heterosynaptic Tag',
    'ABLATE_M10': '-M10: Reconsolidation Window',
}


def run_pilot(n_seeds=3, resume=False):
    print('=' * 60, flush=True)
    print('PHASE 1 PILOT ABLATION', flush=True)
    print(f'Conditions: {list(PILOT_CONDITIONS)}, n_seeds={n_seeds}', flush=True)
    if resume:
        print('RESUME MODE: skipping already-saved conditions.', flush=True)
    print('=' * 60, flush=True)

    conditions = {}

    for cname, abl_dict in PILOT_CONDITIONS.items():
        # Resume: reload saved PKL and skip re-running
        saved_path = os.path.join(OUT_DIR, f'PILOT_{cname}.pkl')
        if resume and os.path.exists(saved_path):
            with open(saved_path, 'rb') as f:
                seed_results = pickle.load(f)
            print(f'\n--- {PILOT_LABELS[cname]} --- [LOADED from {saved_path}]', flush=True)
            conditions[cname] = seed_results
            continue

        print(f'\n--- {PILOT_LABELS[cname]} ---', flush=True)
        seed_results = []
        for si in range(n_seeds):
            seed = BASE_SEED + si * 1000
            print(f'  Seed {si + 1}/{n_seeds} (seed={seed})', flush=True)
            t0 = time.time()
            res = run_one(seed, abl_dict, boost_scale=1.3, label=cname)
            print(f'  Done ({time.time() - t0:.0f}s)', flush=True)
            seed_results.append(res)
        conditions[cname] = seed_results
        _save_condition(f'PILOT_{cname}', seed_results)

    _save_all('pilot_ablations', conditions)
    return conditions


def pilot_summary(conditions, mode='natural'):
    """Print ranked summary and return importance dict."""
    full_agg  = aggregate(conditions.get('FULL', []), mode)
    full_dai  = full_agg.get('dai_core_vals', [])
    full_rs   = full_agg.get('real_schema_vals', [])
    full_n    = len(full_dai)

    print(f'\n{"=" * 70}', flush=True)
    print(f'PILOT SUMMARY  mode={mode}  n_seeds={full_n}', flush=True)
    print(f'{"=" * 70}', flush=True)
    hdr = (f'  {"Condition":<25}  {"DAI_core":>10}  {"ΔDAI":>8}  '
           f'{"RS":>8}  {"ΔRS":>8}  {"Ret":>7}  {"d(DAI)":>7}  Sig')
    print(hdr, flush=True)
    print('  ' + '-' * 68, flush=True)

    # Full model row
    dai_m = full_agg.get('dai_core_mean', 0)
    rs_m  = full_agg.get('real_schema_mean', 0)
    ret_m = full_agg.get('retention_mean_mean', 0)
    print(f'  {"FULL":<25}  {dai_m:+10.4f}  {"—":>8}  {rs_m:8.4f}  '
          f'{"—":>8}  {ret_m:7.4f}  {"—":>7}', flush=True)

    importance = {}
    for cname in list(PILOT_CONDITIONS)[1:]:   # skip FULL
        sl   = conditions.get(cname, [])
        if not sl: continue
        agg  = aggregate(sl, mode)
        m    = agg.get('dai_core_mean', 0)
        rs   = agg.get('real_schema_mean', 0)
        ret  = agg.get('retention_mean_mean', 0)
        vals = agg.get('dai_core_vals', [])
        d_dai = float(m - dai_m)
        d_rs  = float(rs - rs_m)
        cd    = cohen_d(full_dai, vals) if (full_dai and vals) else 0.0
        sig   = ''
        if full_dai and vals:
            _, p = ttest_ind(full_dai, vals, equal_var=False)
            sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        else:
            p = 1.0

        importance[cname] = {
            'delta_dai': d_dai, 'delta_rs': d_rs, 'delta_ret': float(ret - ret_m),
            'cohens_d': cd, 'p': p, 'sig': sig,
            'full_dai': dai_m, 'abl_dai': m,
        }
        label = PILOT_LABELS.get(cname, cname)
        print(f'  {label:<25}  {m:+10.4f}  {d_dai:+8.4f}  {rs:8.4f}  '
              f'{d_rs:+8.4f}  {ret:7.4f}  {cd:+7.3f}  {sig}', flush=True)

    # Rank by |ΔDAI|
    ranked = sorted(importance.items(), key=lambda x: abs(x[1]['delta_dai']), reverse=True)
    print(f'\n{"=" * 70}', flush=True)
    print('MECHANISM RANKING (by |ΔDAI_core|):', flush=True)
    for i, (cname, imp) in enumerate(ranked, 1):
        label = PILOT_LABELS.get(cname, cname)
        impact = ('HIGH' if abs(imp['delta_dai']) > 0.05
                  else 'MEDIUM' if abs(imp['delta_dai']) > 0.01
                  else 'LOW')
        print(f'  #{i}  {label:<28}  ΔDAI={imp["delta_dai"]:+.4f}  '
              f'd={imp["cohens_d"]:+.3f}  {imp["sig"]:5s}  [{impact}]', flush=True)

    # Recommendation
    high  = [c for c, v in ranked if abs(v['delta_dai']) > 0.05]
    med   = [c for c, v in ranked if 0.01 <= abs(v['delta_dai']) <= 0.05]
    low   = [c for c, v in ranked if abs(v['delta_dai']) < 0.01]
    print(f'\nRECOMMENDATION:', flush=True)
    print(f'  HIGH impact (run 10-seed): {high}', flush=True)
    print(f'  MEDIUM impact (consider):  {med}', flush=True)
    print(f'  LOW impact (can skip):     {low}', flush=True)
    print(f'\nNext step: run ablation_pipeline.py --part single --seeds 10', flush=True)
    print(f'  targeting conditions above with HIGH + MEDIUM impact.\n', flush=True)

    # Save importance CSV
    rows = [{'condition': c, **v} for c, v in importance.items()]
    _write_csv('pilot_importance.csv', rows)

    return importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',  type=int, default=3)
    parser.add_argument('--mode',   default='natural', choices=['natural', 'hyper'])
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-saved conditions and continue from where interrupted')
    args = parser.parse_args()

    t0 = time.time()
    conds = run_pilot(n_seeds=args.seeds, resume=args.resume)
    imp   = pilot_summary(conds, mode=args.mode)

    print('Generating pilot figures...', flush=True)
    # Temporarily swap pilot data into the expected filename so figures work
    import pickle, os
    pilot_path  = os.path.join(OUT_DIR, 'pilot_ablations.pkl')
    target_path = os.path.join(OUT_DIR, 'single_ablations.pkl')
    # Load pilot, wrap in full MECH_ORDER keys (missing ones just absent)
    with open(pilot_path, 'rb') as f:
        pilot_data = pickle.load(f)
    with open(target_path, 'wb') as f:
        pickle.dump(pilot_data, f)

    generate_all_figures(mode=args.mode, fig_list=[1, 2, 3, 4, 9, 10])

    elapsed = time.time() - t0
    print(f'\nPilot complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)', flush=True)
    print(f'Results: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
