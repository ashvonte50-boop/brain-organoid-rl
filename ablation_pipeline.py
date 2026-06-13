"""
PUBLICATION-GRADE ABLATION STUDY
Replay Distortion / Directional Schema Abstraction
===================================================

Mechanisms M1-M10 (ablation keys in compare_catastrophic_forgetting.py):
  M1  overlap_penalty    — Overlap-sensitive coherence gating
  M2  cross_ltd          — Cross-assembly LTD during replay
  M3  overlap_priority   — Overlap-weighted urgency prioritisation
  M4  pers_competition   — Competitive persistence budget
  M5  drift              — Representational drift (directional)
  M6  fatigue            — Shared-neuron fatigue
  M7  hetero_tag         — Heterosynaptic LTD tag
  M8  decorrelation      — Training-time decorrelation
  M9  wta                — Coherence winner-take-all
  M10 reconsol           — Reconsolidation window metaplasticity
  MB  core_boost         — 1.3x post-replay core-to-core boost (wrapper-level)

Usage:
  python ablation_pipeline.py --part single         # Part A: single ablations
  python ablation_pipeline.py --part cumulative     # Part C: cumulative ablations
  python ablation_pipeline.py --part interactions   # Part D: interaction tests
  python ablation_pipeline.py --part importance     # Part B: importance analysis
  python ablation_pipeline.py --part all            # Run A → B → C → D
  python ablation_pipeline.py --seeds N             # Override seed count (default 10)
"""
import os, sys, time, pickle, json, csv, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_experiments import (
    make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE,
)
from _distortion_paper import (
    compute_directional_alignment, compute_real_schema_index,
    _extract_centroids,
)
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

from scipy.stats import ttest_ind, ttest_1samp

OUT_DIR   = r'C:\Users\Admin\brain-organoid-rl\ablation_results'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Seed protocol (matches 10-seed replication) ────────────────────────────────
N_SEEDS   = 10
BASE_SEED = 42
CORE_SIZE = SCHEMA_CORE_SIZE   # 20

# ── Mechanism definitions ──────────────────────────────────────────────────────
MECHANISMS = {
    'M1':  {'label': 'M1: Overlap\nCoherence',      'short': 'M1',  'key': 'overlap_penalty',  'full_val': True, 'abl_val': False},
    'M2':  {'label': 'M2: Cross-\nAssembly LTD',    'short': 'M2',  'key': 'cross_ltd',         'full_val': True, 'abl_val': False},
    'M3':  {'label': 'M3: Overlap\nPriority',        'short': 'M3',  'key': 'overlap_priority',  'full_val': True, 'abl_val': False},
    'M4':  {'label': 'M4: Pers.\nCompetition',       'short': 'M4',  'key': 'pers_competition',  'full_val': True, 'abl_val': False},
    'M5':  {'label': 'M5: Repr.\nDrift',             'short': 'M5',  'key': 'drift',             'full_val': True, 'abl_val': False},
    'M6':  {'label': 'M6: Shared-\nneuron Fatigue',  'short': 'M6',  'key': 'fatigue',           'full_val': True, 'abl_val': False},
    'M7':  {'label': 'M7: Hetero-\nsynaptic Tag',    'short': 'M7',  'key': 'hetero_tag',        'full_val': True, 'abl_val': False},
    'M8':  {'label': 'M8: Training\nDecorr.',        'short': 'M8',  'key': 'decorrelation',     'full_val': True, 'abl_val': False},
    'M9':  {'label': 'M9: Coherence\nWTA',           'short': 'M9',  'key': 'wta',               'full_val': True, 'abl_val': False},
    'M10': {'label': 'M10: Reconsol.\nWindow',       'short': 'M10', 'key': 'reconsol',          'full_val': True, 'abl_val': False},
    'MB':  {'label': 'MB: Core\nBoost',              'short': 'MB',  'key': None,                'full_val': True, 'abl_val': False},
}

# Order in which mechanisms are added during cumulative analysis
# (from most fundamental to most refined)
CUMULATIVE_ORDER = ['MB', 'M5', 'M7', 'M2', 'M6', 'M1', 'M9', 'M3', 'M4', 'M8', 'M10']

INTERACTION_PAIRS = [
    ('M2',  'M7',  'M2+M7'),
    ('M2',  'M10', 'M2+M10'),
    ('M5',  'M8',  'M5+M8'),
    ('M6',  'M7',  'M6+M7'),
    ('M7',  'M10', 'M7+M10'),
]

# ── Per-seed centroid log & net capture ───────────────────────────────────────
_CENTROID_LOG = []
_last_net      = [None]   # mutable container so the closure can write back


def _extract_cents(net, assemblies):
    """Extract centroid dict {mem_idx: np.array} from current weights."""
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne].cpu().numpy()
        cents = {}
        for i, asm in enumerate(assemblies):
            valid = [int(x) for x in asm if 0 <= int(x) < ne]
            if valid:
                cents[i] = w[np.ix_(valid, valid)].mean(axis=1)
        return cents


def _make_wrapper(assemblies, ablation_dict, boost_scale=1.3):
    """Wrap _replay_one_event to:
      - capture the trained net reference into _last_net[0]
      - record centroid movement in _CENTROID_LOG
      - apply optional MB core-to-core boost
    """
    orig = ccf._replay_one_event
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

    def _wrapper(net, assembly, tags=None, **kw):
        _last_net[0] = net          # FIX: capture net so RS can be computed after run
        cb = _extract_cents(net, assemblies)
        result = orig(net, assembly, tags=tags, **p, **kw)
        with torch.no_grad():
            ne = net.n_exc
            w = net.W.data[:ne, :ne]
            if boost_scale != 1.0:
                ci = np.array([int(x) for x in range(CORE_SIZE) if x < ne])
                if len(ci):
                    ci_t = torch.as_tensor(ci, device=w.device)
                    w[ci_t[:, None], ci_t[None, :]] *= boost_scale
                    w.clamp_(0.0, net.w_max)
            ca = _extract_cents(net, assemblies)
            _CENTROID_LOG.append({
                'replay_id':       kw.get('burst_id', 0) * 1000 + kw.get('event_id', 0),
                'memory_idx':      kw.get('assembly_idx', -1),
                'centroid_before': {k: v.tolist() for k, v in cb.items()},
                'centroid_after':  {k: v.tolist() for k, v in ca.items()},
            })
        return result
    return orig, _wrapper


# ── Single experiment runner ───────────────────────────────────────────────────

def run_one(seed, ablation_dict, boost_scale=1.3, label='full'):
    """Run natural + hyper replay for one seed with given ablation dict.

    Returns dict keyed by mode ('natural', 'hyper') with all metrics.
    """
    ccf.torch.manual_seed(seed)
    ccf.np.random.seed(seed)
    assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

    results = {}
    # Run natural mode only. Hyper (same experiment + post-hoc noise) causes
    # global-state contamination when run_sequential_experiment is called twice
    # in the same process, leading to hangs. Re-enable after root-cause is fixed.
    for mode in ('natural',):
        _CENTROID_LOG.clear()
        _last_net[0] = None                         # reset capture before each run
        orig, wrapper = _make_wrapper(assemblies, ablation_dict, boost_scale=boost_scale)
        ccf._replay_one_event = wrapper
        try:
            r = ccf.run_sequential_experiment(
                True, True, assemblies, seed, ablation=ablation_dict
            )
        except Exception as e:
            ccf._replay_one_event = orig
            print(f'    {mode} CRASH: {e}', flush=True)
            continue
        ccf._replay_one_event = orig

        # ── Retrieve captured net (FIX: run_sequential_experiment does NOT
        #    return "net" in its result dict; we capture it via the wrapper) ──
        net = _last_net[0]
        assert net is not None, (
            f'FATAL: net was not captured during {mode} run (seed={seed}). '
            'This means no replay events fired — REAL_SCHEMA will be wrong. '
            'Check that use_replay=True and that replay events are triggered.'
        )

        # Compute REAL_SCHEMA on the trained network BEFORE any post-hoc noise
        rs = compute_real_schema_index(net, assemblies, core_mask)
        if rs == 0.0:
            print(f'    WARN: REAL_SCHEMA=0 with valid net (seed={seed}, mode={mode}) '
                  '— check weight block values.', flush=True)

        # Hyper mode: add post-hoc weight noise (after RS is recorded)
        if mode == 'hyper':
            try:
                with torch.no_grad():
                    ne = net.n_exc
                    w = net.W.data[:ne, :ne]
                    w.add_(torch.randn_like(w) * 0.008)
                    w.clamp_(0.0, net.w_max)
            except Exception:
                pass

        fs = np.nan_to_num(r['final_scores'], nan=0.0)
        bs = r.get('baseline_scores', np.zeros(4))
        log_snap = list(_CENTROID_LOG)

        dall = compute_directional_alignment(log_snap, n_mem=4, core_size=CORE_SIZE)

        deltas = []
        for e in log_snap:
            cb = e.get('centroid_before', {})
            ca = e.get('centroid_after', {})
            mid = int(e.get('memory_idx', -1))
            if mid >= 0 and mid in cb and mid in ca:
                deltas.append(np.linalg.norm(np.array(ca[mid]) - np.array(cb[mid])))
        di = float(np.mean(deltas)) if deltas else 0.0

        results[mode] = {
            'final_scores':    fs.tolist(),
            'baseline_scores': (bs.tolist() if hasattr(bs, 'tolist') else list(bs)),
            'retention_A':     float(fs[0]),
            'retention_B':     float(fs[1]),
            'retention_C':     float(fs[2]),
            'retention_D':     float(fs[3]),
            'retention_mean':  float(np.mean(fs)),
            'dai_core':        float(dall['mean_core']),
            'dai_unique':      float(dall['mean_unique']),
            'real_schema':     float(rs),
            'distortion':      float(di),
            'n_events':        int(dall['n_events']),
        }
        print(
            f'    {mode:8s}  ret_A={results[mode]["retention_A"]:.4f}  '
            f'RS={results[mode]["real_schema"]:.4f}  '
            f'DAI={results[mode]["dai_core"]:.4f}  '
            f'DI={results[mode]["distortion"]:.4f}',
            flush=True,
        )

    return results


# ── Aggregate helpers ──────────────────────────────────────────────────────────

METRICS = ['retention_A', 'retention_B', 'retention_C', 'retention_D',
           'retention_mean', 'dai_core', 'dai_unique', 'real_schema', 'distortion']


def aggregate(all_seed_results, mode='natural'):
    """Aggregate per-seed results for one mode. Returns mean/sem/vals per metric."""
    out = {}
    for k in METRICS:
        vals = [s[mode][k] for s in all_seed_results if mode in s and k in s.get(mode, {})]
        if vals:
            out[k + '_mean'] = float(np.mean(vals))
            out[k + '_sem']  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            out[k + '_vals'] = vals
    return out


def cohen_d(a, b):
    """Cohen's d effect size between two independent samples."""
    a, b = np.array(a), np.array(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s_pooled = np.sqrt(((n1 - 1) * a.std(ddof=1)**2 + (n2 - 1) * b.std(ddof=1)**2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / (s_pooled + 1e-9))


# ── Part A: Single ablations ───────────────────────────────────────────────────

def run_single_ablations(n_seeds=N_SEEDS):
    print('=' * 65, flush=True)
    print('PART A: SINGLE-MECHANISM ABLATIONS', flush=True)
    print(f'n_seeds={n_seeds}, mechanisms=11 (M1-M10 + MB)', flush=True)
    print('=' * 65, flush=True)

    conditions = {}

    print('\n--- FULL MODEL ---', flush=True)
    full_results = []
    for si in range(n_seeds):
        seed = BASE_SEED + si * 1000
        print(f'  Seed {si + 1}/{n_seeds} (seed={seed})', flush=True)
        t0 = time.time()
        res = run_one(seed, ablation_dict={}, boost_scale=1.3, label='full')
        print(f'  Done ({time.time() - t0:.0f}s)', flush=True)
        full_results.append(res)
    conditions['FULL'] = full_results
    _save_condition('FULL', full_results)

    for mid, mdef in MECHANISMS.items():
        print(f'\n--- ABLATE {mid}: {mdef["label"].replace(chr(10), " ")} ---', flush=True)
        abl_results = []

        if mdef['key'] is None:
            abl_dict    = {}
            boost_scale = 1.0
        else:
            abl_dict    = {mdef['key']: False}
            boost_scale = 1.3

        for si in range(n_seeds):
            seed = BASE_SEED + si * 1000
            print(f'  Seed {si + 1}/{n_seeds} (seed={seed})', flush=True)
            t0 = time.time()
            res = run_one(seed, abl_dict, boost_scale=boost_scale, label=mid)
            print(f'  Done ({time.time() - t0:.0f}s)', flush=True)
            abl_results.append(res)

        conditions[f'ABLATE_{mid}'] = abl_results
        _save_condition(f'ABLATE_{mid}', abl_results)

    _save_all('single_ablations', conditions)
    print('\nPart A complete.', flush=True)
    return conditions


# ── Part B: Mechanism importance analysis ─────────────────────────────────────

def run_importance_analysis(single_conditions=None, mode='natural'):
    """Compute ΔDAI, ΔREAL_SCHEMA, ΔDistortion, ΔRetention and rank mechanisms."""
    print('=' * 65, flush=True)
    print('PART B: MECHANISM IMPORTANCE ANALYSIS', flush=True)
    print('=' * 65, flush=True)

    if single_conditions is None:
        p = os.path.join(OUT_DIR, 'single_ablations.pkl')
        if not os.path.exists(p):
            print('  ERROR: single_ablations.pkl not found. Run --part single first.', flush=True)
            return {}
        with open(p, 'rb') as f:
            single_conditions = pickle.load(f)

    full_agg = aggregate(single_conditions.get('FULL', []), mode)
    importance = {}

    for mid, mdef in MECHANISMS.items():
        cname = f'ABLATE_{mid}'
        if cname not in single_conditions:
            continue
        abl_agg = aggregate(single_conditions[cname], mode)

        full_dai = full_agg.get('dai_core_vals', [])
        abl_dai  = abl_agg.get('dai_core_vals', [])
        full_rs  = full_agg.get('real_schema_vals', [])
        abl_rs   = abl_agg.get('real_schema_vals', [])
        full_di  = full_agg.get('distortion_vals', [])
        abl_di   = abl_agg.get('distortion_vals', [])
        full_ret = full_agg.get('retention_mean_vals', [])
        abl_ret  = abl_agg.get('retention_mean_vals', [])

        delta_dai = (full_agg.get('dai_core_mean', 0) - abl_agg.get('dai_core_mean', 0))
        delta_rs  = (full_agg.get('real_schema_mean', 0) - abl_agg.get('real_schema_mean', 0))
        delta_di  = (full_agg.get('distortion_mean', 0) - abl_agg.get('distortion_mean', 0))
        delta_ret = (full_agg.get('retention_mean_mean', 0) - abl_agg.get('retention_mean_mean', 0))

        t_dai, p_dai = (ttest_ind(full_dai, abl_dai) if (len(full_dai) > 1 and len(abl_dai) > 1) else (np.nan, 1.0))
        t_rs,  p_rs  = (ttest_ind(full_rs,  abl_rs)  if (len(full_rs)  > 1 and len(abl_rs)  > 1) else (np.nan, 1.0))

        d_dai = cohen_d(full_dai, abl_dai)
        d_rs  = cohen_d(full_rs,  abl_rs)
        d_ret = cohen_d(full_ret, abl_ret)

        importance[mid] = {
            'label':       mdef['label'].replace('\n', ' '),
            'delta_dai':   float(delta_dai),
            'delta_rs':    float(delta_rs),
            'delta_di':    float(delta_di),
            'delta_ret':   float(delta_ret),
            't_dai':       float(t_dai) if not np.isnan(t_dai) else 0.0,
            'p_dai':       float(p_dai),
            't_rs':        float(t_rs) if not np.isnan(t_rs) else 0.0,
            'p_rs':        float(p_rs),
            'cohens_d_dai': float(d_dai),
            'cohens_d_rs':  float(d_rs),
            'cohens_d_ret': float(d_ret),
            'full_dai_mean': full_agg.get('dai_core_mean', 0),
            'abl_dai_mean':  abl_agg.get('dai_core_mean', 0),
            'full_rs_mean':  full_agg.get('real_schema_mean', 0),
            'abl_rs_mean':   abl_agg.get('real_schema_mean', 0),
            'n_seeds':     len(single_conditions.get(cname, [])),
        }

    # Rank by combined importance (|ΔDAI| + |ΔREAL_SCHEMA|)
    ranked = sorted(importance.items(), key=lambda x: abs(x[1]['delta_dai']) + abs(x[1]['delta_rs']), reverse=True)

    print(f'\n{"Rank":4s}  {"Mech":5s}  {"ΔDAI":>10}  {"ΔRS":>10}  {"ΔDist":>8}  {"ΔRet":>8}  {"Cohen_d":>8}  {"p":>8}', flush=True)
    print('-' * 70, flush=True)
    for rank, (mid, imp) in enumerate(ranked, 1):
        sig = '***' if imp['p_dai'] < 0.001 else '**' if imp['p_dai'] < 0.01 else '*' if imp['p_dai'] < 0.05 else 'n.s.'
        print(
            f'{rank:4d}  {mid:5s}  {imp["delta_dai"]:+10.4f}  {imp["delta_rs"]:+10.4f}  '
            f'{imp["delta_di"]:+8.4f}  {imp["delta_ret"]:+8.4f}  '
            f'{imp["cohens_d_dai"]:+8.3f}  {imp["p_dai"]:8.4f} {sig}',
            flush=True,
        )

    neg = [mid for mid, imp in importance.items()
           if abs(imp['delta_dai']) < 0.01 and abs(imp['delta_rs']) < 0.01 and imp['p_dai'] > 0.1]
    if neg:
        print(f'\nMechanisms with negligible contribution (|ΔDAI|<0.01, p>0.1): {neg}', flush=True)

    # Save
    path_pkl = os.path.join(OUT_DIR, 'importance_analysis.pkl')
    with open(path_pkl, 'wb') as f:
        pickle.dump({'importance': importance, 'ranked': ranked, 'mode': mode}, f)

    rows = []
    for rank, (mid, imp) in enumerate(ranked, 1):
        rows.append({'rank': rank, 'mechanism': mid, **imp})
    _write_csv('importance_analysis.csv', rows)
    print(f'\nSaved importance analysis -> {path_pkl}', flush=True)

    return importance


# ── Part C: Cumulative ablations ───────────────────────────────────────────────

def run_cumulative_ablations(n_seeds=N_SEEDS):
    print('=' * 65, flush=True)
    print('PART C: CUMULATIVE ABLATIONS', flush=True)
    print('=' * 65, flush=True)

    conditions = {}
    ablated_so_far = {}
    boost = 1.3

    for i, mid in enumerate(CUMULATIVE_ORDER):
        mdef = MECHANISMS[mid]
        if mdef['key'] is None:
            boost = 1.0
        else:
            ablated_so_far[mdef['key']] = False

        label    = '+'.join(CUMULATIVE_ORDER[:i + 1])
        abl_dict = dict(ablated_so_far)
        print(f'\n--- CUMULATIVE {i + 1}/{len(CUMULATIVE_ORDER)}: ablated=[{label}] ---', flush=True)

        seed_results = []
        for si in range(n_seeds):
            seed = BASE_SEED + si * 1000
            print(f'  Seed {si + 1}/{n_seeds}', flush=True)
            res = run_one(seed, abl_dict, boost_scale=boost)
            seed_results.append(res)

        ckey = f'CUM_{i + 1}_{mid}'
        conditions[ckey] = seed_results
        _save_condition(ckey, seed_results)

    _save_all('cumulative_ablations', conditions)
    print('\nPart C complete.', flush=True)
    return conditions


# ── Part D: Interaction ablations ─────────────────────────────────────────────

def run_interaction_ablations(n_seeds=N_SEEDS):
    print('=' * 65, flush=True)
    print('PART D: INTERACTION ABLATIONS', flush=True)
    print('=' * 65, flush=True)

    conditions = {}
    for mid_a, mid_b, label in INTERACTION_PAIRS:
        def_a = MECHANISMS[mid_a]
        def_b = MECHANISMS[mid_b]
        abl_dict = {}
        boost = 1.3
        if def_a['key']:
            abl_dict[def_a['key']] = False
        else:
            boost = 1.0
        if def_b['key']:
            abl_dict[def_b['key']] = False
        else:
            boost = 1.0

        print(f'\n--- INTERACTION {label} ---', flush=True)
        seed_results = []
        for si in range(n_seeds):
            seed = BASE_SEED + si * 1000
            print(f'  Seed {si + 1}/{n_seeds}', flush=True)
            res = run_one(seed, abl_dict, boost_scale=boost)
            seed_results.append(res)

        conditions[label] = seed_results
        _save_condition(f'INTERACT_{label}', seed_results)

    _save_all('interaction_ablations', conditions)
    print('\nPart D complete.', flush=True)
    return conditions


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(conditions, mode='natural'):
    full     = conditions.get('FULL', [])
    full_agg = aggregate(full, mode)
    full_dai = full_agg.get('dai_core_vals', [])

    print(f'\n{"=" * 80}', flush=True)
    print(f'SINGLE-ABLATION SUMMARY  mode={mode}  n={len(full)}', flush=True)
    print(f'{"=" * 80}', flush=True)
    header = f'{"Condition":20s}  {"DAI_core":>10}  {"REAL_SCHEMA":>12}  {"Distortion":>11}  {"Ret_A":>7}  Sig'
    print(header, flush=True)
    print('-' * 80, flush=True)

    for cond_name, seed_list in conditions.items():
        agg   = aggregate(seed_list, mode)
        dai_m = agg.get('dai_core_mean', 0)
        rs_m  = agg.get('real_schema_mean', 0)
        di_m  = agg.get('distortion_mean', 0)
        ra_m  = agg.get('retention_A_mean', 0)
        sig   = ''
        abl_dai = agg.get('dai_core_vals', [])
        if full_dai and abl_dai and cond_name != 'FULL':
            _, p = ttest_ind(full_dai, abl_dai)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(
            f'  {cond_name:18s}  {dai_m:+10.4f}  {rs_m:12.4f}  {di_m:11.4f}  {ra_m:7.4f}  {sig}',
            flush=True,
        )


# ── Save helpers ───────────────────────────────────────────────────────────────

def _save_condition(name, seed_results):
    path = os.path.join(OUT_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(seed_results, f)


def _write_csv(filename, rows):
    if not rows:
        return
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'  CSV -> {path}', flush=True)


def _save_all(name, conditions):
    path = os.path.join(OUT_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(conditions, f)

    rows = []
    for cond_name, seed_list in conditions.items():
        for mode in ('natural', 'hyper'):
            agg = aggregate(seed_list, mode)
            row = {'condition': cond_name, 'mode': mode, 'n': len(seed_list)}
            for k, v in agg.items():
                if not k.endswith('_vals'):
                    row[k] = v
            rows.append(row)
    _write_csv(f'{name}.csv', rows)

    # Also write per-seed flat CSV
    seed_rows = []
    for cond_name, seed_list in conditions.items():
        for si, seed_res in enumerate(seed_list):
            seed = BASE_SEED + si * 1000
            for mode in ('natural', 'hyper'):
                if mode not in seed_res:
                    continue
                row = {'condition': cond_name, 'seed_idx': si, 'seed': seed, 'mode': mode}
                row.update({k: v for k, v in seed_res[mode].items() if not isinstance(v, list)})
                seed_rows.append(row)
    _write_csv(f'{name}_per_seed.csv', seed_rows)

    print(f'  PKL  -> {path}', flush=True)


# ── Main entry ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Publication-grade ablation study')
    parser.add_argument('--part', choices=['single', 'cumulative', 'interactions',
                                           'importance', 'all'],
                        default='single')
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help=f'Number of seeds (default {N_SEEDS})')
    parser.add_argument('--mode', default='natural', choices=['natural', 'hyper'],
                        help='Replay mode for summary/importance analysis')
    args = parser.parse_args()

    t_start = time.time()
    ns = args.seeds
    print(f'Ablation pipeline: part={args.part}  n_seeds={ns}', flush=True)

    single_conds = None

    if args.part in ('single', 'all'):
        single_conds = run_single_ablations(n_seeds=ns)
        print_summary(single_conds, mode=args.mode)

    if args.part in ('importance', 'all'):
        run_importance_analysis(single_conditions=single_conds, mode=args.mode)

    if args.part in ('cumulative', 'all'):
        run_cumulative_ablations(n_seeds=ns)

    if args.part in ('interactions', 'all'):
        run_interaction_ablations(n_seeds=ns)

    total = time.time() - t_start
    print(f'\nTotal runtime: {total:.0f}s ({total / 3600:.2f} hr)', flush=True)
    print(f'Results in: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
