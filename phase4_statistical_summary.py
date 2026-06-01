"""
PHASE 4 — Statistical Strengthening
=====================================
For all primary metrics: Cohen's d, bootstrap 95% CI, power estimates.
Generate a publication-ready summary table and forest plot.
"""
import sys, os, pickle, json
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, sem as scipy_sem
from scipy.stats import bootstrap as scipy_bootstrap

OUT = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-12))

def bootstrap_ci(x, n=5000, ci=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) == 1 else np.nan, np.nan, np.nan
    try:
        res = scipy_bootstrap((x,), np.mean, n_resamples=n, random_state=RNG)
        lo, hi = res.confidence_interval
        return float(np.mean(x)), float(lo), float(hi)
    except Exception:
        se = float(scipy_sem(x))
        m  = float(np.mean(x))
        return m, m - 1.96*se, m + 1.96*se

def power_estimate(a, b, alpha=0.05, two_tailed=True):
    """
    Approximate power via Cohen's d and n using normal approximation.
    Returns power in [0,1].
    """
    from scipy.stats import norm
    n_a, n_b = len(a), len(b)
    d   = abs(cohen_d(a, b))
    n_h = 2 / (1/n_a + 1/n_b)    # harmonic mean n
    se  = np.sqrt(1/n_a + 1/n_b)
    t_crit = 1.96 if two_tailed else 1.645
    ncp = d / se                  # non-centrality parameter
    beta = norm.cdf(t_crit - ncp) - norm.cdf(-t_crit - ncp)
    power = 1 - beta
    return float(np.clip(power, 0, 1))

def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


# ── Main ─────────────────────────────────────────────────────────────────────

def run_phase4():
    print('='*70, flush=True)
    print('PHASE 4: COMPREHENSIVE STATISTICAL SUMMARY', flush=True)
    print('='*70, flush=True)

    with open('figures/schema/distortion_data.pkl', 'rb') as f:
        data = pickle.load(f)

    MODES = ['no_replay', 'natural', 'hyper']
    LABELS = {'no_replay': 'No Replay', 'natural': 'Natural', 'hyper': 'Hyper'}

    def get_vals(mode, metric):
        if metric == 'Retention_A':
            return np.array([f[0] for f in data[mode].get('finals', []) if f], float)
        if metric == 'REAL_SCHEMA':
            return np.array(data[mode].get('real_schemas', []), float)
        if metric == 'DAI_core':
            return np.array([x.get('mean_core', np.nan)
                             for x in data[mode].get('directional_alignment', [])], float)
        if metric == 'DAI_unique':
            return np.array([x.get('mean_unique', np.nan)
                             for x in data[mode].get('directional_alignment', [])], float)
        if metric == 'Distortion':
            return np.array([s.get('distortion_index', np.nan)
                             for s in data[mode].get('schema', [])], float)
        if metric == 'SchemaScore':
            return np.array([s.get('schema_score', np.nan)
                             for s in data[mode].get('schema', [])], float)
        return np.array([])

    metrics = ['Retention_A', 'REAL_SCHEMA', 'DAI_core', 'DAI_unique',
               'Distortion', 'SchemaScore']

    # ── Per-condition descriptives ─────────────────────────────────────────
    print(f'\nDESCRIPTIVES (Mean [95% CI])', flush=True)
    desc_table = {}
    for metric in metrics:
        print(f'\n  {metric}:', flush=True)
        desc_table[metric] = {}
        for mode in MODES:
            v = get_vals(mode, metric)
            v = v[np.isfinite(v)]
            m, lo, hi = bootstrap_ci(v)
            desc_table[metric][mode] = {'mean': m, 'lo': lo, 'hi': hi,
                                        'sem': float(scipy_sem(v)) if len(v)>1 else 0.0,
                                        'n': int(len(v))}
            print(f'    {LABELS[mode]:12s}: {m:.4f} [{lo:.4f}, {hi:.4f}]  '
                  f'(SEM={scipy_sem(v):.4f}  n={len(v)})', flush=True)

    # ── Pairwise comparisons ───────────────────────────────────────────────
    print(f'\n\nPAIRWISE COMPARISONS', flush=True)
    pairs = [('natural', 'hyper'), ('natural', 'no_replay'), ('hyper', 'no_replay')]
    comparison_table = {}

    for metric in metrics:
        print(f'\n  {metric}:', flush=True)
        comparison_table[metric] = {}
        for ca, cb in pairs:
            va = get_vals(ca, metric); va = va[np.isfinite(va)]
            vb = get_vals(cb, metric); vb = vb[np.isfinite(vb)]
            if len(va) == 0: va = np.zeros(max(len(vb), 2))
            if len(vb) == 0: vb = np.zeros(max(len(va), 2))
            if len(va) < 2 or len(vb) < 2: continue

            t, p = ttest_ind(va, vb)
            d    = cohen_d(va, vb)
            pw   = power_estimate(va, vb)

            key = f'{ca}_vs_{cb}'
            comparison_table[metric][key] = {
                't': float(t), 'p': float(p), 'd': float(d), 'power': float(pw)
            }
            print(f'    {LABELS[ca]:12s} vs {LABELS[cb]:12s}: '
                  f't={t:+.3f}  p={p:.4f}  d={d:+.3f}  '
                  f'power={pw:.3f}  {stars(p)}', flush=True)

    # ── DAI vs 0 one-sample ────────────────────────────────────────────────
    print(f'\n\nDAI ONE-SAMPLE vs ZERO', flush=True)
    for mode in MODES:
        for key, label in [('DAI_core', 'DAI_core'), ('DAI_unique', 'DAI_unique')]:
            v = get_vals(mode, key); v = v[np.isfinite(v)]
            if len(v) < 2: continue
            t, p = ttest_1samp(v, 0.0)
            m, lo, hi = bootstrap_ci(v)
            d = float(np.mean(v) / (np.std(v, ddof=1) + 1e-12))  # Cohen d vs 0
            print(f'  {LABELS[mode]:12s} {label:12s}: '
                  f'mean={m:+.4f} [{lo:.4f},{hi:.4f}]  '
                  f't={t:+.3f}  p={p:.4e}  d={d:+.3f}  {stars(p)}', flush=True)

    # ── Print summary table ────────────────────────────────────────────────
    print(f'\n\nSUMMARY TABLE (Natural vs Hyper)', flush=True)
    print(f'  {"Metric":15s}  {"Nat mean":>10}  {"Hyp mean":>10}  '
          f'{"Cohen d":>9}  {"p-value":>10}  {"Power":>7}  {"Sig"}', flush=True)
    print('  ' + '-'*70, flush=True)

    for metric in metrics:
        va = get_vals('natural', metric); va = va[np.isfinite(va)]
        vb = get_vals('hyper',   metric); vb = vb[np.isfinite(vb)]
        if len(va) < 2 or len(vb) < 2:
            continue
        t, p = ttest_ind(va, vb)
        d  = cohen_d(va, vb)
        pw = power_estimate(va, vb)
        mn = float(np.mean(va))
        mh = float(np.mean(vb))
        print(f'  {metric:15s}  {mn:+10.4f}  {mh:+10.4f}  '
              f'{d:+9.3f}  {p:10.4f}  {pw:7.3f}  {stars(p)}', flush=True)

    # ── Save ──────────────────────────────────────────────────────────────
    def _safe(obj):
        if isinstance(obj, dict): return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_safe(x) for x in obj]
        if isinstance(obj, (np.floating, float)) and not np.isfinite(obj): return None
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        return obj

    save = {'descriptives': desc_table, 'comparisons': comparison_table}
    with open(os.path.join(OUT, 'phase4_statistics.json'), 'w') as f:
        json.dump(_safe(save), f, indent=2)
    print(f'\nSaved -> {OUT}/phase4_statistics.json', flush=True)

    return desc_table, comparison_table


if __name__ == '__main__':
    run_phase4()
