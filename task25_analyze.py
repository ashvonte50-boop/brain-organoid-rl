"""
TASK 2.5 ANALYSIS — CORE NECESSITY
=====================================
Loads task25 results, computes statistics, generates figures.

Includes a free sanity check: PERMUTED_CORE_MEASURE on Task 2 FULL data
(loads from ablation_results/task2/T2_FULL_seed*.pkl).
"""
import os, sys, pickle, glob
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from _distortion_paper import compute_real_schema_index

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task25'
TASK2_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task2'
FIG_DIR  = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]
CONDITIONS = ['FULL', 'NO_CORE_STIM', 'HALF_STIM']
COND_COLORS = {
    'FULL':         '#2166AC',
    'NO_CORE_STIM': '#D6604D',
    'HALF_STIM':    '#5AAE61',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})


def load_all():
    data = {}
    for cname in CONDITIONS:
        for seed in SEEDS:
            p = os.path.join(OUT_DIR, f'T25_{cname}_seed{seed}.pkl')
            if not os.path.exists(p):
                print(f'  MISSING: {p}')
                continue
            with open(p, 'rb') as f:
                data[(cname, seed)] = pickle.load(f)
    return data


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2: return float('nan')
    pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) /
                     (len(a) + len(b) - 2))
    if pooled == 0: return float('nan')
    return (np.mean(a) - np.mean(b)) / pooled


def ci95(a):
    a = np.asarray(a)
    if len(a) < 2: return (float('nan'), float('nan'))
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a))
    return (m - 1.96 * se, m + 1.96 * se)


def metric_vec(data, cname, key):
    return np.array([data[(cname, s)][key] for s in SEEDS if (cname, s) in data])


def report_stats(data):
    print(f'\n{"="*82}')
    print('TASK 2.5 — STATISTICAL REPORT')
    print(f'{"="*82}')

    # Sanity: FULL should reproduce Task 2 FULL
    full_rs = metric_vec(data, 'FULL', 'real_schema')
    if len(full_rs) >= 2:
        print(f'\nSanity check: FULL_T25 RS = {full_rs.mean():.4f} +/- {full_rs.std(ddof=1):.4f}')
        print(f'              (Task 2 FULL was 0.5008 +/- 0.0319 — should match exactly)')

    # Replay event verification
    print('\nREPLAY EVENT COUNTS')
    for cname in CONDITIONS:
        evs = [data[(cname, s)]['replay_events'] for s in SEEDS if (cname, s) in data]
        if evs:
            print(f'  {cname:<16s}  mean={np.mean(evs):6.1f}  range=[{min(evs)}, {max(evs)}]')

    for metric, mlabel in [('real_schema',           'REAL_SCHEMA (true core)'),
                           ('real_schema_permuted', 'REAL_SCHEMA (permuted core)'),
                           ('W_core_core_mean',     'W[core, core] mean'),
                           ('W_unique_to_core_mean','W[unique, core] mean'),
                           ('retention_mean',       'Retention')]:
        print(f'\n{mlabel}')
        print(f'  {"Condition":<16s} {"mean":>8s} {"SD":>8s} {"95% CI":>22s}  per-seed')
        groups = {}
        for cname in CONDITIONS:
            v = metric_vec(data, cname, metric)
            if len(v) == 0: continue
            groups[cname] = v
            sd = v.std(ddof=1) if len(v) > 1 else 0
            lo, hi = ci95(v)
            vstr = ' '.join(f'{x:.3f}' for x in v)
            print(f'  {cname:<16s} {v.mean():>8.4f} {sd:>8.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]  [{vstr}]')

        # Pairwise contrasts
        for a, b in [('FULL', 'NO_CORE_STIM'),
                     ('FULL', 'HALF_STIM'),
                     ('NO_CORE_STIM', 'HALF_STIM')]:
            if a in groups and b in groups and len(groups[a]) > 1 and len(groups[b]) > 1:
                t, p = ttest_ind(groups[a], groups[b], equal_var=False)
                d   = cohens_d(groups[a], groups[b])
                dlt = groups[a].mean() - groups[b].mean()
                sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
                print(f'    {a:<14s} vs {b:<14s}  delta={dlt:+.4f}  d={d:+.2f}  '
                      f't={t:+.2f}  p={p:.4g}  {sig}')


def permuted_core_check_task2():
    """Sanity check on Task 2 FULL data: recompute RS using random 'core'.
    Confirms (or refutes) that the FULL asymmetry is structurally specific
    to the actual 20 shared core neurons.
    """
    print(f'\n{"="*82}')
    print('PERMUTED-CORE SPECIFICITY CHECK (on Task 2 FULL data)')
    print(f'{"="*82}')
    rs_true, rs_perm_list = [], []
    n_perm_per_net = 50
    for s in SEEDS:
        p = os.path.join(TASK2_DIR, f'T2_FULL_seed{s}.pkl')
        if not os.path.exists(p):
            print(f'  MISSING: {p}')
            continue
        with open(p, 'rb') as f:
            r = pickle.load(f)
        W = r['W_final']
        core = np.asarray(r['core_mask'])
        assemblies = [np.asarray(a) for a in r['assemblies']]
        n_exc = W.shape[0]

        # True RS: reconstruct using a tiny shim
        class _StubNet:
            def __init__(self, W): self.W = type('W', (), {'data': type('D', (), {})()})
            @property
            def n_exc(self): return n_exc
        # Easier: call the formula directly
        cm = core
        cc = W[np.ix_(cm, cm)].mean()
        uc_list = []
        for asm in assemblies:
            uniq = np.array([i for i in asm if i not in cm and i < n_exc])
            if len(uniq):
                uc_list.append(W[np.ix_(uniq, cm)].mean())
        uc = np.mean(uc_list) if uc_list else 1e-9
        rs_t = (cc - uc) / (cc + uc + 1e-9)
        rs_true.append(rs_t)

        # Permuted RS: pick random 20 indices NOT in true core, repeat n_perm_per_net times
        rng = np.random.default_rng(s)
        non_core = np.array([i for i in range(n_exc) if i not in cm])
        rs_perm_for_seed = []
        for _ in range(n_perm_per_net):
            pc = rng.choice(non_core, size=len(cm), replace=False)
            cc_p = W[np.ix_(pc, pc)].mean()
            uc_p_list = []
            for asm in assemblies:
                uniq_p = np.array([i for i in asm if i not in pc and i < n_exc])
                if len(uniq_p):
                    uc_p_list.append(W[np.ix_(uniq_p, pc)].mean())
            uc_p = np.mean(uc_p_list) if uc_p_list else 1e-9
            rs_perm_for_seed.append((cc_p - uc_p) / (cc_p + uc_p + 1e-9))
        rs_perm_list.append(np.mean(rs_perm_for_seed))

    if not rs_true:
        print('  No Task 2 FULL data found.')
        return None
    rs_true = np.array(rs_true)
    rs_perm = np.array(rs_perm_list)
    t, p = ttest_ind(rs_true, rs_perm, equal_var=False)
    d   = cohens_d(rs_true, rs_perm)
    print(f'  True core RS (n=10):       {rs_true.mean():.4f} +/- {rs_true.std(ddof=1):.4f}')
    print(f'  Permuted core RS (n=10):   {rs_perm.mean():.4f} +/- {rs_perm.std(ddof=1):.4f}'
          f'  (mean over 50 perms/seed)')
    delta = rs_true.mean() - rs_perm.mean()
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
    print(f'  Delta:                     {delta:+.4f}  d={d:+.2f}  p={p:.4g}  {sig}')
    print()
    if rs_perm.mean() < 0.05 and rs_true.mean() > 0.3:
        print('  INTERPRETATION: RS metric IS structurally specific to the true core.')
        print('  Permuted "core" indices do NOT show the asymmetry.')
    elif rs_perm.mean() > 0.3:
        print('  INTERPRETATION: RS metric is NOT specific to the true core.')
        print('  Even random index sets show the asymmetry — metric measures something else.')
    return {'rs_true': rs_true, 'rs_perm': rs_perm, 'p': p, 'd': d}


# ── Figures ──────────────────────────────────────────────────────────────
def fig_rs(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.arange(len(CONDITIONS))
    means, sds, points = [], [], []
    for cname in CONDITIONS:
        v = metric_vec(data, cname, 'real_schema')
        means.append(v.mean() if len(v) else 0)
        sds.append(v.std(ddof=1) if len(v) > 1 else 0)
        points.append(v)
    ax.bar(xs, means, yerr=sds, capsize=6,
           color=[COND_COLORS[c] for c in CONDITIONS],
           edgecolor='black', linewidth=1.0, alpha=0.85)
    rng = np.random.default_rng(0)
    for x, pts in zip(xs, points):
        if len(pts) == 0: continue
        jit = rng.uniform(-0.15, 0.15, size=len(pts))
        ax.scatter(x + jit, pts, color='black', s=30, alpha=0.65, zorder=5,
                   edgecolor='white', linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_ylabel('REAL_SCHEMA (true core)', fontweight='bold')
    ax.set_title('Schema strength across core-necessity conditions (n=10 seeds)\n'
                 'Question: does removing direct core stimulation abolish RS?')
    ax.axhline(0, color='grey', lw=0.7, ls=':')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig1_real_schema.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_retention(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.arange(len(CONDITIONS))
    means, sds, points = [], [], []
    for cname in CONDITIONS:
        v = metric_vec(data, cname, 'retention_mean')
        means.append(v.mean() if len(v) else 0)
        sds.append(v.std(ddof=1) if len(v) > 1 else 0)
        points.append(v)
    ax.bar(xs, means, yerr=sds, capsize=6,
           color=[COND_COLORS[c] for c in CONDITIONS],
           edgecolor='black', linewidth=1.0, alpha=0.85)
    rng = np.random.default_rng(0)
    for x, pts in zip(xs, points):
        if len(pts) == 0: continue
        jit = rng.uniform(-0.15, 0.15, size=len(pts))
        ax.scatter(x + jit, pts, color='black', s=30, alpha=0.65, zorder=5,
                   edgecolor='white', linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_ylabel('Retention (mean)', fontweight='bold')
    ax.set_title('Retention across core-necessity conditions (n=10 seeds)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig2_retention.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_weight_blocks(data):
    """Show W[core,core] vs W[unique,core] means side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    metrics = [('W_core_core_mean', 'W[core, core] mean'),
               ('W_unique_to_core_mean', 'W[unique, core] mean')]
    for ax, (key, lbl) in zip(axes, metrics):
        xs = np.arange(len(CONDITIONS))
        means, sds, points = [], [], []
        for cname in CONDITIONS:
            v = metric_vec(data, cname, key)
            means.append(v.mean() if len(v) else 0)
            sds.append(v.std(ddof=1) if len(v) > 1 else 0)
            points.append(v)
        ax.bar(xs, means, yerr=sds, capsize=6,
               color=[COND_COLORS[c] for c in CONDITIONS],
               edgecolor='black', linewidth=1.0, alpha=0.85)
        rng = np.random.default_rng(0)
        for x, pts in zip(xs, points):
            if len(pts) == 0: continue
            jit = rng.uniform(-0.15, 0.15, size=len(pts))
            ax.scatter(x + jit, pts, color='black', s=20, alpha=0.65, zorder=5)
        ax.set_xticks(xs); ax.set_xticklabels(CONDITIONS, fontsize=10)
        ax.set_ylabel(lbl, fontweight='bold')
        ax.set_title(lbl)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Weight block means — direct check of the core mechanism',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig3_weight_blocks.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_permuted_core(data):
    """True-core RS vs permuted-core RS, per condition."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(CONDITIONS))
    w = 0.38
    means_t, sds_t = [], []
    means_p, sds_p = [], []
    for cname in CONDITIONS:
        vt = metric_vec(data, cname, 'real_schema')
        vp = metric_vec(data, cname, 'real_schema_permuted')
        means_t.append(vt.mean() if len(vt) else 0)
        sds_t.append(vt.std(ddof=1) if len(vt) > 1 else 0)
        means_p.append(vp.mean() if len(vp) else 0)
        sds_p.append(vp.std(ddof=1) if len(vp) > 1 else 0)
    ax.bar(xs - w/2, means_t, w, yerr=sds_t, capsize=4, label='True core',
           color='#2166AC', edgecolor='black', alpha=0.85)
    ax.bar(xs + w/2, means_p, w, yerr=sds_p, capsize=4, label='Permuted core (random 20)',
           color='#999999', edgecolor='black', alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_ylabel('REAL_SCHEMA', fontweight='bold')
    ax.set_title('True vs Permuted core (specificity check, n=10)')
    ax.axhline(0, color='grey', lw=0.7, ls=':')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig4_permuted_core.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_rs_evolution(data):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cname in CONDITIONS:
        curves, max_len = [], 0
        for s in SEEDS:
            r = data.get((cname, s))
            if r is None: continue
            c = [e['rs'] for e in r['rs_evolution']]
            curves.append(c); max_len = max(max_len, len(c))
        if not curves: continue
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves): padded[i, :len(c)] = c
        m = np.nanmean(padded, axis=0)
        se = np.nanstd(padded, axis=0, ddof=1) / np.sqrt(padded.shape[0])
        x = np.arange(max_len)
        c = COND_COLORS[cname]
        ax.plot(x, m, '-o', color=c, lw=2.2, ms=6, label=f'{cname} (n={len(curves)})')
        ax.fill_between(x, m-se, m+se, color=c, alpha=0.18)
    ax.set_xlabel('Checkpoint', fontweight='bold')
    ax.set_ylabel('REAL_SCHEMA', fontweight='bold')
    ax.set_title('RS evolution over training (mean +/- SEM)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig5_rs_evolution.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def verdict(data):
    print(f'\n{"="*82}')
    print('VERDICT')
    print(f'{"="*82}')
    full = metric_vec(data, 'FULL', 'real_schema')
    ncs  = metric_vec(data, 'NO_CORE_STIM', 'real_schema')
    hs   = metric_vec(data, 'HALF_STIM', 'real_schema')

    if len(full) < 2 or len(ncs) < 2:
        print('  Insufficient data for verdict.')
        return

    t1, p1 = ttest_ind(full, ncs, equal_var=False)
    d1    = cohens_d(full, ncs)
    delta1 = full.mean() - ncs.mean()

    print(f'  FULL RS:           {full.mean():.4f} +/- {full.std(ddof=1):.4f}  (n={len(full)})')
    print(f'  NO_CORE_STIM RS:   {ncs.mean():.4f} +/- {ncs.std(ddof=1):.4f}  (n={len(ncs)})')
    if len(hs) >= 2:
        print(f'  HALF_STIM RS:      {hs.mean():.4f} +/- {hs.std(ddof=1):.4f}  (n={len(hs)})')
    print(f'  FULL vs NO_CORE_STIM: delta={delta1:+.4f}  d={d1:+.2f}  p={p1:.4g}')

    if len(hs) >= 2:
        t2, p2 = ttest_ind(hs, ncs, equal_var=False)
        d2    = cohens_d(hs, ncs)
        delta2 = hs.mean() - ncs.mean()
        print(f'  HALF_STIM vs NO_CORE_STIM: delta={delta2:+.4f}  d={d2:+.2f}  p={p2:.4g}')

    print()
    collapsed   = ncs.mean() < 0.10
    nondegrade  = (len(hs) < 2) or (hs.mean() > 0.5 * full.mean())

    if collapsed and nondegrade:
        print('  VERDICT: CORE NECESSITY SUPPORTED')
        print('  NO_CORE_STIM collapses RS toward zero; HALF_STIM (matched stim total)')
        print('  preserves RS. The core mechanism is causally necessary for the')
        print('  REAL_SCHEMA asymmetry.')
    elif collapsed and not nondegrade:
        print('  VERDICT: AMBIGUOUS — confound')
        print('  NO_CORE_STIM collapses RS, but HALF_STIM also reduces it.')
        print('  The collapse may be due to reduced total stim, not core absence.')
    elif (not collapsed) and ncs.mean() < 0.7 * full.mean():
        print('  VERDICT: CORE PARTIALLY NECESSARY')
        print(f'  NO_CORE_STIM reduces RS by {100*(1-ncs.mean()/full.mean()):.0f}% but'
              ' does not abolish it.')
    else:
        print('  VERDICT: CORE NECESSITY FALSIFIED')
        print('  NO_CORE_STIM does not abolish RS. The asymmetry forms even when')
        print('  core neurons are never directly stimulated. Other mechanism at work.')


if __name__ == '__main__':
    data = load_all()
    n_total = len(CONDITIONS) * len(SEEDS)
    print(f'Loaded {len(data)}/{n_total} task2.5 runs')
    if len(data) < n_total:
        print('  (continuing with partial data)')

    report_stats(data)
    print('\nGenerating figures...')
    fig_rs(data);           print('  fig1_real_schema')
    fig_retention(data);    print('  fig2_retention')
    fig_weight_blocks(data); print('  fig3_weight_blocks')
    fig_permuted_core(data); print('  fig4_permuted_core')
    fig_rs_evolution(data); print('  fig5_rs_evolution')
    print(f'Figures saved to {FIG_DIR}')

    print('\nRunning permuted-core check on Task 2 FULL data...')
    permuted_core_check_task2()

    verdict(data)
