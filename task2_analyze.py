"""
TASK 2 ANALYSIS — REPLAY NECESSITY
====================================
Loads results from ablation_results/task2/, computes statistics, generates 5 figures.

Figures (PNG + PDF + SVG):
  Fig 1: REAL_SCHEMA across conditions (bar + dots)
  Fig 2: Retention across conditions
  Fig 3: RS evolution curves (per condition, across checkpoints)
  Fig 4: Representative final weight matrices (1 seed per condition)
  Fig 5: Effect-size summary (Cohen's d, mean diff, CI)
"""
import os, sys, pickle, glob
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task2'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]
CONDITIONS = ['FULL', 'FULL_NO_MB', 'NO_REPLAY', 'NO_REPLAY_NO_MB']
COND_COLORS = {
    'FULL':            '#2166AC',
    'FULL_NO_MB':      '#5AAE61',
    'NO_REPLAY':       '#D6604D',
    'NO_REPLAY_NO_MB': '#B2182B',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
})


def load_all():
    """Load all PKL results, indexed by (condition, seed)."""
    data = {}
    for cname in CONDITIONS:
        for seed in SEEDS:
            p = os.path.join(OUT_DIR, f'T2_{cname}_seed{seed}.pkl')
            if not os.path.exists(p):
                print(f'  MISSING: {p}')
                continue
            with open(p, 'rb') as f:
                data[(cname, seed)] = pickle.load(f)
    return data


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float('nan')
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0: return float('nan')
    return (np.mean(a) - np.mean(b)) / pooled


def metric_vec(data, cname, key):
    vals = []
    for s in SEEDS:
        r = data.get((cname, s))
        if r is None: continue
        vals.append(r[key])
    return np.array(vals)


def report_stats(data):
    print(f'\n{"="*80}')
    print('TASK 2 — STATISTICAL REPORT')
    print(f'{"="*80}')

    # Replay event sanity
    print('\nREPLAY EVENT COUNTS (sanity check)')
    for cname in CONDITIONS:
        evs = [data[(cname, s)]['replay_events'] for s in SEEDS if (cname, s) in data]
        print(f'  {cname:<18s}  mean={np.mean(evs):6.1f}  '
              f'range=[{min(evs)}, {max(evs)}]  n={len(evs)}')

    for metric, mlabel in [('real_schema',    'REAL_SCHEMA'),
                           ('retention_mean', 'Retention (mean)'),
                           ('dai_core',       'DAI_core')]:
        print(f'\n{mlabel}')
        print(f'  {"Condition":<18s} {"mean":>8s} {"SD":>8s} {"n":>3s}  per-seed')
        groups = {}
        for cname in CONDITIONS:
            v = metric_vec(data, cname, metric)
            if len(v) == 0: continue
            groups[cname] = v
            vstr = ' '.join(f'{x:.3f}' for x in v)
            print(f'  {cname:<18s} {v.mean():>8.4f} {v.std(ddof=1) if len(v)>1 else 0:>8.4f} '
                  f'{len(v):>3d}  [{vstr}]')

        # Pairwise comparisons
        for a, b in [('FULL', 'NO_REPLAY'),
                     ('FULL_NO_MB', 'NO_REPLAY_NO_MB'),
                     ('FULL', 'FULL_NO_MB'),
                     ('NO_REPLAY', 'NO_REPLAY_NO_MB')]:
            if a in groups and b in groups and len(groups[a]) > 1 and len(groups[b]) > 1:
                t, p = ttest_ind(groups[a], groups[b], equal_var=False)
                d   = cohens_d(groups[a], groups[b])
                dlt = groups[a].mean() - groups[b].mean()
                sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
                print(f'    {a:<16s} vs {b:<16s}  delta={dlt:+.4f}  '
                      f'd={d:+.2f}  t={t:+.2f}  p={p:.4f}  {sig}')


# ── Figure 1: REAL_SCHEMA ────────────────────────────────────────────────
def fig_rs(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.arange(len(CONDITIONS))
    means, sds, points = [], [], []
    for cname in CONDITIONS:
        v = metric_vec(data, cname, 'real_schema')
        means.append(v.mean() if len(v) else 0)
        sds.append(v.std(ddof=1) if len(v) > 1 else 0)
        points.append(v)
    bars = ax.bar(xs, means, yerr=sds, capsize=6,
                  color=[COND_COLORS[c] for c in CONDITIONS],
                  edgecolor='black', linewidth=1.0, alpha=0.85)
    rng = np.random.default_rng(0)
    for x, pts in zip(xs, points):
        if len(pts) == 0: continue
        jit = rng.uniform(-0.15, 0.15, size=len(pts))
        ax.scatter(x + jit, pts, color='black', s=30, alpha=0.65, zorder=5,
                   edgecolor='white', linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_ylabel('REAL_SCHEMA', fontweight='bold')
    ax.set_title('Schema strength across conditions (n=10 seeds)\n'
                 'Question: does replay form schema?')
    ax.axhline(0, color='grey', lw=0.7, ls=':')
    ax.set_ylim(min(0, *[m-s for m,s in zip(means, sds)]) - 0.05,
                max([m+s for m,s in zip(means, sds)]) + 0.10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig1_real_schema.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Figure 2: Retention ──────────────────────────────────────────────────
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
    ax.set_ylabel('Retention (mean across A-D)', fontweight='bold')
    ax.set_title('Retention across conditions (n=10 seeds)')
    ax.axhline(0, color='grey', lw=0.7, ls=':')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig2_retention.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Figure 3: RS evolution curves ────────────────────────────────────────
def fig_rs_evolution(data):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Collapse all per-seed rs_evolution onto checkpoint axis
    # Stage labels in order: baseline -> post_encode/post_replay alternating -> final
    for cname in CONDITIONS:
        all_curves = []
        max_len = 0
        for s in SEEDS:
            r = data.get((cname, s))
            if r is None: continue
            curve = [e['rs'] for e in r['rs_evolution']]
            all_curves.append(curve)
            max_len = max(max_len, len(curve))
        if not all_curves: continue
        # Pad to max_len with last value
        padded = np.full((len(all_curves), max_len), np.nan)
        for i, c in enumerate(all_curves):
            padded[i, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        sem  = np.nanstd(padded, axis=0, ddof=1) / np.sqrt(padded.shape[0])
        xs = np.arange(max_len)
        c = COND_COLORS[cname]
        ax.plot(xs, mean, '-o', color=c, lw=2.2, ms=6, label=f'{cname} (n={len(all_curves)})')
        ax.fill_between(xs, mean - sem, mean + sem, color=c, alpha=0.18)

    ax.set_xlabel('Checkpoint (baseline → encode/replay per memory → final)',
                  fontweight='bold')
    ax.set_ylabel('REAL_SCHEMA', fontweight='bold')
    ax.set_title('RS evolution over training\n(mean ± SEM across seeds)')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig3_rs_evolution.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Figure 4: Representative weight matrices ─────────────────────────────
def fig_weight_matrices(data):
    # Pick seed=42 as the representative seed
    seed_pick = 42
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.0 * len(CONDITIONS), 4.2))
    if len(CONDITIONS) == 1:
        axes = [axes]
    # Use common color scale across panels
    Ws = {}
    for cname in CONDITIONS:
        r = data.get((cname, seed_pick))
        if r is None: continue
        Ws[cname] = r['W_final']
    if not Ws:
        plt.close(fig); return
    vmax = max(W.max() for W in Ws.values())

    for ax, cname in zip(axes, CONDITIONS):
        W = Ws.get(cname)
        if W is None:
            ax.set_visible(False); continue
        # Crop to schema-relevant block: core (0..20) + 4 unique pools (20..100)
        crop_n = 100
        Wc = W[:crop_n, :crop_n]
        im = ax.imshow(Wc, cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.axhline(20, color='white', lw=0.8, alpha=0.7)
        ax.axvline(20, color='white', lw=0.8, alpha=0.7)
        for k in range(40, 100, 20):
            ax.axhline(k, color='white', lw=0.5, alpha=0.4)
            ax.axvline(k, color='white', lw=0.5, alpha=0.4)
        ax.set_title(f'{cname}\n(seed={seed_pick})', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('pre →')
        ax.set_ylabel('post →')

    cbar = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02, shrink=0.85)
    cbar.set_label('Weight')
    fig.suptitle(f'Final weight matrices (top-left 100x100: core ∪ unique pools)\n'
                 f'White lines: core/unique boundaries', y=1.02, fontsize=12)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig4_weight_matrices.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Figure 5: Effect-size summary ────────────────────────────────────────
def fig_effect_sizes(data):
    contrasts = [
        ('FULL',       'NO_REPLAY'),
        ('FULL_NO_MB', 'NO_REPLAY_NO_MB'),
        ('FULL',       'FULL_NO_MB'),
        ('NO_REPLAY',  'NO_REPLAY_NO_MB'),
    ]
    metrics = [('real_schema', 'RS'), ('retention_mean', 'Ret'), ('dai_core', 'DAI')]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 5.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (key, lbl) in zip(axes, metrics):
        ds, errs, sigs, labels = [], [], [], []
        for a, b in contrasts:
            va = metric_vec(data, a, key)
            vb = metric_vec(data, b, key)
            if len(va) < 2 or len(vb) < 2:
                ds.append(0); errs.append(0); sigs.append(''); labels.append(f'{a}\nvs\n{b}'); continue
            d  = cohens_d(va, vb)
            t, p = ttest_ind(va, vb, equal_var=False)
            # Bootstrap-ish 95% CI for Cohen's d (approximate from SEs)
            se = np.sqrt((len(va)+len(vb))/(len(va)*len(vb)) +
                         (d**2)/(2*(len(va)+len(vb))))
            ds.append(d); errs.append(1.96 * se)
            sigs.append('***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')
            labels.append(f'{a}\nvs\n{b}')

        xs = np.arange(len(contrasts))
        colors = ['#2166AC' if d > 0 else '#D6604D' for d in ds]
        ax.bar(xs, ds, yerr=errs, capsize=5, color=colors, edgecolor='black',
               linewidth=1.0, alpha=0.85)
        for x, d, s in zip(xs, ds, sigs):
            if s:
                ax.text(x, d + (0.05 if d >= 0 else -0.15), s,
                        ha='center', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', lw=0.7)
        ax.axhline(0.8, color='grey', ls=':', lw=0.6, label='large effect (d=0.8)')
        ax.axhline(-0.8, color='grey', ls=':', lw=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Cohen's d")
        ax.set_title(f'Effect sizes — {lbl}')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Effect sizes (Cohen\'s d) across key contrasts', y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig5_effect_sizes.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def verdict(data):
    print(f'\n{"="*80}')
    print('VERDICT')
    print(f'{"="*80}')
    rs_full      = metric_vec(data, 'FULL',       'real_schema')
    rs_no_replay = metric_vec(data, 'NO_REPLAY',  'real_schema')
    rs_nomb      = metric_vec(data, 'FULL_NO_MB', 'real_schema')
    rs_nr_nomb   = metric_vec(data, 'NO_REPLAY_NO_MB', 'real_schema')

    if len(rs_full) < 2 or len(rs_no_replay) < 2:
        print('  Insufficient data for verdict.')
        return

    t, p = ttest_ind(rs_full, rs_no_replay, equal_var=False)
    d    = cohens_d(rs_full, rs_no_replay)
    dlt  = rs_full.mean() - rs_no_replay.mean()
    print(f'  FULL RS:        {rs_full.mean():.4f} +/- {rs_full.std(ddof=1):.4f}  (n={len(rs_full)})')
    print(f'  NO_REPLAY RS:   {rs_no_replay.mean():.4f} +/- {rs_no_replay.std(ddof=1):.4f}  (n={len(rs_no_replay)})')
    print(f'  Delta:          {dlt:+.4f}  Cohen d: {d:+.2f}  p={p:.4g}')

    if len(rs_nomb) >= 2 and len(rs_nr_nomb) >= 2:
        t2, p2 = ttest_ind(rs_nomb, rs_nr_nomb, equal_var=False)
        d2     = cohens_d(rs_nomb, rs_nr_nomb)
        dlt2   = rs_nomb.mean() - rs_nr_nomb.mean()
        print(f'  FULL_NO_MB RS:        {rs_nomb.mean():.4f} +/- {rs_nomb.std(ddof=1):.4f}')
        print(f'  NO_REPLAY_NO_MB RS:   {rs_nr_nomb.mean():.4f} +/- {rs_nr_nomb.std(ddof=1):.4f}')
        print(f'  Delta(MB-removed contrast): {dlt2:+.4f}  d={d2:+.2f}  p={p2:.4g}')

    print()
    # Decision logic
    collapse = (rs_no_replay.mean() < 0.10) and (p < 0.01)
    preserved = (rs_no_replay.mean() > 0.5 * rs_full.mean()) and (p > 0.05)

    if collapse:
        print('  VERDICT: REPLAY NECESSARY')
        print('  NO_REPLAY collapses to near-zero schema.')
        print('  Schema formation requires replay.')
    elif preserved:
        print('  VERDICT: REPLAY NOT NECESSARY')
        print('  NO_REPLAY preserves schema. Architecture alone generates it.')
        print('  This is a NEGATIVE result for the replay hypothesis.')
    else:
        # Partial — schema reduced but not gone
        pct = 100 * (1 - rs_no_replay.mean() / rs_full.mean())
        print('  VERDICT: REPLAY CONTRIBUTORY')
        print(f'  Replay accounts for ~{pct:.0f}% of schema strength but is not the sole source.')


if __name__ == '__main__':
    data = load_all()
    n_total = len(CONDITIONS) * len(SEEDS)
    print(f'Loaded {len(data)}/{n_total} runs')
    if len(data) < n_total:
        print('  Continuing with partial data...')
    report_stats(data)
    print('\nGenerating figures...')
    fig_rs(data);          print('  fig1_real_schema')
    fig_retention(data);   print('  fig2_retention')
    fig_rs_evolution(data); print('  fig3_rs_evolution')
    fig_weight_matrices(data); print('  fig4_weight_matrices')
    fig_effect_sizes(data); print('  fig5_effect_sizes')
    print(f'Figures saved to {FIG_DIR}')
    verdict(data)
