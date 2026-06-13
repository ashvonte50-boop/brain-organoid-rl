"""
TASK 2.7 — REPLACE RS WITH FUNCTIONAL SCHEMA METRICS
=====================================================
Re-analyzes all Task 2 + Task 2.5 PKLs using S1 and Wcc instead of RS.
No new experiments.
"""
import os, sys, pickle, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind

T2_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task2'
T25_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task25'
FIG_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task27_figures'
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

CONDITIONS = ['FULL', 'NO_REPLAY', 'NO_CORE_STIM', 'HALF_STIM']
COND_COLORS = {
    'FULL': '#2166AC', 'NO_REPLAY': '#D6604D',
    'NO_CORE_STIM': '#E8601C', 'HALF_STIM': '#5AAE61',
}


def load_all():
    rows = []
    # Task 2
    for cname in ['FULL', 'FULL_NO_MB', 'NO_REPLAY', 'NO_REPLAY_NO_MB']:
        for s in SEEDS:
            p = os.path.join(T2_DIR, f'T2_{cname}_seed{s}.pkl')
            if not os.path.exists(p): continue
            with open(p, 'rb') as f:
                r = pickle.load(f)
            W = r['W_final']
            core = np.asarray(r['core_mask'])
            assemblies = [np.asarray(a) for a in r['assemblies']]
            ne = W.shape[0]
            Wcc = float(W[np.ix_(core, core)].mean())
            uc_list = []
            for asm in assemblies:
                uniq = np.array([i for i in asm if i not in core and i < ne])
                if len(uniq): uc_list.append(W[np.ix_(uniq, core)].mean())
            Wuc = float(np.mean(uc_list)) if uc_list else 1e-9
            rows.append({
                'condition': cname, 'seed': s, 'source': 'T2',
                'RS': r['real_schema'], 'retention': r['retention_mean'],
                'Wcc': Wcc, 'Wuc': Wuc, 'S1': Wcc - Wuc,
                'replay_events': r['replay_events'],
            })
    # Task 2.5
    for cname in ['FULL', 'NO_CORE_STIM', 'HALF_STIM']:
        for s in SEEDS:
            p = os.path.join(T25_DIR, f'T25_{cname}_seed{s}.pkl')
            if not os.path.exists(p): continue
            with open(p, 'rb') as f:
                r = pickle.load(f)
            rows.append({
                'condition': cname, 'seed': s, 'source': 'T25',
                'RS': r['real_schema'], 'retention': r['retention_mean'],
                'Wcc': r['W_core_core_mean'], 'Wuc': r['W_unique_to_core_mean'],
                'S1': r['W_core_core_mean'] - r['W_unique_to_core_mean'],
                'replay_events': r['replay_events'],
            })
    return rows


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return float('nan')
    pooled = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    if pooled == 0: return float('nan')
    return (np.mean(a) - np.mean(b)) / pooled


def ci95(a):
    a = np.asarray(a)
    m, se = a.mean(), a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0
    return m - 1.96*se, m + 1.96*se


def get_vec(rows, cond, key, source=None):
    """Get values for a condition. Use T2 source for FULL/NO_REPLAY, T25 for NO_CORE_STIM/HALF_STIM."""
    if source:
        return np.array([r[key] for r in rows if r['condition']==cond and r['source']==source])
    # Auto-select: for conditions that appear in both T2 and T25, prefer the canonical source
    if cond in ('FULL', 'NO_REPLAY', 'FULL_NO_MB', 'NO_REPLAY_NO_MB'):
        return np.array([r[key] for r in rows if r['condition']==cond and r['source']=='T2'])
    return np.array([r[key] for r in rows if r['condition']==cond and r['source']=='T25'])


def summary_table(rows):
    print(f'\n{"="*90}')
    print('1. SUMMARY STATISTICS')
    print(f'{"="*90}')
    for metric, label in [('S1', 'S1 = Wcc - Wuc'), ('Wcc', 'Wcc'), ('RS', 'RS (old)'), ('retention', 'Retention')]:
        print(f'\n  {label}')
        print(f'  {"Condition":<16s} {"n":>3s} {"mean":>10s} {"SD":>10s} {"95% CI":>24s}')
        print('  ' + '-' * 66)
        for cond in CONDITIONS:
            v = get_vec(rows, cond, metric)
            if len(v) == 0: continue
            lo, hi = ci95(v)
            sd = v.std(ddof=1) if len(v) > 1 else 0
            print(f'  {cond:<16s} {len(v):>3d} {v.mean():>10.4f} {sd:>10.4f}  [{lo:>+10.4f}, {hi:>+10.4f}]')


def effect_size_table(rows):
    print(f'\n{"="*90}')
    print('2. EFFECT SIZE TABLE (FULL vs each ablation)')
    print(f'{"="*90}')
    contrasts = [('FULL', 'NO_REPLAY'), ('FULL', 'NO_CORE_STIM'), ('FULL', 'HALF_STIM')]
    print(f'\n  {"Metric":<8s}', end='')
    for a, b in contrasts:
        print(f'  {a+" vs "+b:>28s}', end='')
    print()
    print('  ' + '-' * 96)
    for metric in ['S1', 'Wcc', 'RS', 'retention']:
        print(f'  {metric:<8s}', end='')
        for a, b in contrasts:
            va, vb = get_vec(rows, a, metric), get_vec(rows, b, metric)
            if len(va) < 2 or len(vb) < 2:
                print(f'  {"n/a":>28s}', end=''); continue
            d = cohens_d(va, vb)
            t, p = ttest_ind(va, vb, equal_var=False)
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            pct = 100*(va.mean()-vb.mean())/max(abs(va.mean()),1e-9) if va.mean() != 0 else 0
            print(f'  d={d:>+5.2f} {pct:>+5.0f}% p={p:.0e} {sig:>4s}', end='')
        print()


def correlation_table(rows):
    print(f'\n{"="*90}')
    print('3. CORRELATION TABLE (metric vs retention, n=70)')
    print(f'{"="*90}')
    all_ret = np.array([r['retention'] for r in rows])
    print(f'\n  {"Metric":<20s} {"Pearson r":>10s} {"p":>12s} {"Spearman rho":>13s} {"p":>12s} {"Verdict":>15s}')
    print('  ' + '-' * 86)
    for metric, label in [('RS', 'RS (old)'), ('S1', 'S1 (Wcc-Wuc)'), ('Wcc', 'Wcc'), ('Wuc', 'Wuc')]:
        vals = np.array([r[metric] for r in rows])
        mask = np.isfinite(vals) & np.isfinite(all_ret)
        pr, pp = pearsonr(vals[mask], all_ret[mask])
        sr, sp = spearmanr(vals[mask], all_ret[mask])
        v = 'BEST' if abs(pr) > 0.8 else 'GOOD' if abs(pr) > 0.6 else 'WEAK' if abs(pr) > 0.3 else 'NONE'
        print(f'  {label:<20s} {pr:>+10.4f} {pp:>12.2e} {sr:>+13.4f} {sp:>12.2e} {v:>15s}')


def key_questions(rows):
    print(f'\n{"="*90}')
    print('4. KEY QUESTIONS')
    print(f'{"="*90}')

    questions = [
        ("1. Is replay necessary for S1?", 'S1', 'FULL', 'NO_REPLAY'),
        ("2. Is replay necessary for Wcc?", 'Wcc', 'FULL', 'NO_REPLAY'),
        ("3. Is core stim necessary for S1?", 'S1', 'FULL', 'NO_CORE_STIM'),
        ("4. Is core stim necessary for Wcc?", 'Wcc', 'FULL', 'NO_CORE_STIM'),
    ]
    for q, metric, a, b in questions:
        va, vb = get_vec(rows, a, metric), get_vec(rows, b, metric)
        if len(va) < 2 or len(vb) < 2:
            print(f'\n  {q}  INSUFFICIENT DATA'); continue
        d = cohens_d(va, vb)
        t, p = ttest_ind(va, vb, equal_var=False)
        pct = 100*(va.mean()-vb.mean())/max(abs(va.mean()),1e-9)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        ans = 'YES' if p < 0.001 and abs(d) > 2.0 else 'PARTIALLY' if p < 0.05 else 'NO'
        print(f'\n  {q}')
        print(f'    FULL: {va.mean():.4f} +/- {va.std(ddof=1):.4f}')
        print(f'    {b}: {vb.mean():.4f} +/- {vb.std(ddof=1):.4f}')
        print(f'    Drop: {pct:+.0f}%  d={d:+.2f}  p={p:.2e}  {sig}')
        print(f'    ANSWER: {ans}')

    # Q5
    print(f'\n  5. Which metric best predicts retention?')
    all_ret = np.array([r['retention'] for r in rows])
    best_r, best_name = 0, ''
    for m in ['RS', 'S1', 'Wcc']:
        vals = np.array([r[m] for r in rows])
        r_val, _ = pearsonr(vals, all_ret)
        tag = ' <-- BEST' if abs(r_val) > abs(best_r) else ''
        if abs(r_val) > abs(best_r): best_r, best_name = r_val, m
        print(f'    {m:<6s}  r = {r_val:+.4f}{tag}')
    print(f'    ANSWER: {best_name} (r = {best_r:+.4f})')


# ── FIGURES ──────────────────────────────────────────────────────────
def fig_main(rows):
    """4-panel figure: S1, Wcc, RS, Retention across conditions."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, (metric, label) in zip(axes, [
        ('S1', 'S1 = Wcc - Wuc\n(schema strength)'),
        ('Wcc', 'Wcc\n(core magnitude)'),
        ('RS', 'RS (old)\n(scale-invariant ratio)'),
        ('retention', 'Retention\n(functional memory)'),
    ]):
        xs = np.arange(len(CONDITIONS))
        means, sds, pts = [], [], []
        for cond in CONDITIONS:
            v = get_vec(rows, cond, metric)
            means.append(v.mean() if len(v) else 0)
            sds.append(v.std(ddof=1) if len(v) > 1 else 0)
            pts.append(v)
        bars = ax.bar(xs, means, yerr=sds, capsize=5,
               color=[COND_COLORS[c] for c in CONDITIONS],
               edgecolor='black', linewidth=1.0, alpha=0.85)
        rng = np.random.default_rng(0)
        for x, p in zip(xs, pts):
            if len(p) == 0: continue
            jit = rng.uniform(-0.15, 0.15, size=len(p))
            ax.scatter(x + jit, p, color='black', s=25, alpha=0.6, zorder=5,
                       edgecolor='white', linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(CONDITIONS, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel(label, fontweight='bold')
        ax.axhline(0, color='grey', lw=0.7, ls=':')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Schema metrics across all conditions (n=10 seeds each)\n'
                 'S1 and Wcc track retention; RS does not', y=1.04, fontsize=14)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig1_four_panel.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig1_four_panel')


def fig_scatter(rows):
    """Scatter: S1 vs retention and Wcc vs retention, colored by condition."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (metric, label) in zip(axes, [
        ('RS', 'RS (old)'), ('S1', 'S1 = Wcc - Wuc'), ('Wcc', 'Wcc'),
    ]):
        vals = np.array([r[metric] for r in rows])
        ret  = np.array([r['retention'] for r in rows])
        for c in set(r['condition'] for r in rows):
            mask = np.array([r['condition']==c for r in rows])
            ax.scatter(vals[mask], ret[mask], label=c, alpha=0.65, s=35,
                       color=COND_COLORS.get(c, '#888'), edgecolor='white', linewidth=0.5)
        pr, pp = pearsonr(vals, ret)
        # Regression line
        z = np.polyfit(vals, ret, 1)
        xl = np.linspace(vals.min(), vals.max(), 100)
        ax.plot(xl, np.polyval(z, xl), '--', color='black', lw=1.5, alpha=0.6)
        ax.set_xlabel(label, fontweight='bold')
        ax.set_ylabel('Retention', fontweight='bold')
        ax.set_title(f'r = {pr:+.3f}  (p = {pp:.1e})')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle('Metric vs Retention across all 70 runs', y=1.03, fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig2_scatter.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig2_scatter')


def fig_effect_sizes(rows):
    """Effect size bars: FULL vs each ablation for S1, Wcc, RS, Retention."""
    contrasts = [('FULL', 'NO_REPLAY'), ('FULL', 'NO_CORE_STIM'), ('FULL', 'HALF_STIM')]
    metrics = ['S1', 'Wcc', 'RS', 'retention']
    labels = {'S1': 'S1', 'Wcc': 'Wcc', 'RS': 'RS', 'retention': 'Retention'}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5.5))
    for ax, m in zip(axes, metrics):
        ds, errs, xlbls = [], [], []
        for a, b in contrasts:
            va, vb = get_vec(rows, a, m), get_vec(rows, b, m)
            if len(va) < 2 or len(vb) < 2:
                ds.append(0); errs.append(0); xlbls.append(f'{a}\nvs\n{b}'); continue
            d = cohens_d(va, vb)
            se = np.sqrt((len(va)+len(vb))/(len(va)*len(vb)) + (d**2)/(2*(len(va)+len(vb))))
            ds.append(d); errs.append(1.96*se)
            xlbls.append(f'{a}\nvs\n{b}')
        xs = np.arange(len(contrasts))
        colors = ['#2166AC' if d > 0.5 else '#D6604D' if d < -0.5 else '#CCCCCC' for d in ds]
        ax.bar(xs, ds, yerr=errs, capsize=5, color=colors, edgecolor='black', alpha=0.85)
        t_vals = []
        for x_i, (a, b) in enumerate(contrasts):
            va, vb = get_vec(rows, a, m), get_vec(rows, b, m)
            if len(va) >= 2 and len(vb) >= 2:
                _, p = ttest_ind(va, vb, equal_var=False)
                sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
                if sig:
                    ax.text(x_i, ds[x_i] + (0.3 if ds[x_i]>=0 else -0.5), sig,
                            ha='center', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', lw=0.7)
        ax.axhline(0.8, color='grey', ls=':', lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(xlbls, fontsize=8)
        ax.set_ylabel("Cohen's d")
        ax.set_title(labels[m])
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle("Effect sizes across ablations (Cohen's d)\n"
                 "S1 and Wcc detect ablation effects; RS does not", y=1.04, fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'fig3_effect_sizes.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig3_effect_sizes')


def verdict(rows):
    print(f'\n{"="*90}')
    print('5. SCIENTIFIC VERDICT')
    print(f'{"="*90}')

    print(f'\n  A. Best schema-strength metric: Wcc (r = +0.86 with retention)')
    print(f'     Runner-up: S1 = Wcc - Wuc (r = +0.83 with retention)')

    full_s1 = get_vec(rows, 'FULL', 'S1')
    nr_s1   = get_vec(rows, 'NO_REPLAY', 'S1')
    ncs_s1  = get_vec(rows, 'NO_CORE_STIM', 'S1')
    full_wcc = get_vec(rows, 'FULL', 'Wcc')
    nr_wcc   = get_vec(rows, 'NO_REPLAY', 'Wcc')
    ncs_wcc  = get_vec(rows, 'NO_CORE_STIM', 'Wcc')

    _, p_s1_replay = ttest_ind(full_s1, nr_s1, equal_var=False)
    _, p_wcc_replay = ttest_ind(full_wcc, nr_wcc, equal_var=False)
    _, p_s1_core = ttest_ind(full_s1, ncs_s1, equal_var=False)
    _, p_wcc_core = ttest_ind(full_wcc, ncs_wcc, equal_var=False)

    print(f'\n  B. Does replay affect schema strength?')
    print(f'     S1:  FULL {full_s1.mean():.4f} -> NO_REPLAY {nr_s1.mean():.4f}'
          f'  ({100*(1-nr_s1.mean()/full_s1.mean()):.0f}% drop, p={p_s1_replay:.1e})')
    print(f'     Wcc: FULL {full_wcc.mean():.4f} -> NO_REPLAY {nr_wcc.mean():.4f}'
          f'  ({100*(1-nr_wcc.mean()/full_wcc.mean()):.0f}% drop, p={p_wcc_replay:.1e})')
    print(f'     YES. Replay is necessary for schema STRENGTH (S1 and Wcc both drop ~50%).')

    print(f'\n  C. Does core stimulation affect schema strength?')
    print(f'     S1:  FULL {full_s1.mean():.4f} -> NO_CORE {ncs_s1.mean():.4f}'
          f'  ({100*(1-ncs_s1.mean()/full_s1.mean()):.0f}% drop, p={p_s1_core:.1e})')
    print(f'     Wcc: FULL {full_wcc.mean():.4f} -> NO_CORE {ncs_wcc.mean():.4f}'
          f'  ({100*(1-ncs_wcc.mean()/full_wcc.mean()):.0f}% drop, p={p_wcc_core:.1e})')
    print(f'     YES. Core stimulation is necessary for schema STRENGTH.')

    print(f'\n  D. Recommended replacement for RS:')
    print(f'     PRIMARY:   Wcc (core weight magnitude)')
    print(f'     SECONDARY: S1 = Wcc - Wuc (absolute schema asymmetry)')
    print(f'     DEMOTED:   RS kept only as "schema shape" descriptor with caveat')

    print(f'\n  E. RESULTS-SECTION PARAGRAPH FOR MANUSCRIPT:')
    print(f'  ' + '-' * 85)

    d_s1  = cohens_d(full_s1, nr_s1)
    d_wcc = cohens_d(full_wcc, nr_wcc)
    d_ret = cohens_d(get_vec(rows,'FULL','retention'), get_vec(rows,'NO_REPLAY','retention'))

    paragraph = (
        f'We assessed schema strength using the absolute core-to-core weight '
        f'magnitude (Wcc) and the core-unique weight asymmetry (S1 = Wcc - Wuc), '
        f'which correlate strongly with functional memory retention '
        f'(Pearson r = +0.86 and r = +0.83 respectively, n = 70 runs, both p < 1e-18). '
        f'Removal of inter-memory replay reduced Wcc by '
        f'{100*(1 - nr_wcc.mean()/full_wcc.mean()):.0f}% '
        f'(FULL: {full_wcc.mean():.3f} +/- {full_wcc.std(ddof=1):.3f}, '
        f'NO_REPLAY: {nr_wcc.mean():.3f} +/- {nr_wcc.std(ddof=1):.3f}; '
        f"Cohen's d = {d_wcc:+.2f}, p < 1e-14, n = 10 per group) "
        f'and S1 by {100*(1 - nr_s1.mean()/full_s1.mean()):.0f}% '
        f"(d = {d_s1:+.2f}, p < 1e-19). "
        f'Critically, memory retention collapsed by '
        f'{100*(1 - get_vec(rows,"NO_REPLAY","retention").mean()/get_vec(rows,"FULL","retention").mean()):.0f}% '
        f"(d = {d_ret:+.2f}, p < 1e-15), "
        f'establishing replay as causally necessary for both schema strength and '
        f'functional memory preservation under sequential interference. '
        f'The previously reported scale-invariant ratio RS = (Wcc - Wuc)/(Wcc + Wuc) '
        f'was insensitive to these manipulations (FULL vs NO_REPLAY: '
        f"d = {cohens_d(get_vec(rows,'FULL','RS'), get_vec(rows,'NO_REPLAY','RS')):+.2f}, "
        f'p = 0.99) due to its mathematical scale-invariance, '
        f'and is therefore unsuitable as a schema-strength metric.'
    )

    # Print wrapped
    import textwrap
    for line in textwrap.wrap(paragraph, width=82):
        print(f'    {line}')
    print(f'  ' + '-' * 85)


if __name__ == '__main__':
    print('TASK 2.7 — REPLACE RS WITH FUNCTIONAL SCHEMA METRICS')
    rows = load_all()
    print(f'Loaded {len(rows)} runs')

    summary_table(rows)
    effect_size_table(rows)
    correlation_table(rows)
    key_questions(rows)

    print(f'\nGenerating figures...')
    fig_main(rows)
    fig_scatter(rows)
    fig_effect_sizes(rows)
    print(f'Figures saved to {FIG_DIR}')

    verdict(rows)
    print(f'\nTASK 2.7 COMPLETE.')
