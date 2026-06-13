"""
TASK 5.5 ANALYSIS — Formation-time Causal Test of Wcc
======================================================
Tables, 4 figures, statistics, verdict, CSV, TASK55_REPORT.md
"""
import os, sys, pickle, csv, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task55'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042]
CONDS = ['FULL', 'WCC_FROZEN', 'WCC_CLAMPED_ZERO', 'WCC_NO_STDP']
COLORS = {
    'FULL':            '#2166AC',
    'WCC_FROZEN':      '#F4A736',
    'WCC_CLAMPED_ZERO':'#D6604D',
    'WCC_NO_STDP':     '#9970AB',
}
LABELS = {
    'FULL':            'FULL\n(no intervention)',
    'WCC_FROZEN':      'WCC_FROZEN\n(init values locked)',
    'WCC_CLAMPED_ZERO':'WCC_CLAMPED\n(forced to 0)',
    'WCC_NO_STDP':     'WCC_NO_STDP\n(plastic_mask=0)',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})


def load():
    data = {}
    for s in SEEDS:
        p = os.path.join(OUT_DIR, f'T55_seed{s}.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                data[s] = pickle.load(f)
        else:
            print(f'MISSING {p}')
    return data


def vec(data, cond, key):
    return np.array([data[s]['conditions'][cond][key]
                     for s in SEEDS if s in data and cond in data[s]['conditions']])


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    pool = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                    (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / pool if pool else float('nan')


def per_seed_table(data):
    print(f'\n{"=" * 110}')
    print('TABLE 1: PER-SEED RESULTS (formation-time interventions)')
    print(f'{"=" * 110}')
    print(f'  {"Seed":>5s} {"Condition":<18s} {"Wcc":>8s} {"Wuc":>8s} {"Wuu":>8s} '
          f'{"S1":>8s} {"Retention":>10s} {"Retrieval":>10s} {"Replay":>7s}')
    print('  ' + '-' * 90)
    for s in SEEDS:
        if s not in data:
            continue
        for cond in CONDS:
            if cond not in data[s]['conditions']:
                continue
            c = data[s]['conditions'][cond]
            print(f'  {s:>5d} {cond:<18s} {c["Wcc"]:>8.4f} {c["Wuc"]:>8.4f} {c["Wuu"]:>8.4f} '
                  f'{c["S1"]:>8.4f} {c["retention_mean"]:>10.4f} {c["retrieval_mean"]:>10.4f} '
                  f'{c["replay_events"]:>7d}')
        print()


def means_table(data):
    print(f'\n{"=" * 96}')
    print('TABLE 2: CONDITION MEANS (n=2, mean +/- SEM)')
    print(f'{"=" * 96}')
    print(f'  {"Condition":<18s} {"Wcc":>14s} {"Wuc":>14s} {"Wuu":>14s} '
          f'{"Retention":>16s} {"Retrieval":>16s}')
    print('  ' + '-' * 94)
    for cond in CONDS:
        def fmt(k):
            v = vec(data, cond, k)
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            return f'{v.mean():.4f}+/-{se:.4f}'
        print(f'  {cond:<18s} {fmt("Wcc"):>14s} {fmt("Wuc"):>14s} {fmt("Wuu"):>14s} '
              f'{fmt("retention_mean"):>16s} {fmt("retrieval_mean"):>16s}')


def stats_table(data):
    print(f'\n{"=" * 96}')
    print('TABLE 3: STATISTICS — FULL vs intervention (Welch t-test, n=2)')
    print(f'{"=" * 96}')
    full_r = vec(data, 'FULL', 'retention_mean')
    for metric in ['retention_mean', 'retrieval_mean', 'Wcc', 'S1']:
        print(f'\n  {metric}:')
        full = vec(data, 'FULL', metric)
        for cond in ['WCC_FROZEN', 'WCC_CLAMPED_ZERO', 'WCC_NO_STDP']:
            cv = vec(data, cond, metric)
            if len(full) < 2 or len(cv) < 2:
                print(f'    FULL vs {cond:<18s}  insufficient data')
                continue
            t, p = ttest_ind(full, cv, equal_var=False)
            dd = cohen_d(full, cv)
            dlt = full.mean() - cv.mean()
            pct = 100 * dlt / max(abs(full.mean()), 1e-9)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f'    FULL vs {cond:<18s} delta={dlt:+.4f} ({pct:+.0f}%) '
                  f'd={dd:+.2f} t={t:+.2f} p={p:.4g} {sig}')


# ─── Figures ──────────────────────────────────────────────────────────────────

def _save_fig(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}')


def figures(data):
    xs = np.arange(len(CONDS))
    rng = np.random.default_rng(0)

    # ── Fig 1: Retention by condition ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cond in enumerate(CONDS):
        v = vec(data, cond, 'retention_mean')
        if len(v) == 0:
            continue
        err = v.std(ddof=1) if len(v) > 1 else 0
        ax.bar(i, v.mean(), yerr=err, capsize=6,
               color=COLORS[cond], edgecolor='black', alpha=0.85)
        jit = rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(i + jit, v, color='black', s=40, alpha=0.7, zorder=5,
                   edgecolor='white', linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=9)
    ax.set_ylabel('Retention (isyn_score mean)', fontweight='bold')
    ax.set_title('Fig 1 — Retention by condition\n'
                 'Formation-time Wcc intervention (n=2 seeds per condition)', pad=8)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    fig.tight_layout()
    _save_fig(fig, 'fig1_retention')

    # ── Fig 2: Final Wcc by condition ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cond in enumerate(CONDS):
        v = vec(data, cond, 'Wcc')
        if len(v) == 0:
            continue
        err = v.std(ddof=1) if len(v) > 1 else 0
        ax.bar(i, v.mean(), yerr=err, capsize=6,
               color=COLORS[cond], edgecolor='black', alpha=0.85)
        jit = rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(i + jit, v, color='black', s=40, alpha=0.7, zorder=5,
                   edgecolor='white', linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=9)
    ax.set_ylabel('Final Wcc (core-core weight mean)', fontweight='bold')
    ax.set_title('Fig 2 — Final Wcc by condition\n'
                 'Verifies intervention successfully suppressed Wcc growth', pad=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'fig2_wcc')

    # ── Fig 3: Weight decomposition (Wcc, Wuc, Wuu) ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
    for ax_i, (wkey, wlabel) in enumerate([('Wcc', 'Wcc'), ('Wuc', 'Wuc'), ('Wuu', 'Wuu')]):
        ax = axes[ax_i]
        for i, cond in enumerate(CONDS):
            v = vec(data, cond, wkey)
            if len(v) == 0:
                continue
            err = v.std(ddof=1) if len(v) > 1 else 0
            ax.bar(i, v.mean(), yerr=err, capsize=5,
                   color=COLORS[cond], edgecolor='black', alpha=0.85)
        ax.set_xticks(np.arange(len(CONDS)))
        ax.set_xticklabels([c.replace('WCC_', '') for c in CONDS], fontsize=8, rotation=20)
        ax.set_ylabel(wlabel, fontweight='bold')
        ax.set_title(f'{wlabel} by condition', pad=6)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Fig 3 — Weight decomposition: do Wuc/Wuu compensate for missing Wcc?',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    _save_fig(fig, 'fig3_weight_decomposition')

    # ── Fig 4: Effect sizes (Cohen's d vs FULL) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    int_conds = ['WCC_FROZEN', 'WCC_CLAMPED_ZERO', 'WCC_NO_STDP']
    xi = np.arange(len(int_conds))
    full_r = vec(data, 'FULL', 'retention_mean')
    ds, ps = [], []
    for cond in int_conds:
        cv = vec(data, cond, 'retention_mean')
        ds.append(cohen_d(full_r, cv))
        if len(full_r) >= 2 and len(cv) >= 2:
            _, p = ttest_ind(full_r, cv, equal_var=False)
        else:
            p = float('nan')
        ps.append(p)
    bar_colors = ['#D6604D' if dd > 0 else '#5AAE61' for dd in ds]
    bars = ax.bar(xi, ds, color=bar_colors, edgecolor='black', alpha=0.85)
    for i, (cond, dd, p) in enumerate(zip(int_conds, ds, ps)):
        if np.isnan(p):
            sig = 'n.d.'
        else:
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        offset = 0.3 if dd >= 0 else -0.5
        ax.text(i, dd + offset, sig, ha='center', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', lw=0.9)
    ax.set_xticks(xi)
    ax.set_xticklabels([LABELS[c] for c in int_conds], fontsize=9)
    ax.set_ylabel("Cohen's d  (FULL − intervention, retention)", fontweight='bold')
    ax.set_title("Fig 4 — Effect sizes: formation-time Wcc suppression\n"
                 "Positive d = FULL retained more than intervention", pad=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, 'fig4_effect_sizes')


# ─── Verdict + Report + CSV ───────────────────────────────────────────────────

def verdict_and_report(data):
    full_r  = vec(data, 'FULL',            'retention_mean')
    froz_r  = vec(data, 'WCC_FROZEN',      'retention_mean')
    zero_r  = vec(data, 'WCC_CLAMPED_ZERO','retention_mean')
    nstdp_r = vec(data, 'WCC_NO_STDP',     'retention_mean')

    def cmp(a, b, label):
        if len(a) < 2 or len(b) < 2:
            return float('nan'), float('nan'), float('nan')
        t, p = ttest_ind(a, b, equal_var=False)
        d = cohen_d(a, b)
        return t, p, d

    t_froz,  p_froz,  d_froz  = cmp(full_r, froz_r,  'FROZEN')
    t_zero,  p_zero,  d_zero  = cmp(full_r, zero_r,  'ZERO')
    t_nstdp, p_nstdp, d_nstdp = cmp(full_r, nstdp_r, 'NO_STDP')

    print(f'\n{"=" * 82}\nVERDICT — TASK 5.5\n{"=" * 82}')
    print(f'  FULL            : {full_r.mean():.4f} +/- {full_r.std(ddof=1) if len(full_r)>1 else 0:.4f}')
    print(f'  WCC_FROZEN      : {froz_r.mean():.4f} +/- {froz_r.std(ddof=1) if len(froz_r)>1 else 0:.4f}  '
          f'(d={d_froz:+.2f}, p={p_froz:.3g})')
    print(f'  WCC_CLAMPED_ZERO: {zero_r.mean():.4f} +/- {zero_r.std(ddof=1) if len(zero_r)>1 else 0:.4f}  '
          f'(d={d_zero:+.2f}, p={p_zero:.3g})')
    print(f'  WCC_NO_STDP     : {nstdp_r.mean():.4f} +/- {nstdp_r.std(ddof=1) if len(nstdp_r)>1 else 0:.4f}  '
          f'(d={d_nstdp:+.2f}, p={p_nstdp:.3g})')

    # Assess each question
    collapse_thresh = 0.5 * full_r.mean()
    q1_frozen  = froz_r.mean()  < full_r.mean()   # schema blocked -> ret drops?
    q1_zero    = zero_r.mean()  < full_r.mean()
    q2_frozen  = froz_r.mean()  < collapse_thresh  # collapse?
    q2_zero    = zero_r.mean()  < collapse_thresh
    q3_wuc_up  = (vec(data,'WCC_FROZEN','Wuc').mean() >
                  vec(data,'FULL','Wuc').mean())      # Wuc compensates?
    q3_wuu_up  = (vec(data,'WCC_FROZEN','Wuu').mean() >
                  vec(data,'FULL','Wuu').mean())
    q4_delta   = full_r.mean() - froz_r.mean()       # magnitude of loss

    print(f'\n  Q1 (Wcc prevented -> retention drops?): '
          f'FROZEN={q1_frozen}, CLAMPED={q1_zero}')
    print(f'  Q2 (Wcc prevented -> retention collapses <50% FULL?): '
          f'FROZEN={q2_frozen}, CLAMPED={q2_zero}')
    print(f'  Q3 (Wuc/Wuu compensate? Wuc up={q3_wuc_up}, Wuu up={q3_wuu_up})')
    print(f'  Q4 (retention lost when Wcc never forms): {q4_delta:+.4f}')
    print(f'  Q5 (Task 5 negative result generalized to formation time?): '
          f'{"NO — collapse" if q2_zero else "YES — small/n.s. drop"}')

    # Decision
    sig_drop = p_zero < 0.05 and q1_zero
    large_drop = q2_zero
    if large_drop and sig_drop:
        verdict, vtxt = 'A', 'Wcc required for formation — retention collapses without it'
    elif sig_drop and not large_drop:
        verdict, vtxt = 'B', 'Wcc contributes to formation — partial drop, not collapse'
    elif not q1_frozen and not q1_zero:
        verdict, vtxt = 'C', 'Wcc largely irrelevant to formation — no retention drop'
    else:
        verdict, vtxt = 'D', 'Wcc only reflects other processes — manipulation inconclusive'

    print(f'\n  FINAL VERDICT: {verdict}) {vtxt}')

    # CSV
    csv_path = os.path.join(OUT_DIR, 'task55_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'condition', 'Wcc', 'Wuc', 'Wuu', 'S1',
                    'retention_mean', 'retrieval_mean', 'replay_events'])
        for s in SEEDS:
            if s not in data:
                continue
            for cond in CONDS:
                if cond not in data[s]['conditions']:
                    continue
                c = data[s]['conditions'][cond]
                w.writerow([s, cond, c['Wcc'], c['Wuc'], c['Wuu'], c['S1'],
                             c['retention_mean'], c['retrieval_mean'], c['replay_events']])
    print(f'  CSV: {csv_path}')

    # Report
    rp = os.path.join(OUT_DIR, 'TASK55_REPORT.md')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('# TASK 5.5 -- FORMATION-TIME CAUSAL TEST OF Wcc\n\n')
        f.write('## Background\n\n')
        f.write('Task 5 showed that **post-hoc** Wcc destruction causes only a small, '
                'non-significant retention drop, ruling out Wcc as the primary recall substrate.  \n')
        f.write('Task 5.5 tests whether Wcc must **grow during learning** to build '
                'replay-protected memories.  Each condition is a fully independent '
                'training run from the same seed.\n\n')
        f.write('## Conditions\n\n')
        f.write('| Condition | Intervention during training |\n|---|---|\n')
        f.write('| FULL | Standard training, no intervention |\n')
        f.write('| WCC_FROZEN | Core-core block restored to init values after every STDP step |\n')
        f.write('| WCC_CLAMPED_ZERO | Core-core block zeroed after every STDP step |\n')
        f.write('| WCC_NO_STDP | plastic_mask zeros out core-core pairs |\n\n')
        f.write('## Table 1: Per-seed retention\n\n')
        f.write('| Seed | Condition | Wcc | Wuc | Wuu | S1 | Retention | Retrieval |\n'
                '|---|---|---|---|---|---|---|---|\n')
        for s in SEEDS:
            if s not in data:
                continue
            for cond in CONDS:
                if cond not in data[s]['conditions']:
                    continue
                c = data[s]['conditions'][cond]
                f.write(f'| {s} | {cond} | {c["Wcc"]:.4f} | {c["Wuc"]:.4f} | '
                        f'{c["Wuu"]:.4f} | {c["S1"]:.4f} | '
                        f'{c["retention_mean"]:.4f} | {c["retrieval_mean"]:.4f} |\n')
        f.write('\n## Table 2: Condition means\n\n')
        f.write('| Condition | Wcc | Wuc | Wuu | Retention | Retrieval |\n'
                '|---|---|---|---|---|---|\n')
        for cond in CONDS:
            wcc = vec(data, cond, 'Wcc')
            wuc = vec(data, cond, 'Wuc')
            wuu = vec(data, cond, 'Wuu')
            ret = vec(data, cond, 'retention_mean')
            rtr = vec(data, cond, 'retrieval_mean')
            f.write(f'| {cond} | {wcc.mean():.4f} | {wuc.mean():.4f} | {wuu.mean():.4f} '
                    f'| {ret.mean():.4f} | {rtr.mean():.4f} |\n')
        f.write('\n## Table 3: Statistics (FULL vs each intervention)\n\n')
        f.write('| Comparison | delta retention | Cohen d | t | p | sig |\n'
                '|---|---|---|---|---|---|\n')
        for cond, t, p, d in [('WCC_FROZEN', t_froz, p_froz, d_froz),
                               ('WCC_CLAMPED_ZERO', t_zero, p_zero, d_zero),
                               ('WCC_NO_STDP', t_nstdp, p_nstdp, d_nstdp)]:
            cv = vec(data, cond, 'retention_mean')
            dlt = full_r.mean() - cv.mean()
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            f.write(f'| FULL vs {cond} | {dlt:+.4f} | {d:+.2f} | {t:+.2f} | {p:.3g} | {sig} |\n')
        f.write('\n## Answers\n\n')
        f.write(f'- **Q1** Can schema form when Wcc growth is prevented? '
                f'{"Partially — retention drops" if q1_frozen or q1_zero else "Yes — no change"}\n')
        f.write(f'- **Q2** Can replay still protect retention without Wcc formation? '
                f'{"No — collapses" if q2_zero else "Yes — substantial retention preserved"}\n')
        f.write(f'- **Q3** Do Wuc/Wuu compensate? Wuc up={q3_wuc_up}, Wuu up={q3_wuu_up}\n')
        f.write(f'- **Q4** Retention lost when Wcc never forms: {q4_delta:+.4f}\n')
        f.write(f'- **Q5** Task 5 negative result (post-hoc) still holds at formation time? '
                f'{"NO — collapse" if q2_zero else "YES — small drop only"}\n')
        f.write(f'\n## Final Verdict: {verdict}) {vtxt}\n\n')
        f.write('## Figures\n\n')
        for fn, cap in [
            ('fig1_retention', 'Retention by condition'),
            ('fig2_wcc',       'Final Wcc by condition'),
            ('fig3_weight_decomposition', 'Weight decomposition (Wcc, Wuc, Wuu)'),
            ('fig4_effect_sizes', 'Effect sizes (Cohen d)'),
        ]:
            f.write(f'- `figures/{fn}.png` — {cap}\n')
    print(f'  Report: {rp}')
    return verdict


if __name__ == '__main__':
    data = load()
    print(f'Loaded {len(data)} seeds')
    per_seed_table(data)
    means_table(data)
    stats_table(data)
    print('\nGenerating figures...')
    figures(data)
    verdict_and_report(data)
    print('\nTASK 5.5 COMPLETE.')
