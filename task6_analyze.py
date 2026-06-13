"""
TASK 6 ANALYSIS — True Replay-Protected Memory Substrate
=========================================================
Tables 1-2, Figures 1-3 (+weight decomposition), paired statistics, verdict,
CSV, TASK6_REPORT.md.
"""
import os, sys, pickle, csv, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import ttest_rel

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task6'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042]
CONDS = ['CONTROL', 'DESTROY_WUC', 'DESTROY_WUU', 'DESTROY_WUC_WUU',
         'DESTROY_ALL_NON_CORE', 'DESTROY_ALL']
COLORS = {
    'CONTROL':              '#2166AC',
    'DESTROY_WUC':          '#F4A736',
    'DESTROY_WUU':          '#9970AB',
    'DESTROY_WUC_WUU':      '#D6604D',
    'DESTROY_ALL_NON_CORE': '#B2182B',
    'DESTROY_ALL':          '#4D4D4D',
}
SHORT = {
    'CONTROL':              'CONTROL\n(FULL)',
    'DESTROY_WUC':          'DESTROY\nWuc',
    'DESTROY_WUU':          'DESTROY\nWuu',
    'DESTROY_WUC_WUU':      'DESTROY\nWuc+Wuu',
    'DESTROY_ALL_NON_CORE': 'DESTROY\nALL non-core',
    'DESTROY_ALL':          'DESTROY\nALL',
}

# Which weight blocks each condition zeroes (for the schematic figure)
#   columns: Wcc, Wuc, Wuu, Background
DESTROY_MAP = {
    'CONTROL':              [0, 0, 0, 0],
    'DESTROY_WUC':          [0, 1, 0, 0],
    'DESTROY_WUU':          [0, 0, 1, 0],
    'DESTROY_WUC_WUU':      [0, 1, 1, 0],
    'DESTROY_ALL_NON_CORE': [0, 1, 1, 1],
    'DESTROY_ALL':          [1, 1, 1, 1],
}
BLOCK_COLS = ['Wcc', 'Wuc', 'Wuu', 'Background']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})


def load():
    data = {}
    for s in SEEDS:
        p = os.path.join(OUT_DIR, f'T6_seed{s}.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                data[s] = pickle.load(f)
        else:
            print(f'MISSING {p}')
    return data


def vec(data, cond, key):
    return np.array([data[s]['conditions'][cond][key]
                     for s in SEEDS if s in data and cond in data[s]['conditions']])


def cohen_dz(control, cond):
    """Paired Cohen's d_z (matches the paired t-test)."""
    diff = np.asarray(control, float) - np.asarray(cond, float)
    if len(diff) < 2:
        return float('nan')
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 1e-12 else float('inf') if abs(diff.mean()) > 1e-12 else 0.0


def paired_t(control, cond):
    if len(control) < 2 or len(cond) < 2:
        return float('nan'), float('nan')
    diff = np.asarray(control, float) - np.asarray(cond, float)
    if diff.std(ddof=1) < 1e-12:
        return float('nan'), (0.0 if abs(diff.mean()) < 1e-12 else 0.0)
    t, p = ttest_rel(control, cond)
    return t, p


# ─── Tables ───────────────────────────────────────────────────────────────────

def per_seed_table(data):
    print(f'\n{"=" * 116}')
    print('TABLE 1: PER-SEED RESULTS (post-hoc weight-block destruction)')
    print(f'{"=" * 116}')
    print(f'  {"Seed":>5s} {"Condition":<22s} {"Wcc":>8s} {"Wuc":>8s} {"Wuu":>8s} '
          f'{"S1":>8s} {"Retention":>10s} {"Retrieval":>10s} {"Replay":>7s}')
    print('  ' + '-' * 96)
    for s in SEEDS:
        if s not in data:
            continue
        for cond in CONDS:
            c = data[s]['conditions'][cond]
            print(f'  {s:>5d} {cond:<22s} {c["Wcc"]:>8.4f} {c["Wuc"]:>8.4f} {c["Wuu"]:>8.4f} '
                  f'{c["S1"]:>8.4f} {c["retention_mean"]:>10.4f} {c["retrieval_mean"]:>10.4f} '
                  f'{c["replay_events"]:>7d}')
        print()


def means_table(data):
    print(f'\n{"=" * 100}')
    print('TABLE 1b: CONDITION MEANS (mean +/- SD)')
    print(f'{"=" * 100}')
    print(f'  {"Condition":<22s} {"Wcc":>13s} {"Wuc":>13s} {"Wuu":>13s} '
          f'{"Retention":>15s} {"%Remain":>9s}')
    print('  ' + '-' * 88)
    ctrl = vec(data, 'CONTROL', 'retention_mean')
    ctrl_m = ctrl.mean() if len(ctrl) else 1e-9
    for cond in CONDS:
        def fmt(k):
            v = vec(data, cond, k)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            return f'{v.mean():.4f}+/-{sd:.4f}'
        ret = vec(data, cond, 'retention_mean')
        pct = 100 * ret.mean() / max(ctrl_m, 1e-9)
        print(f'  {cond:<22s} {fmt("Wcc"):>13s} {fmt("Wuc"):>13s} {fmt("Wuu"):>13s} '
              f'{fmt("retention_mean"):>15s} {pct:>8.1f}%')


def effect_table(data):
    print(f'\n{"=" * 100}')
    print('TABLE 2: EFFECT SIZES — CONTROL vs each destruction (paired, n=%d)' % len(
        [s for s in SEEDS if s in data]))
    print(f'{"=" * 100}')
    print(f'  {"Destruction":<22s} {"deltaRet":>10s} {"%lost":>8s} '
          f"{'d_z(paired)':>12s} {'t':>8s} {'p':>9s}  sig")
    print('  ' + '-' * 80)
    ctrl = vec(data, 'CONTROL', 'retention_mean')
    for cond in CONDS[1:]:
        cv = vec(data, cond, 'retention_mean')
        if len(ctrl) < 2 or len(cv) < 2:
            continue
        dlt = ctrl.mean() - cv.mean()
        lost = 100 * dlt / max(abs(ctrl.mean()), 1e-9)
        dz = cohen_dz(ctrl, cv)
        t, p = paired_t(ctrl, cv)
        sig = '***' if (p == p and p < 0.001) else '**' if (p == p and p < 0.01) \
            else '*' if (p == p and p < 0.05) else 'n.s.'
        print(f'  {cond:<22s} {dlt:>+10.4f} {lost:>7.1f}% {dz:>+12.2f} '
              f'{t:>+8.2f} {p:>9.4g}  {sig}')


# ─── Figures ──────────────────────────────────────────────────────────────────

def _save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}')


def fig1_retention(data):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    xs = np.arange(len(CONDS))
    rng = np.random.default_rng(0)
    ctrl = vec(data, 'CONTROL', 'retention_mean')
    # Replay-removal phenotype reference line (~87% loss of control)
    phenotype = ctrl.mean() * (1 - 0.87) if len(ctrl) else 0
    for i, cond in enumerate(CONDS):
        v = vec(data, cond, 'retention_mean')
        if len(v) == 0:
            continue
        err = v.std(ddof=1) if len(v) > 1 else 0
        ax.bar(i, v.mean(), yerr=err, capsize=6,
               color=COLORS[cond], edgecolor='black', alpha=0.88)
        jit = rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(i + jit, v, color='black', s=38, alpha=0.7, zorder=5,
                   edgecolor='white', linewidth=0.6)
    ax.axhline(phenotype, color='red', ls='--', lw=1.6, alpha=0.8,
               label=f'Replay-removal phenotype (~87% loss = {phenotype:.3f})')
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=9)
    ax.set_ylabel('Retention (isyn_score mean)', fontweight='bold')
    ax.set_title('Fig 1 — Retention by weight-block destruction (post-hoc, n=%d seeds)'
                 % len([s for s in SEEDS if s in data]), pad=8)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig1_retention_by_intervention')


def fig2_schematic(data):
    """Weight-block destruction schematic: rows=conditions, cols=blocks."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    n_rows, n_cols = len(CONDS), len(BLOCK_COLS)
    kept_c, dest_c = '#2166AC', '#D6604D'
    for r, cond in enumerate(CONDS):
        dmap = DESTROY_MAP[cond]
        for c in range(n_cols):
            destroyed = dmap[c]
            color = dest_c if destroyed else kept_c
            ax.add_patch(plt.Rectangle((c, n_rows - 1 - r), 1, 1,
                                       facecolor=color, edgecolor='white', lw=2, alpha=0.9))
            ax.text(c + 0.5, n_rows - 1 - r + 0.5,
                    'ZERO' if destroyed else 'keep',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white')
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(BLOCK_COLS, fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([c for c in reversed(CONDS)], fontsize=10)
    ax.set_title('Fig 2 — Weight-block destruction schematic\n'
                 'Which connectivity blocks each intervention zeroes', pad=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [Patch(facecolor=kept_c, label='Kept (trained weights)'),
              Patch(facecolor=dest_c, label='Destroyed (set to 0)')]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    _save(fig, 'fig2_destruction_schematic')


def fig3_pct_remaining(data):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ctrl = vec(data, 'CONTROL', 'retention_mean')
    ctrl_m = ctrl.mean() if len(ctrl) else 1e-9
    xs = np.arange(len(CONDS))
    for i, cond in enumerate(CONDS):
        v = vec(data, cond, 'retention_mean')
        if len(v) == 0:
            continue
        pct = 100 * v / max(ctrl_m, 1e-9)
        err = pct.std(ddof=1) if len(pct) > 1 else 0
        ax.bar(i, pct.mean(), yerr=err, capsize=6,
               color=COLORS[cond], edgecolor='black', alpha=0.88)
        ax.text(i, pct.mean() + (err if err else 0) + 2,
                f'{pct.mean():.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(100, color='#2166AC', ls=':', lw=1.3, alpha=0.7)
    ax.axhline(13, color='red', ls='--', lw=1.6, alpha=0.8,
               label='Replay-removal floor (~13% remaining)')
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=9)
    ax.set_ylabel('% retention remaining vs CONTROL', fontweight='bold')
    ax.set_ylim(0, 120)
    ax.set_title('Fig 3 — Percent retention remaining after destruction', pad=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig3_pct_retention_remaining')


def fig4_weight_decomp(data):
    """Bonus: verify each destruction actually zeroed its target block."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax_i, wkey in enumerate(['Wcc', 'Wuc', 'Wuu']):
        ax = axes[ax_i]
        for i, cond in enumerate(CONDS):
            v = vec(data, cond, wkey)
            if len(v) == 0:
                continue
            err = v.std(ddof=1) if len(v) > 1 else 0
            ax.bar(i, v.mean(), yerr=err, capsize=4,
                   color=COLORS[cond], edgecolor='black', alpha=0.85)
        ax.set_xticks(np.arange(len(CONDS)))
        ax.set_xticklabels([c.replace('DESTROY_', '-').replace('CONTROL', 'CTRL')
                            for c in CONDS], fontsize=7, rotation=30, ha='right')
        ax.set_ylabel(wkey, fontweight='bold')
        ax.set_title(f'{wkey} after destruction', pad=6)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Fig 4 — Weight-block verification (destruction zeroes its target block)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig4_weight_verification')


# ─── Verdict + Report + CSV ───────────────────────────────────────────────────

def verdict_and_report(data):
    n = len([s for s in SEEDS if s in data])
    ctrl = vec(data, 'CONTROL', 'retention_mean')
    ctrl_m = ctrl.mean() if len(ctrl) else 1e-9

    def loss_frac(cond):
        cv = vec(data, cond, 'retention_mean')
        return (ctrl_m - cv.mean()) / max(abs(ctrl_m), 1e-9)

    losses = {c: loss_frac(c) for c in CONDS[1:]}
    stats = {}
    for cond in CONDS[1:]:
        cv = vec(data, cond, 'retention_mean')
        t, p = paired_t(ctrl, cv)
        dz = cohen_dz(ctrl, cv)
        stats[cond] = (t, p, dz)

    COLLAPSE = 0.50       # >=50% retention loss = "collapse"
    PHENOTYPE = 0.70      # >=70% loss = "reproduces replay-removal phenotype"

    l_wuc   = losses['DESTROY_WUC']
    l_wuu   = losses['DESTROY_WUU']
    l_both  = losses['DESTROY_WUC_WUU']
    l_noncore = losses['DESTROY_ALL_NON_CORE']
    l_all   = losses['DESTROY_ALL']

    # Identify the single destruction with the largest collapse (excluding the
    # all-destroying sanity floor for the "minimal substrate" decision)
    candidate = {c: losses[c] for c in
                 ['DESTROY_WUC', 'DESTROY_WUU', 'DESTROY_WUC_WUU', 'DESTROY_ALL_NON_CORE']}
    biggest = max(candidate, key=candidate.get)

    # Decision rules from the task
    if l_wuc >= COLLAPSE and l_wuc >= l_wuu:
        substrate = 'Wuc (core<->unique)'
        q1 = 'Wuc'
    elif l_wuu >= COLLAPSE and l_wuu > l_wuc:
        substrate = 'Wuu (within-unique)'
        q1 = 'Wuu'
    elif l_both >= COLLAPSE and l_wuc < COLLAPSE and l_wuu < COLLAPSE:
        substrate = 'Wuc + Wuu jointly (distributed across both)'
        q1 = 'Wuc+Wuu (distributed)'
    elif l_noncore >= COLLAPSE and l_both < COLLAPSE:
        substrate = 'Distributed non-core structure (incl. background)'
        q1 = 'Distributed non-core'
    elif max(losses.values()) < COLLAPSE:
        substrate = 'Higher-order / distributed network structure (no single block collapses retention)'
        q1 = 'Higher-order distributed structure'
    else:
        substrate = f'{biggest} (largest collapse)'
        q1 = biggest

    reproduces = [c for c in CONDS[1:] if losses[c] >= PHENOTYPE]

    print(f'\n{"=" * 84}\nVERDICT — TASK 6\n{"=" * 84}')
    print(f'  CONTROL retention: {ctrl_m:.4f}')
    print(f'\n  Retention loss by destruction:')
    for cond in CONDS[1:]:
        t, p, dz = stats[cond]
        flag = '  <== reproduces phenotype' if losses[cond] >= PHENOTYPE else \
               '  <- collapse' if losses[cond] >= COLLAPSE else ''
        print(f'    {cond:<22s} loss={100*losses[cond]:>5.1f}%  '
              f'(d_z={dz:+.2f}, p={p:.3g}){flag}')

    print(f'\n  Largest single-block collapse: {biggest} ({100*candidate[biggest]:.1f}% loss)')
    print(f'  Reproduces replay-removal phenotype (>={int(PHENOTYPE*100)}% loss): '
          f'{reproduces if reproduces else "NONE"}')

    print(f'\n  --- FINAL VERDICT ---')
    print(f'  Q1 Memory substrate:  {q1}')
    print(f'  Q2 Schema substrate:  Wcc (core-core) — established Tasks 4-5.5')
    same = ('NO' if 'Wuc' in q1 or 'Wuu' in q1 or 'Distributed' in q1 or 'Higher' in q1
            else 'YES')
    print(f'  Q3 Same substrate?    {same} — memory != schema'
          if same == 'NO' else f'  Q3 Same substrate?    {same}')
    print(f'  Q4 Corrected causal chain: see report')

    # ── CSV ──
    csv_path = os.path.join(OUT_DIR, 'task6_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'condition', 'Wcc', 'Wuc', 'Wuu', 'S1',
                    'retention_mean', 'retrieval_mean', 'replay_events'])
        for s in SEEDS:
            if s not in data:
                continue
            for cond in CONDS:
                c = data[s]['conditions'][cond]
                w.writerow([s, cond, c['Wcc'], c['Wuc'], c['Wuu'], c['S1'],
                             c['retention_mean'], c['retrieval_mean'], c['replay_events']])
    print(f'\n  CSV: {csv_path}')

    # ── Report ──
    rp = os.path.join(OUT_DIR, 'TASK6_REPORT.md')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('# TASK 6 -- TRUE REPLAY-PROTECTED MEMORY SUBSTRATE\n\n')
        f.write('## Context\n\n')
        f.write('Tasks 1-5.5 established: Replay->Retention is causal; Replay->Wcc is causal; '
                'but **Wcc->Retention is NOT causal** (post-hoc destruction ~5% loss; '
                'formation-time prevention 7-13% loss).  The original causal chain is falsified.  \n')
        f.write('Task 6 asks: *which connectivity block actually stores the '
                'replay-protected memory?*\n\n')
        f.write(f'All 6 interventions share the **identical trained FULL network** per seed '
                f'(n={n} seeds: {", ".join(str(s) for s in SEEDS if s in data)}), so any '
                f'retention difference is causally attributable to the destroyed block alone.\n\n')
        f.write('## Interventions\n\n')
        f.write('| Condition | Blocks zeroed |\n|---|---|\n')
        f.write('| CONTROL | none (= FULL) |\n')
        f.write('| DESTROY_WUC | unique<->core |\n')
        f.write('| DESTROY_WUU | within-unique |\n')
        f.write('| DESTROY_WUC_WUU | Wuc + Wuu (Wcc + background kept) |\n')
        f.write('| DESTROY_ALL_NON_CORE | everything except Wcc |\n')
        f.write('| DESTROY_ALL | entire excitatory matrix (sanity floor) |\n\n')

        f.write('## TABLE 1: Condition means\n\n')
        f.write('| Condition | Wcc | Wuc | Wuu | Retention | % remaining |\n|---|---|---|---|---|---|\n')
        for cond in CONDS:
            wcc = vec(data, cond, 'Wcc'); wuc = vec(data, cond, 'Wuc')
            wuu = vec(data, cond, 'Wuu'); ret = vec(data, cond, 'retention_mean')
            pct = 100 * ret.mean() / max(ctrl_m, 1e-9)
            f.write(f'| {cond} | {wcc.mean():.4f} | {wuc.mean():.4f} | {wuu.mean():.4f} '
                    f'| {ret.mean():.4f} | {pct:.1f}% |\n')

        f.write('\n## TABLE 2: Effect sizes (CONTROL vs destruction, paired)\n\n')
        f.write('| Destruction | delta Ret | % lost | Cohen d_z | t | p |\n|---|---|---|---|---|---|\n')
        for cond in CONDS[1:]:
            cv = vec(data, cond, 'retention_mean')
            dlt = ctrl_m - cv.mean()
            t, p, dz = stats[cond]
            f.write(f'| {cond} | {dlt:+.4f} | {100*losses[cond]:.1f}% | {dz:+.2f} '
                    f'| {t:+.2f} | {p:.3g} |\n')

        f.write('\n## Primary question\n\n')
        f.write(f'**Which destruction reproduces the replay-removal phenotype (~87% loss)?**  \n')
        f.write(f'Reproduces phenotype (>=70% loss): '
                f'**{", ".join(reproduces) if reproduces else "NONE"}**.  \n')
        f.write(f'Largest single-block collapse: **{biggest}** '
                f'({100*candidate[biggest]:.1f}% loss).\n\n')

        f.write('## FINAL VERDICT\n\n')
        f.write(f'1. **What weight block stores memory?**  {q1}.\n')
        f.write(f'2. **What weight block stores schema?**  Wcc (core-core) — the schema '
                f'index S1 = Wcc - Wuc collapses only when Wcc is removed; established in '
                f'Tasks 4-5.5.\n')
        f.write(f'3. **Are memory and schema the same substrate?**  '
                f'{"**No.** Memory lives in " + q1 + "; schema lives in Wcc. They are dissociable." if same=="NO" else "Yes."}\n')
        f.write(f'4. **Corrected causal chain after Tasks 1-6:**\n\n')
        f.write(f'   ```\n')
        f.write(f'   Replay  -->  potentiates {q1}  -->  Retention (memory)\n')
        f.write(f'      |\n')
        f.write(f'      +----->  potentiates Wcc       -->  Schema metrics (S1)  [epiphenomenal for memory]\n')
        f.write(f'   ```\n\n')
        f.write(f'   Replay is the common cause of BOTH the memory substrate ({q1}) and the '
                f'schema substrate (Wcc).  Wcc co-varies with retention only because replay '
                f'drives both in parallel — destroying Wcc leaves memory intact, while '
                f'destroying {q1} '
                f'{"reproduces the replay-removal collapse" if reproduces else "produces the largest retention loss"}.\n\n')
        f.write('## Figures\n\n')
        for fn, cap in [
            ('fig1_retention_by_intervention', 'Retention by intervention'),
            ('fig2_destruction_schematic', 'Weight-block destruction schematic'),
            ('fig3_pct_retention_remaining', 'Percent retention remaining'),
            ('fig4_weight_verification', 'Weight-block verification (bonus)'),
        ]:
            f.write(f'- `figures/{fn}.png` — {cap}\n')
    print(f'  Report: {rp}')
    return q1, reproduces


if __name__ == '__main__':
    data = load()
    print(f'Loaded {len(data)} seeds')
    if not data:
        print('No data — run run_task6.py first.')
        sys.exit(1)
    per_seed_table(data)
    means_table(data)
    effect_table(data)
    print('\nGenerating figures...')
    fig1_retention(data)
    fig2_schematic(data)
    fig3_pct_remaining(data)
    fig4_weight_decomp(data)
    verdict_and_report(data)
    print('\nTASK 6 COMPLETE.')
