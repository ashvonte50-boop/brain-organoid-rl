"""
TASK 2 MASTER RECOMPUTE — Replace RS with Wcc / S1
====================================================
Loads all Task 2 (40 PKLs) + Task 2.5 (30 PKLs).
Recomputes Wcc, Wuc, S1 from W_final matrices.
Generates publication-ready tables + Master Summary figure.
"""
import os, sys, pickle, warnings, textwrap
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind

T2_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task2'
T25_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task25'
FIG_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task2_master'
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 150,
})

# ── Load ──────────────────────────────────────────────────────────────────
def load_all():
    rows = []
    for cname in ['FULL', 'FULL_NO_MB', 'NO_REPLAY', 'NO_REPLAY_NO_MB']:
        for s in SEEDS:
            p = os.path.join(T2_DIR, f'T2_{cname}_seed{s}.pkl')
            if not os.path.exists(p): continue
            with open(p, 'rb') as f: r = pickle.load(f)
            W = r['W_final']; core = np.asarray(r['core_mask'])
            assemblies = [np.asarray(a) for a in r['assemblies']]
            ne = W.shape[0]
            Wcc = float(W[np.ix_(core, core)].mean())
            uc = [W[np.ix_(np.array([i for i in a if i not in core and i<ne]), core)].mean()
                  for a in assemblies if len([i for i in a if i not in core and i<ne])>0]
            Wuc = float(np.mean(uc)) if uc else 1e-9
            rows.append({
                'condition': cname, 'seed': s, 'source': 'T2',
                'RS': r['real_schema'], 'retention': r['retention_mean'],
                'Wcc': Wcc, 'Wuc': Wuc, 'S1': Wcc-Wuc,
                'replay_events': r['replay_events'],
                'ret_A': r.get('retention_A', r['final_scores'][0]),
                'ret_B': r.get('retention_B', r['final_scores'][1]),
                'ret_C': r.get('retention_C', r['final_scores'][2]),
                'ret_D': r.get('retention_D', r['final_scores'][3]),
            })
    for cname in ['FULL', 'NO_CORE_STIM', 'HALF_STIM']:
        for s in SEEDS:
            p = os.path.join(T25_DIR, f'T25_{cname}_seed{s}.pkl')
            if not os.path.exists(p): continue
            with open(p, 'rb') as f: r = pickle.load(f)
            rows.append({
                'condition': cname, 'seed': s, 'source': 'T25',
                'RS': r['real_schema'], 'retention': r['retention_mean'],
                'Wcc': r['W_core_core_mean'], 'Wuc': r['W_unique_to_core_mean'],
                'S1': r['W_core_core_mean'] - r['W_unique_to_core_mean'],
                'replay_events': r['replay_events'],
                'ret_A': r['retention_A'], 'ret_B': r['retention_B'],
                'ret_C': r['retention_C'], 'ret_D': r['retention_D'],
            })
    return rows

def vec(rows, cond, key, src=None):
    if src:
        return np.array([r[key] for r in rows if r['condition']==cond and r['source']==src])
    if cond in ('FULL','FULL_NO_MB','NO_REPLAY','NO_REPLAY_NO_MB'):
        return np.array([r[key] for r in rows if r['condition']==cond and r['source']=='T2'])
    return np.array([r[key] for r in rows if r['condition']==cond and r['source']=='T25'])

def d(a, b):
    a,b = np.asarray(a,float), np.asarray(b,float)
    if len(a)<2 or len(b)<2: return float('nan')
    p = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/p if p else float('nan')


# ── Tables ────────────────────────────────────────────────────────────────
CONDITIONS = ['FULL', 'FULL_NO_MB', 'NO_REPLAY', 'NO_CORE_STIM', 'HALF_STIM']
COND_LABELS = {
    'FULL': 'FULL (baseline)',
    'FULL_NO_MB': 'FULL - MB',
    'NO_REPLAY': 'NO REPLAY',
    'NO_CORE_STIM': 'NO CORE STIM',
    'HALF_STIM': 'HALF STIM',
}
COND_COLORS = {
    'FULL': '#2166AC', 'FULL_NO_MB': '#5AAE61', 'NO_REPLAY': '#D6604D',
    'NO_CORE_STIM': '#E8601C', 'HALF_STIM': '#F4A736',
}


def print_per_seed_table(rows):
    print(f'\n{"="*120}')
    print('TABLE 1: PER-SEED RESULTS (all conditions, all metrics)')
    print(f'{"="*120}')
    print(f'{"Condition":<16s} {"Seed":>5s} {"Wcc":>8s} {"Wuc":>8s} {"S1":>8s} '
          f'{"RS":>8s} {"Ret":>8s} {"RetA":>7s} {"RetB":>7s} {"RetC":>7s} {"RetD":>7s} {"rep":>4s}')
    print('-'*120)
    for cond in CONDITIONS:
        for s in SEEDS:
            matches = [r for r in rows if r['condition']==cond and r['seed']==s
                       and ((cond in ('FULL','FULL_NO_MB','NO_REPLAY','NO_REPLAY_NO_MB') and r['source']=='T2')
                            or (cond in ('NO_CORE_STIM','HALF_STIM') and r['source']=='T25')
                            or (cond=='FULL' and r['source']=='T2'))]
            if not matches: continue
            r = matches[0]
            print(f'{cond:<16s} {s:>5d} {r["Wcc"]:>8.4f} {r["Wuc"]:>8.4f} {r["S1"]:>8.4f} '
                  f'{r["RS"]:>8.4f} {r["retention"]:>8.4f} '
                  f'{r["ret_A"]:>7.3f} {r["ret_B"]:>7.3f} {r["ret_C"]:>7.3f} {r["ret_D"]:>7.3f} '
                  f'{r["replay_events"]:>4d}')
        print()


def print_summary_table(rows):
    print(f'\n{"="*110}')
    print('TABLE 2: SUMMARY STATISTICS (n=10 per condition)')
    print(f'{"="*110}')
    print(f'{"Condition":<16s} {"Wcc":>12s} {"S1":>12s} {"RS":>12s} '
          f'{"Retention":>12s} {"rep":>6s}')
    print('-'*110)
    for cond in CONDITIONS:
        wcc = vec(rows, cond, 'Wcc')
        s1  = vec(rows, cond, 'S1')
        rs  = vec(rows, cond, 'RS')
        ret = vec(rows, cond, 'retention')
        rep = vec(rows, cond, 'replay_events')
        if len(wcc) == 0: continue
        def fmt(v): return f'{v.mean():.4f}+/-{v.std(ddof=1):.4f}' if len(v)>1 else f'{v.mean():.4f}'
        print(f'{cond:<16s} {fmt(wcc):>12s} {fmt(s1):>12s} {fmt(rs):>12s} '
              f'{fmt(ret):>12s} {rep.mean():>6.1f}')


def print_pct_change_table(rows):
    print(f'\n{"="*90}')
    print('TABLE 3: PERCENTAGE CHANGE FROM FULL')
    print(f'{"="*90}')
    print(f'{"Condition":<16s} {"Wcc":>10s} {"S1":>10s} {"RS":>10s} {"Retention":>10s}')
    print('-'*90)
    full_wcc = vec(rows,'FULL','Wcc').mean()
    full_s1  = vec(rows,'FULL','S1').mean()
    full_rs  = vec(rows,'FULL','RS').mean()
    full_ret = vec(rows,'FULL','retention').mean()
    for cond in CONDITIONS:
        wcc = vec(rows,cond,'Wcc').mean()
        s1  = vec(rows,cond,'S1').mean()
        rs  = vec(rows,cond,'RS').mean()
        ret = vec(rows,cond,'retention').mean()
        def pct(v, ref): return f'{100*(v-ref)/abs(ref):+.0f}%' if ref else 'n/a'
        print(f'{cond:<16s} {pct(wcc,full_wcc):>10s} {pct(s1,full_s1):>10s} '
              f'{pct(rs,full_rs):>10s} {pct(ret,full_ret):>10s}')


def print_effect_size_table(rows):
    print(f'\n{"="*100}')
    print('TABLE 4: EFFECT SIZES (Cohen\'s d, FULL vs each ablation)')
    print(f'{"="*100}')
    ablations = ['NO_REPLAY', 'NO_CORE_STIM', 'HALF_STIM', 'FULL_NO_MB']
    print(f'{"Metric":<12s}', end='')
    for ab in ablations: print(f'  {"vs "+ab:>24s}', end='')
    print()
    print('-'*100)
    for metric in ['Wcc', 'S1', 'RS', 'retention']:
        print(f'{metric:<12s}', end='')
        for ab in ablations:
            va, vb = vec(rows,'FULL',metric), vec(rows,ab,metric)
            if len(va)<2 or len(vb)<2:
                print(f'  {"n/a":>24s}', end=''); continue
            dd = d(va, vb)
            t, p = ttest_ind(va, vb, equal_var=False)
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'  d={dd:>+6.2f} p={p:.0e} {sig:>4s}', end='')
        print()


def print_correlation_table(rows):
    print(f'\n{"="*90}')
    print('TABLE 5: CORRELATION WITH RETENTION (n=70 runs)')
    print(f'{"="*90}')
    all_ret = np.array([r['retention'] for r in rows])
    print(f'{"Metric":<20s} {"Pearson r":>10s} {"p":>12s} {"Spearman":>10s} {"p":>12s}')
    print('-'*90)
    for metric, label in [('Wcc','Wcc'), ('S1','S1 (Wcc-Wuc)'), ('RS','RS (old)')]:
        vals = np.array([r[metric] for r in rows])
        pr, pp = pearsonr(vals, all_ret)
        sr, sp = spearmanr(vals, all_ret)
        print(f'{label:<20s} {pr:>+10.4f} {pp:>12.2e} {sr:>+10.4f} {sp:>12.2e}')


# ── Master Summary Figure ─────────────────────────────────────────────────
def fig_master(rows):
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    conds_bar = ['FULL', 'NO_REPLAY', 'NO_CORE_STIM', 'HALF_STIM']

    def bar_panel(ax, metric, ylabel, title, show_legend=False):
        xs = np.arange(len(conds_bar))
        means, sds, pts_all = [], [], []
        for c in conds_bar:
            v = vec(rows, c, metric)
            means.append(v.mean()); sds.append(v.std(ddof=1) if len(v)>1 else 0); pts_all.append(v)
        ax.bar(xs, means, yerr=sds, capsize=5,
               color=[COND_COLORS[c] for c in conds_bar],
               edgecolor='black', linewidth=1.0, alpha=0.85)
        rng = np.random.default_rng(0)
        for x, pts in zip(xs, pts_all):
            jit = rng.uniform(-0.15, 0.15, size=len(pts))
            ax.scatter(x+jit, pts, color='black', s=20, alpha=0.55, zorder=5,
                       edgecolor='white', linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace('_','\n') for c in conds_bar], fontsize=9)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='grey', lw=0.5, ls=':')
        # Add effect sizes as annotations
        full_v = vec(rows, 'FULL', metric)
        for i, c in enumerate(conds_bar[1:], 1):
            cv = vec(rows, c, metric)
            dd = d(full_v, cv)
            _, p = ttest_ind(full_v, cv, equal_var=False)
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
            if sig:
                ypos = means[i] + sds[i] + 0.003 * max(means)
                ax.text(i, ypos, f'd={dd:+.1f}{sig}', ha='center', fontsize=8, fontweight='bold')

    # Panel A: Wcc
    ax1 = fig.add_subplot(gs[0, 0])
    bar_panel(ax1, 'Wcc', 'Wcc (core weight magnitude)', 'A. Schema Strength (Wcc)')

    # Panel B: S1
    ax2 = fig.add_subplot(gs[0, 1])
    bar_panel(ax2, 'S1', 'S1 = Wcc - Wuc', 'B. Schema Asymmetry (S1)')

    # Panel C: Retention
    ax3 = fig.add_subplot(gs[0, 2])
    bar_panel(ax3, 'retention', 'Retention (mean)', 'C. Memory Retention')

    # Panel D: RS (old, for comparison)
    ax4 = fig.add_subplot(gs[1, 0])
    bar_panel(ax4, 'RS', 'RS (old metric)', 'D. RS (scale-invariant ratio)\n[DEMOTED - shown for comparison]')

    # Panel E: Scatter — Wcc vs Retention
    ax5 = fig.add_subplot(gs[1, 1])
    all_wcc = np.array([r['Wcc'] for r in rows])
    all_ret = np.array([r['retention'] for r in rows])
    for c in set(r['condition'] for r in rows):
        mask = np.array([r['condition']==c for r in rows])
        ax5.scatter(all_wcc[mask], all_ret[mask], label=c.replace('_',' '), alpha=0.65, s=35,
                   color=COND_COLORS.get(c, '#888'), edgecolor='white', linewidth=0.5)
    pr, pp = pearsonr(all_wcc, all_ret)
    z = np.polyfit(all_wcc, all_ret, 1)
    xl = np.linspace(all_wcc.min(), all_wcc.max(), 100)
    ax5.plot(xl, np.polyval(z, xl), '--k', lw=1.5, alpha=0.5)
    ax5.set_xlabel('Wcc', fontweight='bold')
    ax5.set_ylabel('Retention', fontweight='bold')
    ax5.set_title(f'E. Wcc predicts Retention\n(r = {pr:+.3f}, p = {pp:.1e})')
    ax5.legend(fontsize=7, loc='upper left')
    ax5.grid(alpha=0.3)

    # Panel F: Scatter — RS vs Retention
    ax6 = fig.add_subplot(gs[1, 2])
    all_rs = np.array([r['RS'] for r in rows])
    for c in set(r['condition'] for r in rows):
        mask = np.array([r['condition']==c for r in rows])
        ax6.scatter(all_rs[mask], all_ret[mask], label=c.replace('_',' '), alpha=0.65, s=35,
                   color=COND_COLORS.get(c, '#888'), edgecolor='white', linewidth=0.5)
    pr2, pp2 = pearsonr(all_rs, all_ret)
    z2 = np.polyfit(all_rs, all_ret, 1)
    xl2 = np.linspace(all_rs.min(), all_rs.max(), 100)
    ax6.plot(xl2, np.polyval(z2, xl2), '--k', lw=1.5, alpha=0.5)
    ax6.set_xlabel('RS (old metric)', fontweight='bold')
    ax6.set_ylabel('Retention', fontweight='bold')
    ax6.set_title(f'F. RS does NOT predict Retention\n(r = {pr2:+.3f}, p = {pp2:.1e})')
    ax6.grid(alpha=0.3)

    fig.suptitle('Task 2 Master Summary: Schema Metrics Across All Conditions\n'
                 'Wcc and S1 track retention; RS is scale-invariant and insensitive to ablations',
                 fontsize=15, fontweight='bold', y=1.02)

    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'task2_master_summary.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nMaster figure saved: {FIG_DIR}/task2_master_summary.[png|pdf|svg]')


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('TASK 2 MASTER RECOMPUTE')
    rows = load_all()
    print(f'Loaded {len(rows)} runs')

    print_per_seed_table(rows)
    print_summary_table(rows)
    print_pct_change_table(rows)
    print_effect_size_table(rows)
    print_correlation_table(rows)

    print('\nGenerating master summary figure...')
    fig_master(rows)

    print('\nDONE.')
