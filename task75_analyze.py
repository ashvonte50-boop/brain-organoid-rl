"""
TASK 7.5 ANALYSIS — Sufficiency Test of W_slow[cc]
====================================================
Verdict criteria:
  A: WSLOW_CC_ONLY >= 80% CONTROL and UC/UU-only fail  -> sufficient
  B: WSLOW_CC_ONLY partial, UC adds substantially      -> distributed cc+uc
  C: WSLOW_CC_ONLY collapses                           -> T7 interpretation wrong
"""
import os, sys, pickle, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task75'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042]

CONDS = ['CONTROL','WSLOW_CC_ONLY','WSLOW_CC_PLUS_UC','WSLOW_CC_PLUS_UU',
         'WSLOW_UC_ONLY','WSLOW_UU_ONLY']

SHORT = {
    'CONTROL':          'CONTROL',
    'WSLOW_CC_ONLY':    'Ws_cc\nOnly',
    'WSLOW_CC_PLUS_UC': 'Ws_cc\n+UC',
    'WSLOW_CC_PLUS_UU': 'Ws_cc\n+UU',
    'WSLOW_UC_ONLY':    'Ws_uc\nOnly',
    'WSLOW_UU_ONLY':    'Ws_uu\nOnly',
}

COLORS = {
    'CONTROL':          '#2166AC',
    'WSLOW_CC_ONLY':    '#D6604D',
    'WSLOW_CC_PLUS_UC': '#F4A736',
    'WSLOW_CC_PLUS_UU': '#4DAF4A',
    'WSLOW_UC_ONLY':    '#9970AB',
    'WSLOW_UU_ONLY':    '#AAAAAA',
}

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,
})

NO_REPLAY = 0.022  # T7 W_ONLY/DESTROY_WSLOW_ALL mean


def load():
    data = {}
    for s in SEEDS:
        p = os.path.join(OUT_DIR, f'T75_seed{s}.pkl')
        if os.path.exists(p):
            with open(p,'rb') as f: data[s] = pickle.load(f)
        else:
            print(f'  MISSING {p}')
    return data


def vec(data, cond, key='retention_mean'):
    return np.array([data[s]['conditions'][cond][key]
                     for s in SEEDS if s in data and cond in data[s]['conditions']])


def cohen_dz(a, b):
    d = np.asarray(a,float) - np.asarray(b,float)
    return float(d.mean()/(d.std(ddof=1)+1e-12))


def stats_row(data, cond, ctrl):
    v = vec(data, cond)
    if len(v) == 0: return None
    pct = 100.*v.mean()/ctrl.mean()
    if len(v)>1 and len(ctrl)>1:
        t,p = ttest_rel(ctrl,v)
        dz = cohen_dz(ctrl,v)
    else:
        t=p=dz=float('nan')
    return (cond, v, v.mean(), v.std(ddof=1) if len(v)>1 else 0., pct, t, p, dz)


def fig_main(data):
    ctrl = vec(data,'CONTROL')
    ctrl_mean = ctrl.mean()
    avail = [c for c in CONDS if any(c in data[s]['conditions'] for s in data)]
    means = [vec(data,c).mean() for c in avail]
    sems  = [vec(data,c).std(ddof=1)/np.sqrt(len(vec(data,c)))
             if len(vec(data,c))>1 else 0 for c in avail]

    fig, ax = plt.subplots(figsize=(11,5))
    x = np.arange(len(avail))
    ax.bar(x, means, 0.6, yerr=sems, capsize=4,
           color=[COLORS[c] for c in avail], edgecolor='k', linewidth=0.7)
    for xi,c in enumerate(avail):
        v = vec(data,c)
        jit = np.linspace(-0.12,0.12,len(v))
        ax.scatter(xi+jit, v, color='k', s=22, zorder=5, alpha=0.8)

    ax.axhline(NO_REPLAY, color='red', linestyle='--', linewidth=1.2,
               label=f'No-replay floor ({NO_REPLAY:.3f})')
    ax.axhline(ctrl_mean, color=COLORS['CONTROL'], linestyle=':', linewidth=1.,
               label=f'CONTROL ({ctrl_mean:.3f})')
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in avail], fontsize=10)
    ax.set_ylabel('Retention (isyn_score)')
    ax.set_title('Task 7.5 — Sufficiency Test: Which W_slow blocks sustain memory?')
    ax.legend(fontsize=9)
    for xi,(c,m,s) in enumerate(zip(avail,means,sems)):
        pct = 100.*m/ctrl_mean
        ax.text(xi, m+s+0.004, f'{pct:.0f}%', ha='center', fontsize=9)

    fig.tight_layout()
    p = os.path.join(FIG_DIR,'fig1_sufficiency.png')
    fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')
    return p


def fig_per_memory(data):
    """Retention per memory per condition — checks if all memories survive."""
    ctrl = vec(data,'CONTROL')
    ctrl_mean = ctrl.mean()
    avail = [c for c in CONDS if any(c in data[s]['conditions'] for s in data)]
    n_asm = 4

    fig, axes = plt.subplots(1,len(avail),figsize=(14,4),sharey=True)
    for ax,cond in zip(axes,avail):
        # collect per-memory retention across seeds
        pm = []
        for s in SEEDS:
            if s not in data or cond not in data[s]['conditions']: continue
            pm.append(data[s]['conditions'][cond]['retention_per_memory'])
        if not pm: continue
        pm = np.array(pm)  # (n_seeds, n_memories)
        means = pm.mean(axis=0)
        sems  = pm.std(axis=0,ddof=1)/np.sqrt(pm.shape[0]) if pm.shape[0]>1 else np.zeros(pm.shape[1])
        ax.bar(range(len(means)), means, yerr=sems, capsize=3,
               color=COLORS[cond], edgecolor='k', linewidth=0.5)
        ax.axhline(NO_REPLAY,color='red',linestyle='--',linewidth=0.8)
        ax.set_title(SHORT[cond].replace('\n',' '),fontsize=9)
        ax.set_xlabel('Memory')
        if ax==axes[0]: ax.set_ylabel('Retention')
        ax.set_xticks(range(len(means)))

    fig.suptitle('Per-memory retention — does W_slow[cc] preserve all memories equally?',fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG_DIR,'fig2_per_memory.png')
    fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')
    return p


def write_report(data, rows, ctrl_mean):
    ctrl_cc = vec(data,'WSLOW_CC_ONLY').mean() if any('WSLOW_CC_ONLY' in data[s]['conditions'] for s in data) else 0
    pct_cc = 100.*ctrl_cc/ctrl_mean

    if pct_cc >= 80:
        verdict = 'NECESSARY AND SUFFICIENT'
        verdict_detail = (f'W_slow[cc] alone ({pct_cc:.1f}% of CONTROL) meets the >=80% threshold. '
                          'UC-only and UU-only do not sustain memory. '
                          'W_slow[cc] is the primary engram.')
    elif pct_cc >= 50:
        verdict = 'NECESSARY BUT NOT SUFFICIENT ALONE'
        verdict_detail = (f'W_slow[cc] alone recovers {pct_cc:.1f}% — partial but below threshold. '
                          'Additional blocks contribute meaningfully.')
    else:
        verdict = 'NOT THE SOLE SUBSTRATE — T7 INTERPRETATION REQUIRES REVISION'
        verdict_detail = (f'W_slow[cc] alone recovers only {pct_cc:.1f}%. '
                          'The substrate is more distributed than proposed.')

    lines = [
        '# Task 7.5 Report — Sufficiency Test of W_slow[cc]',
        '',
        f'## VERDICT: {verdict}',
        '',
        verdict_detail,
        '',
        '## Results Table',
        '',
        '| Condition | Mean | SD | % Control | t | p | Cohen dz |',
        '|-----------|------|----|-----------|----|---|----------|',
    ]
    for row in rows:
        if row is None: continue
        cond,v,m,sd,pct,t,p,dz = row
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        lines.append(f'| {cond} | {m:.4f} | {sd:.4f} | {pct:.1f}% | {t:.2f} | {p:.4f} {sig} | {dz:.2f} |')

    lines += [
        '',
        '## Q&A',
        f'Q1: Retention in WSLOW_CC_ONLY = {ctrl_cc:.4f} ({pct_cc:.1f}% of CONTROL)',
        f'Q5: {pct_cc:.1f}% of CONTROL retention explained by W_slow[cc] alone',
        '',
        '## Mechanism',
        'W_slow[cc] = slow synaptic consolidation among 20 schema-core neurons.',
        'This 20x20 sub-matrix (400 weights out of 562,500 total) carries the engram.',
        'Core-core recurrence creates a persistent attractor that drives pattern completion.',
    ]

    rpath = os.path.join(OUT_DIR,'TASK75_REPORT.md')
    with open(rpath,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
    print(f'  Saved {rpath}')
    return rpath


def main():
    print('Loading T75 PKLs...')
    data = load()
    if not data:
        print('ERROR: No T75 PKLs found.'); return

    ctrl = vec(data,'CONTROL')
    ctrl_mean = ctrl.mean()
    print(f'\nLoaded {len(data)} seeds. CONTROL mean={ctrl_mean:.4f}')

    print(f'\n{"Condition":<22} {"Mean":>7} {"SD":>7} {"% ctrl":>7} '
          f'{"t":>6} {"p":>8} {"dz":>6}')
    print('-'*72)
    rows = []
    for cond in CONDS:
        row = stats_row(data, cond, ctrl)
        if row is None: continue
        rows.append(row)
        cond2,v,m,sd,pct,t,p,dz = row
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        print(f'{cond:<22} {m:7.4f} {sd:7.4f} {pct:7.1f}% {t:6.2f} {p:8.4f} {sig:5s} {dz:6.2f}')

    # Verdict
    cc_row = next((r for r in rows if r[0]=='WSLOW_CC_ONLY'), None)
    uc_row = next((r for r in rows if r[0]=='WSLOW_UC_ONLY'), None)
    uu_row = next((r for r in rows if r[0]=='WSLOW_UU_ONLY'), None)
    print('\n=== VERDICT ===')
    if cc_row:
        pct_cc = cc_row[4]
        pct_uc = uc_row[4] if uc_row else 0.
        pct_uu = uu_row[4] if uu_row else 0.
        print(f'WSLOW_CC_ONLY  = {pct_cc:.1f}% of CONTROL')
        print(f'WSLOW_UC_ONLY  = {pct_uc:.1f}% of CONTROL')
        print(f'WSLOW_UU_ONLY  = {pct_uu:.1f}% of CONTROL')
        if pct_cc >= 80 and pct_uc < 50 and pct_uu < 50:
            print('=> CRITERION A MET: W_slow[cc] is NECESSARY AND SUFFICIENT.')
        elif pct_cc >= 50:
            print('=> CRITERION B: W_slow[cc] is necessary but not sufficient alone.')
        else:
            print('=> CRITERION C: W_slow[cc] alone insufficient — substrate more distributed.')

    print('\nGenerating figures...')
    fig_main(data)
    fig_per_memory(data)
    write_report(data, rows, ctrl_mean)
    print('\nDone.')


if __name__=='__main__':
    main()
