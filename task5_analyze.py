"""
TASK 5 ANALYSIS — Causal Role of Wcc
=====================================
Tables, statistics, 4 figures, verdict, CSV, report.
"""
import os, sys, pickle, csv, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task5'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042]
CONDS = ['FULL', 'WCC_WEAKEN', 'WCC_DESTROY', 'WCC_ENHANCE']
COLORS = {'FULL':'#2166AC','WCC_WEAKEN':'#F4A736','WCC_DESTROY':'#D6604D','WCC_ENHANCE':'#5AAE61'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})


def load():
    data = {}
    for s in SEEDS:
        p = os.path.join(OUT_DIR, f'T5_seed{s}.pkl')
        if os.path.exists(p):
            with open(p,'rb') as f: data[s] = pickle.load(f)
        else: print(f'MISSING {p}')
    return data


def vec(data, cond, key):
    return np.array([data[s]['conditions'][cond][key] for s in SEEDS if s in data])


def d(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or len(b)<2: return float('nan')
    pl=np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/pl if pl else float('nan')


def per_seed_table(data):
    print(f'\n{"="*100}')
    print('TABLE 1: PER-SEED RESULTS')
    print(f'{"="*100}')
    print(f'  {"Seed":>5s} {"Condition":<13s} {"Wcc":>8s} {"Wuc":>8s} {"Wuu":>8s} '
          f'{"S1":>8s} {"Retention":>10s} {"Retrieval":>10s} {"Replay":>7s}')
    print('  '+'-'*86)
    for s in SEEDS:
        if s not in data: continue
        for cond in CONDS:
            c = data[s]['conditions'][cond]
            print(f'  {s:>5d} {cond:<13s} {c["Wcc"]:>8.4f} {c["Wuc"]:>8.4f} {c["Wuu"]:>8.4f} '
                  f'{c["S1"]:>8.4f} {c["retention_mean"]:>10.4f} {c["retrieval_mean"]:>10.4f} '
                  f'{c["replay_events"]:>7d}')
        print()


def means_table(data):
    print(f'\n{"="*90}')
    print('TABLE 2: CONDITION MEANS (n=5, mean +/- SEM)')
    print(f'{"="*90}')
    print(f'  {"Condition":<13s} {"Wcc":>14s} {"S1":>14s} {"Retention":>16s} {"Retrieval":>16s}')
    print('  '+'-'*76)
    for cond in CONDS:
        def fmt(k):
            v=vec(data,cond,k); se=v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0
            return f'{v.mean():.4f}+/-{se:.4f}'
        print(f'  {cond:<13s} {fmt("Wcc"):>14s} {fmt("S1"):>14s} '
              f'{fmt("retention_mean"):>16s} {fmt("retrieval_mean"):>16s}')


def stats_table(data):
    print(f'\n{"="*90}')
    print('TABLE 3: STATISTICS (FULL vs each intervention)')
    print(f'{"="*90}')
    for metric in ['retention_mean','Wcc','S1','retrieval_mean']:
        print(f'\n  {metric}:')
        full = vec(data,'FULL',metric)
        for cond in ['WCC_WEAKEN','WCC_DESTROY','WCC_ENHANCE']:
            cv = vec(data,cond,metric)
            if len(full)<2 or len(cv)<2: continue
            t,p = ttest_ind(full,cv,equal_var=False)
            dd = d(full,cv)
            dlt = full.mean()-cv.mean()
            pct = 100*dlt/max(abs(full.mean()),1e-9)
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'    FULL vs {cond:<13s} delta={dlt:+.4f} ({pct:+.0f}%) '
                  f'd={dd:+.2f} t={t:+.2f} p={p:.4g} {sig}')


def figures(data):
    # Fig 1: Retention across conditions
    fig,ax=plt.subplots(figsize=(8,5.5))
    xs=np.arange(len(CONDS))
    for i,cond in enumerate(CONDS):
        v=vec(data,cond,'retention_mean')
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v)>1 else 0, capsize=5,
               color=COLORS[cond], edgecolor='black', alpha=0.85)
        jit=np.random.default_rng(0).uniform(-0.12,0.12,len(v))
        ax.scatter(i+jit, v, color='black', s=28, alpha=0.6, zorder=5, edgecolor='white', linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(CONDS, fontsize=9, rotation=15)
    ax.set_ylabel('Retention (mean)', fontweight='bold')
    ax.set_title('Retention across Wcc interventions (n=5)\nSame trained network, post-hoc core-core edit')
    ax.grid(axis='y',alpha=0.3); ax.axhline(0,color='grey',lw=0.5,ls=':')
    fig.tight_layout()
    for e in ('png','pdf','svg'): fig.savefig(os.path.join(FIG_DIR,f'fig1_retention.{e}'),dpi=300,bbox_inches='tight')
    plt.close(fig); print('  fig1_retention')

    # Fig 2: Wcc across conditions
    fig,ax=plt.subplots(figsize=(8,5.5))
    for i,cond in enumerate(CONDS):
        v=vec(data,cond,'Wcc')
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v)>1 else 0, capsize=5,
               color=COLORS[cond], edgecolor='black', alpha=0.85)
    ax.set_xticks(np.arange(len(CONDS))); ax.set_xticklabels(CONDS, fontsize=9, rotation=15)
    ax.set_ylabel('Wcc (core-core weight)', fontweight='bold')
    ax.set_title('Wcc across interventions (verification of manipulation)')
    ax.grid(axis='y',alpha=0.3)
    fig.tight_layout()
    for e in ('png','pdf','svg'): fig.savefig(os.path.join(FIG_DIR,f'fig2_wcc.{e}'),dpi=300,bbox_inches='tight')
    plt.close(fig); print('  fig2_wcc')

    # Fig 3: Retention vs Wcc scatter
    fig,ax=plt.subplots(figsize=(8,5.5))
    allw, allr = [], []
    for cond in CONDS:
        w=vec(data,cond,'Wcc'); rr=vec(data,cond,'retention_mean')
        ax.scatter(w, rr, color=COLORS[cond], s=60, alpha=0.75, label=cond,
                   edgecolor='white', linewidth=0.6, zorder=5)
        allw.extend(w); allr.extend(rr)
    allw,allr=np.array(allw),np.array(allr)
    if len(allw)>=3:
        rv,pv=pearsonr(allw,allr)
        z=np.polyfit(allw,allr,1); xl=np.linspace(allw.min(),allw.max(),50)
        ax.plot(xl,np.polyval(z,xl),'--k',lw=1.5,alpha=0.6)
        ax.set_title(f'Retention vs Wcc (intervention sweep)\nr={rv:+.3f}, p={pv:.2e}, R^2={rv**2:.3f}')
    ax.set_xlabel('Wcc', fontweight='bold'); ax.set_ylabel('Retention', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    for e in ('png','pdf','svg'): fig.savefig(os.path.join(FIG_DIR,f'fig3_retention_vs_wcc.{e}'),dpi=300,bbox_inches='tight')
    plt.close(fig); print('  fig3_retention_vs_wcc')

    # Fig 4: Effect sizes
    fig,ax=plt.subplots(figsize=(9,5.5))
    conds=['WCC_WEAKEN','WCC_DESTROY','WCC_ENHANCE']; x=np.arange(len(conds))
    full=vec(data,'FULL','retention_mean')
    ds=[d(full,vec(data,c,'retention_mean')) for c in conds]
    colors=['#D6604D' if dd>0 else '#5AAE61' for dd in ds]
    ax.bar(x, ds, color=colors, edgecolor='black', alpha=0.85)
    for xi,(c,dd) in enumerate(zip(conds,ds)):
        cv=vec(data,c,'retention_mean')
        _,p=ttest_ind(full,cv,equal_var=False)
        sig='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        ax.text(xi, dd+(0.3 if dd>=0 else -0.6), sig, ha='center', fontsize=11, fontweight='bold')
    ax.axhline(0,color='black',lw=0.8); ax.set_xticks(x); ax.set_xticklabels(conds,fontsize=9)
    ax.set_ylabel("Cohen's d (FULL - intervention, retention)")
    ax.set_title('Effect of Wcc intervention on Retention')
    ax.grid(axis='y',alpha=0.3)
    fig.tight_layout()
    for e in ('png','pdf','svg'): fig.savefig(os.path.join(FIG_DIR,f'fig4_effect_sizes.{e}'),dpi=300,bbox_inches='tight')
    plt.close(fig); print('  fig4_effect_sizes')


def verdict_and_report(data):
    full_r = vec(data,'FULL','retention_mean')
    weak_r = vec(data,'WCC_WEAKEN','retention_mean')
    dest_r = vec(data,'WCC_DESTROY','retention_mean')
    enh_r  = vec(data,'WCC_ENHANCE','retention_mean')

    # R^2 across intervention sweep
    allw,allr=[],[]
    for cond in CONDS:
        allw.extend(vec(data,cond,'Wcc')); allr.extend(vec(data,cond,'retention_mean'))
    rv,pv = pearsonr(allw,allr) if len(allw)>=3 else (float('nan'),float('nan'))

    # Q-answers
    q1 = weak_r.mean() < full_r.mean()  # reducing Wcc reduces retention?
    _,p_weak = ttest_ind(full_r,weak_r,equal_var=False)
    q2 = dest_r.mean() < 0.5*full_r.mean()  # destroying collapses?
    _,p_dest = ttest_ind(full_r,dest_r,equal_var=False)
    q3 = enh_r.mean() > full_r.mean()  # enhancing improves?
    _,p_enh = ttest_ind(full_r,enh_r,equal_var=False)

    print(f'\n{"="*82}\nVERDICT\n{"="*82}')
    print(f'  FULL retention:        {full_r.mean():.4f} +/- {full_r.std(ddof=1):.4f}')
    print(f'  WCC_WEAKEN retention:  {weak_r.mean():.4f} +/- {weak_r.std(ddof=1):.4f}  '
          f'(d={d(full_r,weak_r):+.2f}, p={p_weak:.3g})')
    print(f'  WCC_DESTROY retention: {dest_r.mean():.4f} +/- {dest_r.std(ddof=1):.4f}  '
          f'(d={d(full_r,dest_r):+.2f}, p={p_dest:.3g})')
    print(f'  WCC_ENHANCE retention: {enh_r.mean():.4f} +/- {enh_r.std(ddof=1):.4f}  '
          f'(d={d(full_r,enh_r):+.2f}, p={p_enh:.3g})')
    print(f'\n  Q1 (reduce Wcc -> reduce retention?): {"YES" if q1 and p_weak<0.05 else "NO/weak"}')
    print(f'  Q2 (destroy Wcc -> collapse retention?): {"YES" if q2 and p_dest<0.05 else "NO"}')
    print(f'  Q3 (enhance Wcc -> improve retention?): {"YES" if q3 and p_enh<0.05 else "NO"}')
    print(f'  Q4 (retention variance explained by Wcc): R^2={rv**2:.3f} (r={rv:+.3f}, p={pv:.2e})')
    print(f'  Q5 (replay reproduced by Wcc manip?): see report')

    # Decide verdict
    necessary = (q2 and p_dest < 0.05)  # destroying Wcc collapses retention
    monotone  = (q1 and q3)             # both directions move retention
    if necessary and monotone:
        verdict = 'C'; vtxt = 'Wcc is necessary but not sufficient (or A/B — see nuance)'
    elif necessary:
        verdict = 'A'; vtxt = 'Wcc is causally necessary'
    elif q3 and p_enh < 0.05:
        verdict = 'B'; vtxt = 'Wcc is causally sufficient (enhancing improves)'
    elif rv**2 > 0.5 and not necessary:
        verdict = 'D'; vtxt = 'Wcc is only a correlate (manipulation does not move retention)'
    else:
        verdict = 'D'; vtxt = 'Wcc is only a correlate'
    print(f'\n  VERDICT: {verdict}) {vtxt}')

    # CSV
    csv_path=os.path.join(OUT_DIR,'task5_summary.csv')
    with open(csv_path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['seed','condition','Wcc','Wuc','Wuu','S1','retention_mean','retrieval_mean','replay_events'])
        for s in SEEDS:
            if s not in data: continue
            for cond in CONDS:
                c=data[s]['conditions'][cond]
                w.writerow([s,cond,c['Wcc'],c['Wuc'],c['Wuu'],c['S1'],
                            c['retention_mean'],c['retrieval_mean'],c['replay_events']])
    print(f'  CSV: {csv_path}')

    # Report
    rp=os.path.join(OUT_DIR,'TASK5_REPORT.md')
    with open(rp,'w',encoding='utf-8') as f:
        f.write('# TASK 5 -- CAUSAL ROLE OF Wcc\n\n')
        f.write('Post-hoc intervention on the trained core-core weight block.\n')
        f.write('All 4 conditions share the identical trained network per seed.\n\n')
        f.write('## Table 2: Condition means (n=5)\n\n')
        f.write('| Condition | Wcc | S1 | Retention | Retrieval |\n|---|---|---|---|---|\n')
        for cond in CONDS:
            f.write(f'| {cond} | {vec(data,cond,"Wcc").mean():.4f} | {vec(data,cond,"S1").mean():.4f} '
                    f'| {vec(data,cond,"retention_mean").mean():.4f} | {vec(data,cond,"retrieval_mean").mean():.4f} |\n')
        f.write('\n## Table 3: Statistics (FULL vs intervention, retention)\n\n')
        f.write('| Comparison | delta | Cohen d | p |\n|---|---|---|---|\n')
        for cond in ['WCC_WEAKEN','WCC_DESTROY','WCC_ENHANCE']:
            cv=vec(data,cond,'retention_mean')
            t,p=ttest_ind(full_r,cv,equal_var=False)
            f.write(f'| FULL vs {cond} | {full_r.mean()-cv.mean():+.4f} | {d(full_r,cv):+.2f} | {p:.3g} |\n')
        f.write(f'\n## Answers\n\n')
        f.write(f'- Q1 reduce Wcc -> reduce retention: {"YES" if q1 and p_weak<0.05 else "NO/weak"}\n')
        f.write(f'- Q2 destroy Wcc -> collapse retention: {"YES" if q2 and p_dest<0.05 else "NO"}\n')
        f.write(f'- Q3 enhance Wcc -> improve retention: {"YES" if q3 and p_enh<0.05 else "NO"}\n')
        f.write(f'- Q4 retention variance explained by Wcc: R^2={rv**2:.3f}\n')
        f.write(f'\n## Verdict: {verdict}) {vtxt}\n')
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
    print('\nTASK 5 COMPLETE.')
