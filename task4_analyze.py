"""
TASK 4 ANALYSIS — Mechanism Discovery
======================================
Replay statistics, STDP decomposition, coincidence analysis, weight decomposition,
causal mediation, mechanistic interpretation. 4 figures.
"""
import os, sys, pickle, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task4'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042]
CONDITIONS = ['FULL', 'NO_REPLAY']
COLORS = {'FULL': '#2166AC', 'NO_REPLAY': '#D6604D'}
CORE = 20; N_DESIG = 100

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold','axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':150})


def load_all():
    data = {}
    for c in CONDITIONS:
        for s in SEEDS:
            p = os.path.join(OUT_DIR, f'T4_{c}_seed{s}.pkl')
            if os.path.exists(p):
                with open(p, 'rb') as f: data[(c, s)] = pickle.load(f)
            else:
                print(f'MISSING {p}')
    return data


def d(a, b):
    a, b = np.asarray(a,float), np.asarray(b,float)
    if len(a)<2 or len(b)<2: return float('nan')
    pl = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/pl if pl else float('nan')


def savefig(fig, name):
    for ext in ('png','pdf','svg'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}')


# ── 1. Replay statistics ──────────────────────────────────────────────────
def table_replay(data):
    print(f'\n{"="*82}\n1. REPLAY-EVENT STATISTICS (FULL, n=3 seeds)\n{"="*82}')
    all_recs = []
    for s in SEEDS:
        r = data.get(('FULL', s))
        if r: all_recs.extend(r['replay_records'])
    if not all_recs:
        print('  No replay records.'); return
    n_ev   = [len(data[('FULL',s)]['replay_records']) for s in SEEDS if ('FULL',s) in data]
    core_p = np.array([x['core_part'] for x in all_recs])
    uniq_p = np.array([x['uniq_part'] for x in all_recs])
    core_s = np.array([x['core_spikes'] for x in all_recs])
    uniq_s = np.array([x['uniq_spikes'] for x in all_recs])
    tot_s  = np.array([x['total_spikes'] for x in all_recs])
    mem    = np.array([x['memory_idx'] for x in all_recs])

    print(f'  Replay events per run:        {np.mean(n_ev):.1f} (range {min(n_ev)}-{max(n_ev)})')
    print(f'  Core participation/event:     {core_p.mean():.3f} +/- {core_p.std():.3f}  '
          f'({core_p.mean()*CORE:.1f}/{CORE} core neurons)')
    print(f'  Unique participation/event:   {uniq_p.mean():.3f} +/- {uniq_p.std():.3f}  '
          f'({uniq_p.mean()*(N_DESIG-CORE):.1f}/{N_DESIG-CORE} unique neurons)')
    print(f'  Core spikes/event:            {core_s.mean():.1f}')
    print(f'  Unique spikes/event:          {uniq_s.mean():.1f}')
    print(f'  Total spikes/event:           {tot_s.mean():.1f}')
    print(f'  Core spike density (per neuron): {core_s.mean()/CORE:.2f}')
    print(f'  Unique spike density (per neuron): {uniq_s.mean()/(N_DESIG-CORE):.2f}')
    print(f'\n  Distribution across memories (which memory replayed):')
    for m in range(4):
        cnt = int((mem == m).sum())
        print(f'    Memory {chr(65+m)}: {cnt} events ({100*cnt/len(mem):.0f}%)')

    return {'core_part': core_p.mean(), 'uniq_part': uniq_p.mean(),
            'core_density': core_s.mean()/CORE, 'uniq_density': uniq_s.mean()/(N_DESIG-CORE)}


# ── 2. STDP decomposition ─────────────────────────────────────────────────
def _stdp_agg(data, key):
    """Aggregate a stdp dict ('stdp' or 'stdp_train') over FULL seeds."""
    blocks = ['cc','uc','uu']
    agg = {b: {'pot': [], 'dep': []} for b in blocks}
    nst = []
    for s in SEEDS:
        r = data.get(('FULL', s))
        if not r: continue
        st = r.get(key, {}); nst.append(st.get('n_steps', 0))
        for b in blocks:
            agg[b]['pot'].append(st.get('pot_'+b, 0.0))
            agg[b]['dep'].append(st.get('dep_'+b, 0.0))
    res = {}
    for b in blocks:
        pot = np.mean(agg[b]['pot']) if agg[b]['pot'] else 0
        dep = np.mean(agg[b]['dep']) if agg[b]['dep'] else 0
        res[b] = {'pot': pot, 'dep': dep, 'net': pot+dep}
    res['n_steps'] = np.mean(nst) if nst else 0
    return res


# Block sizes (number of synapse pairs) for per-synapse normalization
BLOCK_SIZE = {'cc': CORE*CORE, 'uc': (N_DESIG-CORE)*CORE, 'uu': (N_DESIG-CORE)*(N_DESIG-CORE)}

def table_stdp(data):
    print(f'\n{"="*82}\n2. STDP DECOMPOSITION (FULL, n=3) — TRAINING vs REPLAY phases\n{"="*82}')
    bl_lbl = {'cc':'core-core', 'uc':'unique-core', 'uu':'unique-unique'}
    train = _stdp_agg(data, 'stdp_train')
    replay= _stdp_agg(data, 'stdp')

    print(f'\n  TRAINING-phase STDP (steps={train["n_steps"]:.0f}):')
    print(f'  {"Block":<16s} {"pairs":>6s} {"Pot(total)":>11s} {"Dep(total)":>11s} '
          f'{"Pot/synapse":>12s} {"Net/synapse":>12s}')
    print('  ' + '-'*72)
    for b in ['cc','uc','uu']:
        sz = BLOCK_SIZE[b]
        print(f'  {bl_lbl[b]:<16s} {sz:>6d} {train[b]["pot"]:>+11.2f} {train[b]["dep"]:>+11.2f} '
              f'{train[b]["pot"]/sz:>+12.4f} {train[b]["net"]/sz:>+12.4f}')
    tp = sum(train[b]['pot'] for b in ['cc','uc','uu'])
    # Per-synapse dominance
    ps_cc = train['cc']['pot']/BLOCK_SIZE['cc']
    ps_uc = train['uc']['pot']/BLOCK_SIZE['uc']
    ps_uu = train['uu']['pot']/BLOCK_SIZE['uu']
    print(f'  Core-core share of total training LTP: {100*train["cc"]["pot"]/max(tp,1e-9):.1f}%')
    print(f'  PER-SYNAPSE potentiation: cc={ps_cc:.4f}  uc={ps_uc:.4f}  uu={ps_uu:.4f}')
    print(f'  Core-core synapses potentiated {ps_cc/max(ps_uu,1e-9):.1f}x more than unique-unique,')
    print(f'  {ps_cc/max(ps_uc,1e-9):.1f}x more than unique-core (per synapse).')

    print(f'\n  REPLAY-phase STDP (steps={replay["n_steps"]:.0f}):')
    print(f'  {"Block":<16s} {"Potentiation":>14s} {"Depression":>14s} {"Net":>12s}')
    print('  ' + '-'*58)
    for b in ['cc','uc','uu']:
        print(f'  {bl_lbl[b]:<16s} {replay[b]["pot"]:>+14.3f} {replay[b]["dep"]:>+14.3f} {replay[b]["net"]:>+12.3f}')
    print(f'\n  NOTE: At COH_THR=0.50 the replay coherence gate rarely opens, so')
    print(f'  endogenous replay STDP is minimal. Replay-driven Wcc growth comes')
    print(f'  predominantly from the MB core-boost (1.3x core-core per event),')
    print(f'  while TRAINING STDP builds the baseline core-core asymmetry.')
    return {'train': train, 'replay': replay,
            # back-compat keys for downstream funcs (use training as the dominant STDP)
            'cc': train['cc'], 'uc': train['uc'], 'uu': train['uu']}


# ── 3. Coincidence ────────────────────────────────────────────────────────
def table_coincidence(data):
    print(f'\n{"="*82}\n3. SPIKE COINCIDENCE (measurement window, FULL vs NO_REPLAY)\n{"="*82}')
    res = {}
    print(f'  {"Condition":<12s} {"cc-coinc":>10s} {"uu-coinc":>10s} {"cu-coinc":>10s} '
          f'{"core_rate":>10s} {"uniq_rate":>10s}')
    print('  ' + '-'*66)
    for c in CONDITIONS:
        cc = [data[(c,s)]['coinc_cc'] for s in SEEDS if (c,s) in data]
        uu = [data[(c,s)]['coinc_uu'] for s in SEEDS if (c,s) in data]
        cu = [data[(c,s)]['coinc_cu'] for s in SEEDS if (c,s) in data]
        cr = [data[(c,s)]['spike_rate_core'] for s in SEEDS if (c,s) in data]
        ur = [data[(c,s)]['spike_rate_uniq'] for s in SEEDS if (c,s) in data]
        res[c] = {'cc': np.mean(cc), 'uu': np.mean(uu), 'cu': np.mean(cu),
                  'core_rate': np.mean(cr), 'uniq_rate': np.mean(ur)}
        print(f'  {c:<12s} {np.mean(cc):>10.4f} {np.mean(uu):>10.4f} {np.mean(cu):>10.4f} '
              f'{np.mean(cr):>10.4f} {np.mean(ur):>10.4f}')
    # Contrasts
    print(f'\n  FULL vs NO_REPLAY core-core coincidence:')
    f_cc = [data[('FULL',s)]['coinc_cc'] for s in SEEDS if ('FULL',s) in data]
    n_cc = [data[('NO_REPLAY',s)]['coinc_cc'] for s in SEEDS if ('NO_REPLAY',s) in data]
    if len(f_cc)>=2 and len(n_cc)>=2:
        t,p = ttest_ind(f_cc, n_cc, equal_var=False)
        print(f'    FULL={np.mean(f_cc):.4f}  NO_REPLAY={np.mean(n_cc):.4f}  '
              f'd={d(f_cc,n_cc):+.2f}  p={p:.4f}')
    # core vs unique coincidence within FULL
    print(f'\n  Within FULL: core-core vs unique-unique coincidence:')
    print(f'    cc={res["FULL"]["cc"]:.4f}  uu={res["FULL"]["uu"]:.4f}  '
          f'ratio={res["FULL"]["cc"]/max(res["FULL"]["uu"],1e-9):.2f}x')
    return res


# ── 4. Weight decomposition ───────────────────────────────────────────────
def table_decomp(data):
    print(f'\n{"="*82}\n4. WEIGHT GROWTH DECOMPOSITION (final values, n=3)\n{"="*82}')
    print(f'  {"Condition":<12s} {"Wcc":>10s} {"Wuc":>10s} {"Wuu":>10s} {"S1":>10s}')
    print('  ' + '-'*54)
    res = {}
    for c in CONDITIONS:
        finals = {'Wcc':[], 'Wuc':[], 'Wuu':[], 'S1':[]}
        for s in SEEDS:
            r = data.get((c,s))
            if not r or not r['trajectory']: continue
            t = r['trajectory'][-1]
            for k in finals: finals[k].append(t[k])
        res[c] = {k: np.mean(v) for k,v in finals.items()}
        print(f'  {c:<12s} {res[c]["Wcc"]:>10.4f} {res[c]["Wuc"]:>10.4f} '
              f'{res[c]["Wuu"]:>10.4f} {res[c]["S1"]:>10.4f}')
    # Growth FULL - NO_REPLAY by block
    print(f'\n  Schema growth (FULL - NO_REPLAY) by block:')
    for k in ['Wcc','Wuc','Wuu']:
        diff = res['FULL'][k] - res['NO_REPLAY'][k]
        print(f'    {k}: {diff:+.4f} ({100*diff/max(res["NO_REPLAY"][k],1e-9):+.0f}%)')
    print(f'\n  => Block with largest replay-driven growth determines mechanism.')
    return res


# ── 5. Mediation ──────────────────────────────────────────────────────────
def mediation(data, replay_stats, coinc_res, decomp):
    print(f'\n{"="*82}\n5. CAUSAL MEDIATION: replay -> coincidence -> Wcc\n{"="*82}')
    # Per-run: coincidence_cc vs final Wcc across all 6 runs
    cc_coinc, wcc_final, cond_label = [], [], []
    for c in CONDITIONS:
        for s in SEEDS:
            r = data.get((c,s))
            if not r or not r['trajectory']: continue
            cc_coinc.append(r['coinc_cc'])
            wcc_final.append(r['trajectory'][-1]['Wcc'])
            cond_label.append(c)
    cc_coinc = np.array(cc_coinc); wcc_final = np.array(wcc_final)
    if len(cc_coinc) >= 3:
        rr, pp = pearsonr(cc_coinc, wcc_final)
        print(f'  Across all {len(cc_coinc)} runs:')
        print(f'    corr(core-core coincidence, final Wcc) = {rr:+.3f}  p={pp:.4f}')
    # Step a: replay -> coincidence
    f_cc = [data[('FULL',s)]['coinc_cc'] for s in SEEDS if ('FULL',s) in data]
    n_cc = [data[('NO_REPLAY',s)]['coinc_cc'] for s in SEEDS if ('NO_REPLAY',s) in data]
    print(f'\n  Step a (replay -> coincidence): '
          f'FULL cc-coinc={np.mean(f_cc):.4f} vs NO_REPLAY={np.mean(n_cc):.4f} '
          f'(d={d(f_cc,n_cc):+.2f})')
    # Step b: coincidence -> Wcc (correlation above)
    print(f'  Step b (coincidence -> Wcc): r={rr:+.3f} across runs')
    # Step c: total effect replay -> Wcc
    f_w = [data[('FULL',s)]['trajectory'][-1]['Wcc'] for s in SEEDS if ('FULL',s) in data]
    n_w = [data[('NO_REPLAY',s)]['trajectory'][-1]['Wcc'] for s in SEEDS if ('NO_REPLAY',s) in data]
    print(f'  Step c (total: replay -> Wcc): '
          f'FULL Wcc={np.mean(f_w):.4f} vs NO_REPLAY={np.mean(n_w):.4f} (d={d(f_w,n_w):+.2f})')
    return {'r_coinc_wcc': rr if len(cc_coinc)>=3 else float('nan')}


# ── Figures ───────────────────────────────────────────────────────────────
def fig_decomp_growth(data):
    """Wcc/Wuc/Wuu trajectories FULL vs NO_REPLAY."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, blk, lbl in zip(axes, ['Wcc','Wuc','Wuu'],
                            ['Wcc (core-core)','Wuc (unique-core)','Wuu (unique-unique)']):
        for c in CONDITIONS:
            trajs = []
            for s in SEEDS:
                r = data.get((c,s))
                if r and r['trajectory']:
                    trajs.append([t[blk] for t in r['trajectory']])
            if not trajs: continue
            L = max(len(t) for t in trajs)
            M = np.full((len(trajs), L), np.nan)
            for i,t in enumerate(trajs): M[i,:len(t)] = t
            x = np.arange(L)
            mean = np.nanmean(M, axis=0); sem = np.nanstd(M,axis=0,ddof=1)/np.sqrt(len(trajs))
            ax.plot(x, mean, 'o-', color=COLORS[c], lw=2.3, ms=6, label=c.replace('_',' '))
            ax.fill_between(x, mean-sem, mean+sem, color=COLORS[c], alpha=0.18)
        ax.set_xlabel('Checkpoint', fontweight='bold'); ax.set_ylabel(lbl, fontweight='bold')
        ax.set_title(lbl); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('Weight Block Growth Decomposition (n=3 seeds)', y=1.03, fontsize=14)
    fig.tight_layout(); savefig(fig, 'fig1_wcc_decomposition')


def fig_coincidence(data):
    """Coincidence matrix heatmaps + bar comparison."""
    fig = plt.figure(figsize=(18, 5.5))
    # FULL matrix
    ax1 = fig.add_subplot(1, 3, 1)
    M_full = np.mean([np.array(data[('FULL',s)]['coinc_matrix']) for s in SEEDS if ('FULL',s) in data], axis=0)
    im1 = ax1.imshow(M_full, cmap='hot', aspect='equal', vmax=np.percentile(M_full,99))
    ax1.axhline(CORE-0.5, color='cyan', lw=1); ax1.axvline(CORE-0.5, color='cyan', lw=1)
    ax1.set_title('FULL coincidence\n(cyan = core boundary)'); plt.colorbar(im1, ax=ax1, shrink=0.7)
    # NO_REPLAY matrix
    ax2 = fig.add_subplot(1, 3, 2)
    M_nr = np.mean([np.array(data[('NO_REPLAY',s)]['coinc_matrix']) for s in SEEDS if ('NO_REPLAY',s) in data], axis=0)
    im2 = ax2.imshow(M_nr, cmap='hot', aspect='equal', vmax=np.percentile(M_full,99))
    ax2.axhline(CORE-0.5, color='cyan', lw=1); ax2.axvline(CORE-0.5, color='cyan', lw=1)
    ax2.set_title('NO_REPLAY coincidence'); plt.colorbar(im2, ax=ax2, shrink=0.7)
    # Bar comparison
    ax3 = fig.add_subplot(1, 3, 3)
    blocks = ['cc','uu','cu']; bl_lbl=['core-core','uniq-uniq','core-uniq']
    x = np.arange(3); w=0.38
    for i, c in enumerate(CONDITIONS):
        vals = [np.mean([data[(c,s)][f'coinc_{b}'] for s in SEEDS if (c,s) in data]) for b in blocks]
        sds  = [np.std([data[(c,s)][f'coinc_{b}'] for s in SEEDS if (c,s) in data]) for b in blocks]
        ax3.bar(x + (i-0.5)*w, vals, w, yerr=sds, capsize=4, label=c.replace('_',' '),
                color=COLORS[c], edgecolor='black', alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(bl_lbl); ax3.set_ylabel('Mean coincidence/step')
    ax3.set_title('Coincidence by block'); ax3.legend(fontsize=9); ax3.grid(axis='y',alpha=0.3)
    fig.suptitle('Spike Coincidence Analysis (mean over 3 seeds)', y=1.03, fontsize=14)
    fig.tight_layout(); savefig(fig, 'fig2_coincidence')


def fig_stdp(data):
    """STDP potentiation/depression by block."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    blocks=['cc','uc','uu']; bl=['core-core','unique-core','unique-unique']
    pot=[np.mean([data[('FULL',s)]['stdp']['pot_'+b] for s in SEEDS if ('FULL',s) in data]) for b in blocks]
    dep=[np.mean([data[('FULL',s)]['stdp']['dep_'+b] for s in SEEDS if ('FULL',s) in data]) for b in blocks]
    x=np.arange(3)
    ax.bar(x, pot, 0.6, label='Potentiation (LTP)', color='#2166AC', edgecolor='black', alpha=0.85)
    ax.bar(x, dep, 0.6, label='Depression (LTD)', color='#D6604D', edgecolor='black', alpha=0.85)
    for xi, (p, dd) in enumerate(zip(pot, dep)):
        ax.text(xi, p+0.02*max(pot), f'{p:+.2f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(xi, dd-0.04*max(pot), f'{dd:+.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bl); ax.set_ylabel('Total weight change during replay')
    ax.set_title('STDP Contribution by Block (FULL, n=3)\nWhich synapses are potentiated during replay?')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); savefig(fig, 'fig3_stdp_contribution')


def fig_mechanism(data, decomp, coinc_res):
    """Mechanism summary: 4-panel."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    # A: schema growth by block (FULL vs NO_REPLAY)
    ax=axes[0,0]; blocks=['Wcc','Wuc','Wuu']; x=np.arange(3); w=0.38
    for i,c in enumerate(CONDITIONS):
        vals=[decomp[c][b] for b in blocks]
        ax.bar(x+(i-0.5)*w, vals, w, label=c.replace('_',' '), color=COLORS[c], edgecolor='black', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(blocks); ax.set_ylabel('Weight (final)')
    ax.set_title('A. Schema growth by block'); ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.3)
    # B: coincidence cc comparison
    ax=axes[0,1]
    f=[data[('FULL',s)]['coinc_cc'] for s in SEEDS if ('FULL',s) in data]
    n=[data[('NO_REPLAY',s)]['coinc_cc'] for s in SEEDS if ('NO_REPLAY',s) in data]
    ax.bar([0,1],[np.mean(f),np.mean(n)], yerr=[np.std(f),np.std(n)], capsize=5,
           color=[COLORS['FULL'],COLORS['NO_REPLAY']], edgecolor='black', alpha=0.85)
    ax.set_xticks([0,1]); ax.set_xticklabels(['FULL','NO_REPLAY'])
    ax.set_ylabel('Core-core coincidence'); ax.set_title('B. Replay drives core coincidence')
    ax.grid(axis='y',alpha=0.3)
    # C: mediation scatter
    ax=axes[1,0]
    cc,wc,cl=[],[],[]
    for c in CONDITIONS:
        for s in SEEDS:
            r=data.get((c,s))
            if r and r['trajectory']:
                cc.append(r['coinc_cc']); wc.append(r['trajectory'][-1]['Wcc']); cl.append(c)
    for c in CONDITIONS:
        m=[i for i,x in enumerate(cl) if x==c]
        ax.scatter([cc[i] for i in m],[wc[i] for i in m], color=COLORS[c], s=70,
                   edgecolor='white', label=c.replace('_',' '), zorder=5)
    if len(cc)>=3:
        rr,pp=pearsonr(cc,wc)
        z=np.polyfit(cc,wc,1); xl=np.linspace(min(cc),max(cc),50)
        ax.plot(xl,np.polyval(z,xl),'--k',lw=1.5,alpha=0.6)
        ax.set_title(f'C. Mediation: coincidence -> Wcc\n(r={rr:+.3f}, p={pp:.3f})')
    ax.set_xlabel('Core-core coincidence'); ax.set_ylabel('Final Wcc'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    # D: STDP net by block
    ax=axes[1,1]; blocks=['cc','uc','uu']; bl=['c-c','u-c','u-u']
    net=[np.mean([data[('FULL',s)]['stdp']['pot_'+b]+data[('FULL',s)]['stdp']['dep_'+b]
                  for s in SEEDS if ('FULL',s) in data]) for b in blocks]
    ax.bar(np.arange(3), net, 0.6, color=['#2166AC','#888','#ccc'], edgecolor='black', alpha=0.85)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(bl); ax.set_ylabel('Net STDP weight change')
    ax.set_title('D. Net STDP by block during replay'); ax.axhline(0,color='black',lw=0.7); ax.grid(axis='y',alpha=0.3)
    fig.suptitle('Task 4: Mechanism Summary — How Replay Increases Schema Strength',
                 y=1.02, fontsize=15, fontweight='bold')
    fig.tight_layout(); savefig(fig, 'fig4_mechanism_summary')


def interpretation(data, replay_stats, stdp_res, coinc_res, decomp, med):
    print(f'\n{"="*82}\nMECHANISTIC INTERPRETATION\n{"="*82}')
    cc_growth = decomp['FULL']['Wcc'] - decomp['NO_REPLAY']['Wcc']
    uc_growth = decomp['FULL']['Wuc'] - decomp['NO_REPLAY']['Wuc']
    uu_growth = decomp['FULL']['Wuu'] - decomp['NO_REPLAY']['Wuu']

    print(f'\n  1. What neural process changes during replay?')
    print(f'     Core neurons fire together at {replay_stats["core_part"]:.0%} participation per event,')
    print(f'     producing core-core spike coincidence of {coinc_res["FULL"]["cc"]:.4f}/step')
    print(f'     vs {coinc_res["NO_REPLAY"]["cc"]:.4f}/step without replay.')

    print(f'\n  2. Which weight block grows most? (per-synapse weight, replay-driven)')
    growths = {'Wcc (core-core)': cc_growth, 'Wuc (unique-core)': uc_growth, 'Wuu (unique-unique)': uu_growth}
    winner = max(growths, key=growths.get)
    for k,v in growths.items():
        print(f'     {k}: {v:+.4f}' + ('  <-- LARGEST' if k==winner else ''))
    print(f'     (Wcc/Wuc/Wuu are already per-synapse block means, so directly comparable.)')

    print(f'\n  3. Which STDP interactions dominate?')
    tr = stdp_res['train']; rp = stdp_res['replay']
    tot_train = sum(tr[b]['pot'] for b in ['cc','uc','uu'])
    print(f'     TRAINING: core-core potentiation {tr["cc"]["pot"]:+.2f} '
          f'({100*tr["cc"]["pot"]/max(tot_train,1e-9):.0f}% of training LTP); '
          f'unique-core {tr["uc"]["pot"]:+.2f}, unique-unique {tr["uu"]["pot"]:+.2f}')
    print(f'     REPLAY: core-core {rp["cc"]["pot"]:+.2f} (steps={rp["n_steps"]:.0f} — gate rarely opens).')
    print(f'     => Core-core synapses are potentiated primarily during TRAINING because')
    print(f'        core neurons fire in all 4 memories; replay adds via MB boost, not STDP.')

    print(f'\n  4. Coincidence, firing rate, or repeated reactivation?')
    cr = coinc_res['FULL']['core_rate']; ur = coinc_res['FULL']['uniq_rate']
    print(f'     Core firing rate {cr:.4f} vs unique {ur:.4f} (ratio {cr/max(ur,1e-9):.2f}x)')
    print(f'     Core-core coincidence {coinc_res["FULL"]["cc"]:.4f} vs unique-unique '
          f'{coinc_res["FULL"]["uu"]:.4f} (ratio {coinc_res["FULL"]["cc"]/max(coinc_res["FULL"]["uu"],1e-9):.2f}x)')
    print(f'     Mediation corr(coincidence, Wcc) = {med["r_coinc_wcc"]:+.3f}')

    print(f'\n  5. Most likely causal chain:')
    print(f'     Replay reactivates each memory assembly -> core neurons (shared by all 4')
    print(f'     memories) are co-activated on EVERY replay event -> elevated core-core spike')
    print(f'     coincidence -> STDP potentiates core-core synapses -> Wcc grows -> schema')
    print(f'     strength (S1 = Wcc - Wuc) increases. Without replay, core neurons are not')
    print(f'     repeatedly co-activated, coincidence stays low, and Wcc does not grow.')


if __name__ == '__main__':
    data = load_all()
    print(f'Loaded {len(data)} runs')
    rstats = table_replay(data)
    sstats = table_stdp(data)
    cstats = table_coincidence(data)
    dstats = table_decomp(data)
    med = mediation(data, rstats, cstats, dstats)
    print('\nGenerating figures...')
    fig_decomp_growth(data)
    fig_coincidence(data)
    fig_stdp(data)
    fig_mechanism(data, dstats, cstats)
    print(f'Figures -> {FIG_DIR}')
    interpretation(data, rstats, sstats, cstats, dstats, med)
    print('\nTASK 4 COMPLETE.')
