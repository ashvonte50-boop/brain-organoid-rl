"""
TASK 3 ANALYSIS — Schema Formation Dynamics
============================================
Loads Task 3 PKLs, builds trajectories, runs stats, generates 4 figures.
"""
import os, sys, pickle, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task3'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042]
CONDITIONS = ['FULL', 'NO_REPLAY']
COLORS = {'FULL': '#2166AC', 'NO_REPLAY': '#D6604D'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
})


def load_all():
    data = {}
    for cond in CONDITIONS:
        for s in SEEDS:
            p = os.path.join(OUT_DIR, f'T3_{cond}_seed{s}.pkl')
            if not os.path.exists(p): print(f'MISSING: {p}'); continue
            with open(p, 'rb') as f: data[(cond, s)] = pickle.load(f)
    return data


def build_traj_matrix(data, cond, metric):
    """Return (n_seeds × n_checkpoints) matrix and checkpoint labels."""
    # Collect all trajectories
    all_trajs = []
    for s in SEEDS:
        r = data.get((cond, s))
        if r is None: continue
        traj = r.get('trajectory', [])
        all_trajs.append(traj)
    if not all_trajs: return None, []

    # Align on label order from first seed
    labels = [t['label'] for t in all_trajs[0]]
    mat = np.full((len(all_trajs), len(labels)), np.nan)
    for i, traj in enumerate(all_trajs):
        lmap = {t['label']: t[metric] for t in traj}
        for k, lbl in enumerate(labels):
            if lbl in lmap: mat[i, k] = lmap[lbl]
    return mat, labels


def build_ret_traj(data, cond):
    """Return (n_seeds × n_stages) retention matrix."""
    mats = []
    for s in SEEDS:
        r = data.get((cond, s))
        if r is None: continue
        rt = r.get('ret_traj', [])
        if rt:
            mats.append([pt['retention_mean'] for pt in rt])
    if not mats: return None
    # Pad to same length
    maxlen = max(len(m) for m in mats)
    out = np.full((len(mats), maxlen), np.nan)
    for i, m in enumerate(mats): out[i, :len(m)] = m
    return out


def label_to_pct(labels):
    """Convert checkpoint labels to approximate % of training."""
    pcts = []
    for lbl in labels:
        if lbl == 'baseline': pcts.append(0.0)
        elif lbl == 'final':  pcts.append(100.0)
        elif 'post_encode' in lbl:
            j = int(lbl.split('_')[-1])
            pcts.append(12.5 + j * 25.0)
        elif 'post_replay' in lbl:
            j = int(lbl.split('_')[-1])
            pcts.append(25.0 + j * 25.0)
        else: pcts.append(np.nan)
    return np.array(pcts)


def print_trajectory_table(data):
    print(f'\n{"="*100}')
    print('TRAJECTORY TABLE: mean ± SEM per checkpoint per condition')
    print(f'{"="*100}')
    for metric, mlabel in [('S1', 'S1 = Wcc-Wuc'), ('Wcc', 'Wcc'), ('RS', 'RS (old)')]:
        print(f'\n  {mlabel}')
        mat_full, labels = build_traj_matrix(data, 'FULL', metric)
        mat_nr,   _      = build_traj_matrix(data, 'NO_REPLAY', metric)
        if mat_full is None: continue
        pcts = label_to_pct(labels)
        print(f'  {"Checkpoint":<20s} {"%":>5s} {"FULL mean":>10s} {"SEM":>8s} '
              f'{"NO_REPLAY mean":>14s} {"SEM":>8s} {"d":>6s} {"p":>8s}')
        print('  ' + '-'*90)
        for k, (lbl, pct) in enumerate(zip(labels, pcts)):
            f_vals = mat_full[:, k][np.isfinite(mat_full[:, k])]
            n_vals = mat_nr[:, k][np.isfinite(mat_nr[:, k])] if mat_nr is not None else np.array([])
            f_m = f_vals.mean() if len(f_vals) else np.nan
            f_se = f_vals.std(ddof=1)/np.sqrt(len(f_vals)) if len(f_vals)>1 else 0
            n_m = n_vals.mean() if len(n_vals) else np.nan
            n_se = n_vals.std(ddof=1)/np.sqrt(len(n_vals)) if len(n_vals)>1 else 0
            if len(f_vals)>=2 and len(n_vals)>=2:
                from scipy.stats import ttest_ind as _t
                t_s, p_s = _t(f_vals, n_vals, equal_var=False)
                pool = np.sqrt(((len(f_vals)-1)*f_vals.var(ddof=1)+(len(n_vals)-1)*n_vals.var(ddof=1))/(len(f_vals)+len(n_vals)-2))
                dd = (f_m - n_m)/pool if pool else np.nan
                sig = '***' if p_s<0.001 else '**' if p_s<0.01 else '*' if p_s<0.05 else 'n.s.'
                print(f'  {lbl:<20s} {pct:>5.1f} {f_m:>10.4f} {f_se:>8.4f} '
                      f'{n_m:>14.4f} {n_se:>8.4f} {dd:>+6.2f} {p_s:>7.3f} {sig}')
            else:
                print(f'  {lbl:<20s} {pct:>5.1f} {f_m:>10.4f} {f_se:>8.4f} '
                      f'{n_m:>14.4f} {n_se:>8.4f}')


def find_key_moments(data):
    print(f'\n{"="*80}')
    print('KEY EMERGENCE MOMENTS')
    print(f'{"="*80}')
    for metric, mlabel in [('S1', 'S1'), ('Wcc', 'Wcc')]:
        mat_full, labels = build_traj_matrix(data, 'FULL', metric)
        mat_nr, _ = build_traj_matrix(data, 'NO_REPLAY', metric)
        if mat_full is None: continue
        pcts = label_to_pct(labels)
        baseline_vals = mat_full[:, 0][np.isfinite(mat_full[:, 0])]
        final_vals    = mat_full[:, -1][np.isfinite(mat_full[:, -1])]
        final_mean = final_vals.mean() if len(final_vals) else 1.0
        threshold_50 = 0.5 * final_mean

        print(f'\n  {mlabel}:')
        # a) First checkpoint where S1 > baseline (using t-test vs baseline_vals)
        for k, (lbl, pct) in enumerate(zip(labels, pcts)):
            if k == 0: continue
            v = mat_full[:, k][np.isfinite(mat_full[:, k])]
            if len(v) >= 2 and len(baseline_vals) >= 2:
                _, p = ttest_ind(v, baseline_vals, equal_var=False)
                if p < 0.05 and v.mean() > baseline_vals.mean():
                    print(f'    a) First sig > baseline: {lbl} ({pct:.0f}%) '
                          f'mean={v.mean():.4f} p={p:.3f}')
                    break

        # b) First checkpoint where FULL > NO_REPLAY significantly
        for k, (lbl, pct) in enumerate(zip(labels, pcts)):
            if mat_nr is None: break
            fv = mat_full[:, k][np.isfinite(mat_full[:, k])]
            nv = mat_nr[:, k][np.isfinite(mat_nr[:, k])]
            if len(fv)>=2 and len(nv)>=2:
                _, p = ttest_ind(fv, nv, equal_var=False)
                if p < 0.05 and fv.mean() > nv.mean():
                    print(f'    b) FULL first exceeds NO_REPLAY: {lbl} ({pct:.0f}%) p={p:.3f}')
                    break

        # c) Time-to-schema: first checkpoint where FULL mean > 50% of final
        for k, (lbl, pct) in enumerate(zip(labels, pcts)):
            v = mat_full[:, k][np.isfinite(mat_full[:, k])]
            if len(v) and v.mean() >= threshold_50:
                print(f'    c) Exceeds 50% of final value: {lbl} ({pct:.0f}%) '
                      f'mean={v.mean():.4f} threshold={threshold_50:.4f}')
                break


def savefig(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}')


def plot_traj(data, metric, ylabel, title, figname, include_ret=False):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cond in CONDITIONS:
        mat, labels = build_traj_matrix(data, cond, metric)
        if mat is None: continue
        pcts = label_to_pct(labels)
        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(mat), axis=0))
        c = COLORS[cond]
        ax.plot(pcts, mean, 'o-', color=c, lw=2.5, ms=7, label=cond.replace('_', ' '))
        ax.fill_between(pcts, mean-sem, mean+sem, color=c, alpha=0.18)

    # Mark memory boundaries
    for pct, mem in [(12.5, 'A↓'), (37.5, 'B↓'), (62.5, 'C↓'), (87.5, 'D↓')]:
        ax.axvline(pct, color='grey', ls=':', lw=0.8, alpha=0.5)
        ax.text(pct+0.5, ax.get_ylim()[1]*0.97, mem, fontsize=8, color='grey')

    ax.set_xlabel('Training progress (%)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title)
    ax.set_xlim(-3, 103)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, figname)


def plot_master(data):
    """4-panel figure: S1, Wcc, Retention, RS."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    panels = [
        ('S1',  'S1 = Wcc - Wuc', 'A. Schema Asymmetry (S1)',  axes[0,0], False),
        ('Wcc', 'Wcc',             'B. Core Weight Magnitude',  axes[0,1], False),
        (None,  'Retention',       'C. Memory Retention',       axes[1,0], True),
        ('RS',  'RS (old)',        'D. RS — scale-invariant\n[shown for comparison]', axes[1,1], False),
    ]

    for metric, ylabel, title, ax, is_ret in panels:
        for cond in CONDITIONS:
            c = COLORS[cond]
            if is_ret:
                mat = build_ret_traj(data, cond)
                if mat is None: continue
                n_stages = mat.shape[1]
                stage_pcts = np.linspace(12.5, 100, n_stages)
                mean = np.nanmean(mat, axis=0)
                sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(mat), axis=0))
                ax.plot(stage_pcts, mean, 'o-', color=c, lw=2.5, ms=7, label=cond.replace('_',' '))
                ax.fill_between(stage_pcts, mean-sem, mean+sem, color=c, alpha=0.18)
            else:
                mat, labels = build_traj_matrix(data, cond, metric)
                if mat is None: continue
                pcts = label_to_pct(labels)
                mean = np.nanmean(mat, axis=0)
                sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(mat), axis=0))
                ax.plot(pcts, mean, 'o-', color=c, lw=2.5, ms=7, label=cond.replace('_',' '))
                ax.fill_between(pcts, mean-sem, mean+sem, color=c, alpha=0.18)

            # Stat annotations at each checkpoint
            if not is_ret and metric in ('S1', 'Wcc'):
                mat_f, labels_f = build_traj_matrix(data, 'FULL', metric)
                mat_n, _        = build_traj_matrix(data, 'NO_REPLAY', metric)
                if mat_f is not None and mat_n is not None and cond == 'FULL':
                    pcts_f = label_to_pct(labels_f)
                    for k in range(mat_f.shape[1]):
                        fv = mat_f[:, k][np.isfinite(mat_f[:, k])]
                        nv = mat_n[:, k][np.isfinite(mat_n[:, k])]
                        if len(fv)>=2 and len(nv)>=2:
                            _, p = ttest_ind(fv, nv, equal_var=False)
                            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
                            if sig:
                                yp = np.nanmean(mat_f[:, k]) + np.nanstd(mat_f[:, k], ddof=1)/np.sqrt(len(fv))
                                ax.text(pcts_f[k], yp*1.08, sig, ha='center', fontsize=9,
                                        fontweight='bold', color='black')

        for pct in [12.5, 37.5, 62.5, 87.5]:
            ax.axvline(pct, color='grey', ls=':', lw=0.7, alpha=0.4)
        ax.set_xlabel('Training progress (%)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title)
        ax.set_xlim(-3, 103)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Task 3: Schema Formation Dynamics (n=5 seeds, mean ± SEM)\n'
                 'Vertical dashed lines = memory boundaries (A/B/C/D)\n'
                 '* p<0.05  ** p<0.01  *** p<0.001  (FULL vs NO_REPLAY)',
                 y=1.02, fontsize=14, fontweight='bold')
    fig.tight_layout()
    savefig(fig, 'fig4_full_vs_noreplay_master')


def final_report(data):
    print(f'\n{"="*80}')
    print('TASK 3 — FINAL REPORT')
    print(f'{"="*80}')

    mat_f_s1, labels = build_traj_matrix(data, 'FULL',      'S1')
    mat_n_s1, _      = build_traj_matrix(data, 'NO_REPLAY', 'S1')
    mat_f_wcc, _     = build_traj_matrix(data, 'FULL',      'Wcc')
    mat_n_wcc, _     = build_traj_matrix(data, 'NO_REPLAY', 'Wcc')
    pcts = label_to_pct(labels)

    # Final values
    fs1  = np.nanmean(mat_f_s1[:, -1]);  ns1  = np.nanmean(mat_n_s1[:, -1])
    fwcc = np.nanmean(mat_f_wcc[:, -1]); nwcc = np.nanmean(mat_n_wcc[:, -1])

    # Retention
    ret_f = np.array([data[('FULL',s)]['retention_mean'] for s in SEEDS if ('FULL',s) in data])
    ret_n = np.array([data[('NO_REPLAY',s)]['retention_mean'] for s in SEEDS if ('NO_REPLAY',s) in data])

    # Separation point (first checkpoint where FULL significantly > NO_REPLAY)
    sep_label, sep_pct = 'not found', np.nan
    for k, (lbl, pct) in enumerate(zip(labels, pcts)):
        fv = mat_f_s1[:, k][np.isfinite(mat_f_s1[:, k])]
        nv = mat_n_s1[:, k][np.isfinite(mat_n_s1[:, k])]
        if len(fv)>=2 and len(nv)>=2:
            _, p = ttest_ind(fv, nv, equal_var=False)
            if p < 0.05 and fv.mean() > nv.mean():
                sep_label, sep_pct = lbl, pct
                break

    print(f'\n  FULL  final S1 = {fs1:.4f}  Wcc = {fwcc:.4f}  Ret = {ret_f.mean():.4f}')
    print(f'  NO_REPLAY final S1 = {ns1:.4f}  Wcc = {nwcc:.4f}  Ret = {ret_n.mean():.4f}')
    print(f'\n  Q1: Does replay accelerate schema formation?')
    if not np.isnan(sep_pct):
        print(f'      YES. FULL first exceeds NO_REPLAY at {sep_label} ({sep_pct:.0f}% training).')
        print(f'      By end of training: FULL S1 = {fs1:.4f}, NO_REPLAY S1 = {ns1:.4f} '
              f'({100*(fs1-ns1)/max(ns1,1e-9):.0f}% higher with replay)')
    else:
        print(f'      Separation not detected at p<0.05 (n=5 seeds).')

    # Gradual vs abrupt: slope of S1 across checkpoints
    means_s1 = np.nanmean(mat_f_s1, axis=0)
    diffs = np.diff(means_s1)
    max_jump_idx = np.argmax(np.abs(diffs))
    max_jump = diffs[max_jump_idx]
    total_rise = means_s1[-1] - means_s1[0]
    print(f'\n  Q2: When does schema first emerge?')
    print(f'      Largest single jump: {labels[max_jump_idx]} → {labels[max_jump_idx+1]} '
          f'(ΔS1 = {max_jump:+.4f}, {100*max_jump/total_rise:.0f}% of total rise)')
    print(f'      Total rise from baseline to final: {total_rise:.4f}')

    print(f'\n  Q3: Gradual or abrupt?')
    # Check if >50% of rise happens in first 50% of training
    mid_k = sum(1 for p in pcts if p <= 50)
    rise_first_half = means_s1[min(mid_k, len(means_s1)-1)] - means_s1[0]
    print(f'      Rise in first 50% of training: {rise_first_half:.4f} '
          f'({100*rise_first_half/max(total_rise,1e-9):.0f}% of total)')
    if rise_first_half/max(total_rise,1e-9) > 0.7:
        print(f'      → Mostly FRONT-LOADED: schema forms early, saturates later.')
    elif rise_first_half/max(total_rise,1e-9) < 0.4:
        print(f'      → Mostly BACK-LOADED: schema forms late in training.')
    else:
        print(f'      → GRADUAL: schema rises steadily across training.')

    print(f'\n  Q4: Is schema growth coupled to retention growth?')
    # Cross-checkpoint correlation between S1 and retention
    s1_trace = np.nanmean(mat_f_s1, axis=0)
    ret_trace = build_ret_traj(data, 'FULL')
    if ret_trace is not None:
        ret_mean = np.nanmean(ret_trace, axis=0)
        # Align: retention has n_mem checkpoints, S1 has more
        from scipy.stats import pearsonr as _pr
        min_len = min(len(s1_trace), len(ret_mean))
        r_val, p_val = _pr(s1_trace[:min_len], ret_mean[:min_len])
        print(f'      Cross-checkpoint correlation S1 vs Retention: r={r_val:+.3f} p={p_val:.3f}')
        if abs(r_val) > 0.8:
            print(f'      → STRONGLY COUPLED: schema strength and retention grow together.')
        elif abs(r_val) > 0.5:
            print(f'      → MODERATELY COUPLED.')
        else:
            print(f'      → DECOUPLED: schema and retention grow independently.')


if __name__ == '__main__':
    data = load_all()
    print(f'Loaded {len(data)} runs')

    print_trajectory_table(data)
    find_key_moments(data)

    print('\nGenerating figures...')
    plot_traj(data, 'S1',  'S1 = Wcc-Wuc', 'Schema Asymmetry (S1) vs Training Progress',
              'fig1_schema_growth')
    plot_traj(data, 'Wcc', 'Wcc',           'Core Weight Magnitude vs Training Progress',
              'fig2_wcc_growth')

    # Retention figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cond in CONDITIONS:
        mat = build_ret_traj(data, cond)
        if mat is None: continue
        pcts = np.linspace(12.5, 100, mat.shape[1])
        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0, ddof=1)/np.sqrt(np.sum(np.isfinite(mat), axis=0))
        ax.plot(pcts, mean, 'o-', color=COLORS[cond], lw=2.5, ms=7,
                label=cond.replace('_',' '))
        ax.fill_between(pcts, mean-sem, mean+sem, color=COLORS[cond], alpha=0.18)
    ax.set_xlabel('Training progress (%)', fontweight='bold')
    ax.set_ylabel('Retention (mean across trained memories)', fontweight='bold')
    ax.set_title('Memory Retention vs Training Progress')
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    savefig(fig, 'fig3_retention_growth')

    plot_master(data)
    print(f'Figures → {FIG_DIR}')

    final_report(data)
    print('\nTASK 3 COMPLETE.')
