"""
COHERENCE THRESHOLD SWEEP
=========================

Sweeps REPLAY_COHERENCE_THR across the observed coherence range
to find the phase transition point where M1-M10 mechanisms activate.

Design:
  COH_THR values : [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.50]
  Conditions     : FULL  vs  -M5 (drift=False)
  Seeds          : 3 per cell
  Total          : 8 × 2 × 3 = 48 seeds  (~6.4 hr)

  Fast pilot (--fast): [0.0, 0.02, 0.05, 0.08, 0.50] × [FULL, M5] × 3 seeds
                        5 × 2 × 3 = 30 seeds  (~4 hr)

Key question: at what COH_THR does ΔDAI (FULL - M5) become nonzero?
              Is the transition sharp (phase-like) or gradual?

Usage:
  python run_coherence_sweep.py          # fast pilot
  python run_coherence_sweep.py --full   # complete sweep
  python run_coherence_sweep.py --resume
  python run_coherence_sweep.py --figures-only
"""
import os, sys, time, argparse, pickle, json, subprocess
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import numpy as np

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\coherence_sweep'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

BASE_SEED = 42
N_SEEDS   = 3

# Threshold values — dense near observed coherence max (0.078)
COH_THR_FAST = [0.0, 0.02, 0.05, 0.08, 0.50]
COH_THR_FULL = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.50]

CONDITIONS = {
    'FULL': {},
    'M5':   {'drift': False},
    'M1':   {'overlap_penalty': False},
    'M10':  {'reconsol': False},
}


def cell_key(coh_thr, cname):
    return f'coh{coh_thr:.3f}_{cname}'


def chk_path(coh_thr, cname, si):
    k = cell_key(coh_thr, cname)
    return os.path.join(OUT_DIR, f'{k}_seed{si}.pkl')


def cell_done(coh_thr, cname):
    return os.path.join(OUT_DIR, f'{cell_key(coh_thr, cname)}_DONE.pkl')


def run_cell(coh_thr, cname, abl_dict, n_seeds=N_SEEDS, resume=False):
    done_path = cell_done(coh_thr, cname)
    if resume and os.path.exists(done_path):
        with open(done_path, 'rb') as f:
            return pickle.load(f)

    results = []
    for si in range(n_seeds):
        chk = chk_path(coh_thr, cname, si)
        if resume and os.path.exists(chk):
            with open(chk, 'rb') as f:
                results.append(pickle.load(f))
            print(f'    seed{si} loaded from checkpoint', flush=True)
            continue

        seed = BASE_SEED + si * 1000
        log  = chk.replace('.pkl', '.log')
        cmd  = [sys.executable, 'ablation_single_seed.py',
                cname, str(si), str(seed), json.dumps(abl_dict),
                '--prefix', f'COH{coh_thr:.3f}',
                '--coh_thr', str(coh_thr)]
        env  = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}

        t0 = time.time()
        with open(log, 'w', encoding='utf-8') as lf:
            proc = subprocess.run(cmd, env=env, cwd=WORK_DIR,
                                  stdout=lf, stderr=subprocess.STDOUT)
        elapsed = int(time.time() - t0)

        if proc.returncode == 0 and os.path.exists(chk):
            with open(chk, 'rb') as f:
                res = pickle.load(f)
            results.append(res)
            n = res.get('natural', {})
            print(f'    seed{si}({seed})  {elapsed:4d}s  '
                  f'DAI={n.get("dai_core",0):.4f}  '
                  f'RS={n.get("real_schema",0):.4f}', flush=True)
        else:
            # Retry once
            with open(log, 'a', encoding='utf-8') as lf:
                proc2 = subprocess.run(cmd, env=env, cwd=WORK_DIR,
                                       stdout=lf, stderr=subprocess.STDOUT)
            if proc2.returncode == 0 and os.path.exists(chk):
                with open(chk, 'rb') as f:
                    res = pickle.load(f)
                results.append(res)
            else:
                print(f'    seed{si} FAILED (exit={proc.returncode})', flush=True)

    with open(done_path, 'wb') as f:
        pickle.dump(results, f)
    return results


def run_sweep(coh_thrs, n_seeds=N_SEEDS, resume=False):
    print('=' * 65, flush=True)
    print('COHERENCE THRESHOLD SWEEP', flush=True)
    print(f'COH_THR values: {coh_thrs}', flush=True)
    print(f'Conditions: {list(CONDITIONS)}  n_seeds={n_seeds}', flush=True)
    print('=' * 65, flush=True)

    all_results = {}
    for coh_thr in coh_thrs:
        print(f'\n--- COH_THR = {coh_thr} ---', flush=True)
        for cname, abl_dict in CONDITIONS.items():
            print(f'  [{cname}]', flush=True)
            res = run_cell(coh_thr, cname, abl_dict, n_seeds=n_seeds, resume=resume)
            all_results[(coh_thr, cname)] = res

    pkl_path = os.path.join(OUT_DIR, 'sweep_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'results': all_results, 'coh_thrs': coh_thrs,
                     'conditions': list(CONDITIONS)}, f)
    print(f'\nSaved: {pkl_path}', flush=True)
    return all_results


def summarise(all_results, coh_thrs):
    print(f'\n{"=" * 70}', flush=True)
    print('SWEEP SUMMARY', flush=True)
    print(f'{"=" * 70}', flush=True)
    print(f'{"COH_THR":>8}  {"Cond":>6}  {"DAI":>8}  {"RS":>8}  {"n":>3}  '
          f'{"ΔDAI(vs FULL)":>14}', flush=True)
    print('-' * 60, flush=True)
    for coh_thr in coh_thrs:
        full_res = all_results.get((coh_thr, 'FULL'), [])
        full_dai = np.mean([s.get('natural', {}).get('dai_core', 0) for s in full_res]) if full_res else np.nan
        for cname in CONDITIONS:
            res = all_results.get((coh_thr, cname), [])
            if not res:
                continue
            dai_vals = [s.get('natural', {}).get('dai_core', 0) for s in res]
            rs_vals  = [s.get('natural', {}).get('real_schema', 0) for s in res]
            m_dai = np.mean(dai_vals)
            m_rs  = np.mean(rs_vals)
            delta = (m_dai - full_dai) if cname != 'FULL' else 0.0
            print(f'  {coh_thr:>6.3f}  {cname:>6}  {m_dai:>8.4f}  {m_rs:>8.4f}  '
                  f'{len(res):>3}  {delta:>+14.4f}', flush=True)
    print(flush=True)


def make_figures(all_results, coh_thrs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'axes.labelsize': 12, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.dpi': 150,
    })

    FIG_DIR = os.path.join(OUT_DIR, 'figures')
    os.makedirs(FIG_DIR, exist_ok=True)

    metrics = [
        ('dai_core',    'DAI$_{core}$'),
        ('real_schema', 'REAL_SCHEMA'),
        ('distortion',  'Distortion'),
    ]
    cond_colors = {'FULL': '#2166AC', 'M5': '#D73027',
                   'M1': '#E8601C', 'M10': '#F4A736'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for cname in CONDITIONS:
            means, sems, xs = [], [], []
            for coh_thr in coh_thrs:
                res = all_results.get((coh_thr, cname), [])
                vals = [s.get('natural', {}).get(metric, 0) for s in res]
                if vals:
                    xs.append(coh_thr)
                    means.append(np.mean(vals))
                    sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            c = cond_colors.get(cname, '#888888')
            lw = 2.5 if cname == 'FULL' else 1.8
            ax.plot(xs, means, 'o-', color=c, lw=lw, label=cname, ms=7)
            ax.fill_between(xs,
                            [m-s for m,s in zip(means,sems)],
                            [m+s for m,s in zip(means,sems)],
                            color=c, alpha=0.15)

        # Mark threshold region (observed coherence max = 0.078)
        ax.axvspan(0, 0.08, alpha=0.07, color='green',
                   label='Observed coh range')
        ax.axvline(0.08, color='green', lw=1.0, ls=':', alpha=0.7)
        ax.set_xlabel('REPLAY_COHERENCE_THR', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{ylabel} vs Coherence Threshold')
        ax.grid(alpha=0.3)

    axes[0].legend(loc='best', fontsize=9, framealpha=0.85)

    fig.suptitle('Coherence Threshold Sweep\n'
                 'Does schema formation depend on mechanism activation?',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.02,
             f'Green band = observed coherence range (0–0.078). '
             f'At COH_THR=0.5 (current), mechanisms never fire. '
             f'n={N_SEEDS} seeds per cell.',
             ha='center', fontsize=9, style='italic')
    fig.tight_layout()

    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'coherence_sweep.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Fig 2: ΔDAI heatmap (COH_THR × condition)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    abl_conds = [c for c in CONDITIONS if c != 'FULL']
    delta_mat  = np.zeros((len(abl_conds), len(coh_thrs)))
    sig_mat    = np.full((len(abl_conds), len(coh_thrs)), '', dtype=object)

    for ci, cname in enumerate(abl_conds):
        for ti, coh_thr in enumerate(coh_thrs):
            full_vals = [s.get('natural', {}).get('dai_core', 0)
                         for s in all_results.get((coh_thr, 'FULL'), [])]
            abl_vals  = [s.get('natural', {}).get('dai_core', 0)
                         for s in all_results.get((coh_thr, cname), [])]
            if full_vals and abl_vals:
                delta_mat[ci, ti] = np.mean(abl_vals) - np.mean(full_vals)
                if len(full_vals) > 1 and len(abl_vals) > 1:
                    _, p = ttest_ind(full_vals, abl_vals, equal_var=False)
                    sig_mat[ci, ti] = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

    from matplotlib.colors import TwoSlopeNorm
    vabs = max(abs(delta_mat).max(), 0.01)
    im = ax2.imshow(delta_mat, cmap='RdBu_r', vmin=-vabs, vmax=vabs, aspect='auto')
    for i in range(len(abl_conds)):
        for j in range(len(coh_thrs)):
            txt = f'{delta_mat[i,j]:+.3f}\n{sig_mat[i,j]}'
            ax2.text(j, i, txt, ha='center', va='center',
                     fontsize=8.5, fontweight='bold',
                     color='white' if abs(delta_mat[i,j]) > 0.7*vabs else 'black')
    ax2.set_xticks(range(len(coh_thrs)))
    ax2.set_xticklabels([f'{t:.3f}' for t in coh_thrs], fontsize=9)
    ax2.set_yticks(range(len(abl_conds)))
    ax2.set_yticklabels(abl_conds, fontsize=10)
    ax2.set_xlabel('REPLAY_COHERENCE_THR', fontsize=11, fontweight='bold')
    ax2.set_title('ΔDAI_core (Ablated − Full)\nBlue = ablation hurts, Red = ablation helps',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='ΔDAI_core', shrink=0.8)
    fig2.tight_layout()
    for ext in ('pdf', 'svg', 'png'):
        fig2.savefig(os.path.join(FIG_DIR, f'coherence_sweep_delta.{ext}'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Figures saved: {FIG_DIR}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full',         action='store_true', help='Full sweep (8 COH_THR values)')
    parser.add_argument('--resume',       action='store_true')
    parser.add_argument('--figures-only', action='store_true')
    parser.add_argument('--seeds',        type=int, default=N_SEEDS)
    args = parser.parse_args()

    coh_thrs = COH_THR_FULL if args.full else COH_THR_FAST

    if args.figures_only:
        pkl = os.path.join(OUT_DIR, 'sweep_results.pkl')
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        make_figures(data['results'], data['coh_thrs'])
        return

    t0 = time.time()
    all_results = run_sweep(coh_thrs, n_seeds=args.seeds, resume=args.resume)
    summarise(all_results, coh_thrs)
    make_figures(all_results, coh_thrs)
    elapsed = time.time() - t0
    print(f'\nSweep complete in {elapsed:.0f}s ({elapsed/3600:.2f} hr)', flush=True)
    print(f'Results: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
