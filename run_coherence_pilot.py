"""
COHERENCE PILOT — 5 runs, ~40 min
===================================
FULL model only, 1 seed each, varying REPLAY_COHERENCE_THR
across the observed coherence range (max = 0.078).

If curves are flat → coherence hypothesis wrong, stop.
If curves change → justify full multi-seed sweep.

COH_THR values: [0.00, 0.02, 0.04, 0.06, 0.08]
(0.50 already known: mechanisms dormant, skipped)
"""
import os, sys, time, json, subprocess, pickle
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\coherence_sweep'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
os.makedirs(OUT_DIR, exist_ok=True)

COH_THRS = [0.00, 0.02, 0.04, 0.06, 0.08]
SEED     = 42   # single seed


def run_one(coh_thr):
    chk = os.path.join(OUT_DIR, f'pilot_FULL_coh{coh_thr:.2f}.pkl')
    log = chk.replace('.pkl', '.log')

    if os.path.exists(chk):
        print(f'  coh_thr={coh_thr:.2f} — loaded from cache', flush=True)
        with open(chk, 'rb') as f: return pickle.load(f)

    cmd = [sys.executable, 'ablation_single_seed.py',
           'FULL', '0', str(SEED), '{}',
           '--prefix', f'PILOT_coh{coh_thr:.2f}',
           '--coh_thr', str(coh_thr)]
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}

    print(f'  coh_thr={coh_thr:.2f} running...', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, env=env, cwd=WORK_DIR,
                              stdout=lf, stderr=subprocess.STDOUT)
    elapsed = int(time.time() - t0)

    # Read actual checkpoint saved by worker (prefix-based naming)
    worker_chk = os.path.join(r'C:\Users\Admin\brain-organoid-rl\ablation_results',
                              f'PILOT_coh{coh_thr:.2f}_FULL_seed0.pkl')
    if proc.returncode == 0 and os.path.exists(worker_chk):
        with open(worker_chk, 'rb') as f: res = pickle.load(f)
        with open(chk, 'wb') as f: pickle.dump(res, f)
        n = res.get('natural', {})
        print(f'  coh_thr={coh_thr:.2f}  {elapsed}s  '
              f'DAI={n.get("dai_core",0):.4f}  '
              f'RS={n.get("real_schema",0):.4f}  '
              f'Ret={n.get("retention_mean",0):.4f}', flush=True)
        return res
    else:
        print(f'  coh_thr={coh_thr:.2f}  FAILED (exit={proc.returncode})', flush=True)
        try:
            print('  log tail:', open(log).readlines()[-3:], flush=True)
        except: pass
        return None


def print_summary(results):
    print(f'\n{"="*65}', flush=True)
    print('COHERENCE PILOT SUMMARY', flush=True)
    print(f'{"="*65}', flush=True)
    print(f'  {"COH_THR":>8}  {"DAI_core":>10}  {"REAL_SCHEMA":>12}  {"Retention":>10}', flush=True)
    print('  ' + '-'*46, flush=True)
    dai_vals, rs_vals = [], []
    for coh_thr, res in results:
        if res is None:
            print(f'  {coh_thr:>8.2f}  {"FAILED":>10}', flush=True)
            continue
        n = res.get('natural', {})
        dai = n.get('dai_core',0); rs = n.get('real_schema',0); ret = n.get('retention_mean',0)
        dai_vals.append(dai); rs_vals.append(rs)
        print(f'  {coh_thr:>8.2f}  {dai:>10.4f}  {rs:>12.4f}  {ret:>10.4f}', flush=True)

    if len(dai_vals) >= 2:
        dai_range = max(dai_vals) - min(dai_vals)
        rs_range  = max(rs_vals)  - min(rs_vals)
        print(f'\n  DAI range across thresholds: {dai_range:.4f}', flush=True)
        print(f'  RS  range across thresholds: {rs_range:.4f}', flush=True)
        print(flush=True)
        if dai_range < 0.005 and rs_range < 0.01:
            print('VERDICT: FLAT — curves do not change with COH_THR.', flush=True)
            print('=> Mechanism activation does NOT affect schema metrics.', flush=True)
            print('=> The coherence-gated mechanisms are truly inert.', flush=True)
            print('=> Only MB (core boost) drives schema formation.', flush=True)
            print('=> Do NOT run the 60-run sweep. Use existing data.', flush=True)
        else:
            print('VERDICT: SIGNAL — curves change with COH_THR.', flush=True)
            print('=> At lower thresholds, mechanisms activate and affect metrics.', flush=True)
            print(f'=> ΔDAI = {dai_range:.4f}  ΔRS = {rs_range:.4f}', flush=True)
            print('=> Justify the multi-seed sweep at informative threshold values.', flush=True)


def make_figure(results):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.titlesize':13,'axes.titleweight':'bold',
                         'axes.spines.top':False,'axes.spines.right':False})

    thrs, dai_v, rs_v, ret_v = [], [], [], []
    for coh_thr, res in results:
        if res is None: continue
        n = res.get('natural', {})
        thrs.append(coh_thr)
        dai_v.append(n.get('dai_core', 0))
        rs_v.append(n.get('real_schema', 0))
        ret_v.append(n.get('retention_mean', 0))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, vals, label in zip(axes,
            [dai_v, rs_v, ret_v],
            ['DAI$_{core}$', 'REAL_SCHEMA', 'Retention (mean)']):
        ax.plot(thrs, vals, 'o-', color='#2166AC', lw=2.5, ms=9,
                markerfacecolor='white', markeredgewidth=2.5)
        # Mark observed coherence max
        ax.axvline(0.078, color='#D73027', lw=1.2, ls='--', alpha=0.8,
                   label='Max observed coh (0.078)')
        ax.set_xlabel('REPLAY_COHERENCE_THR', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} vs Coherence Gate')
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=9)

    fig.suptitle('Coherence Threshold Pilot (FULL model, n=1 seed)\n'
                 'Does lowering the gate change schema metrics?',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
             'Red dashed line = max observed coherence (0.078). '
             'Flat curve = mechanisms inert. Sloped curve = mechanisms matter.',
             ha='center', fontsize=9, style='italic')
    fig.tight_layout()
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(os.path.join(fig_dir, f'coherence_pilot.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {os.path.join(fig_dir, "coherence_pilot.[pdf|svg|png]")}', flush=True)


if __name__ == '__main__':
    print('COHERENCE PILOT — 5 runs, 1 seed each (~40 min)', flush=True)
    print(f'Seed: {SEED}  COH_THR values: {COH_THRS}', flush=True)
    print('Question: does lowering COH_THR change DAI/RS?', flush=True)
    print(flush=True)

    t0 = time.time()
    results = []
    for coh_thr in COH_THRS:
        res = run_one(coh_thr)
        results.append((coh_thr, res))

    print_summary(results)
    make_figure(results)

    elapsed = time.time() - t0
    print(f'\nPilot complete in {elapsed:.0f}s ({elapsed/60:.1f} min)', flush=True)
