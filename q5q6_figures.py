"""
Q5 + Q6 Figure Generator
=========================
Run AFTER q5_dose_response.py and q6_param_sweep.py complete.
Generates:
  - Fig Q5: Dose-response curve (M0 suppression vs M0 retention)
  - Fig Q6: Parameter sensitivity 4-panel figure
"""
import numpy as np
import pickle, os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = Path(r'C:\Users\Admin\brain-organoid-rl\ablation_results\q_fixes')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Q5: DOSE-RESPONSE FIGURE
# ═══════════════════════════════════════════════════════════════════════════
Q5_PKL = Path(r'C:\Users\Admin\brain-organoid-rl\ablation_results\q5_dose\Q5_dose_response.pkl')
if Q5_PKL.exists():
    with open(Q5_PKL, 'rb') as f:
        q5 = pickle.load(f)

    bias_levels = []
    m0_replays = []
    m0_ret = []
    mean_ret = []

    for bias_m0 in [1.0, 0.5, 0.2, 0.1, 0.0]:
        cond = f'bias_{bias_m0:.1f}'
        if cond in q5:
            d = q5[cond]
            bias_levels.append(bias_m0)
            m0_replays.append(d['replay_counts'][0])
            m0_ret.append(d['retention'][0])
            mean_ret.append(np.mean(d['retention']))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Bias vs M0 replay count
    ax = axes[0]
    ax.plot(bias_levels, m0_replays, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.set_xlabel('M0 Bias Weight', fontsize=12)
    ax.set_ylabel('M0 Replay Count', fontsize=12)
    ax.set_title('A. Replay allocation', fontsize=13, fontweight='bold')
    ax.invert_xaxis()
    ax.set_xticks(bias_levels)

    # Panel B: Bias vs M0 retention
    ax = axes[1]
    ax.plot(bias_levels, m0_ret, 'o-', color='#F44336', linewidth=2, markersize=8)
    ax.set_xlabel('M0 Bias Weight', fontsize=12)
    ax.set_ylabel('M0 Retention', fontsize=12)
    ax.set_title('B. M0 retention dose-response', fontsize=13, fontweight='bold')
    ax.invert_xaxis()
    ax.set_xticks(bias_levels)

    # Panel C: M0 replays vs M0 retention
    ax = axes[2]
    ax.scatter(m0_replays, m0_ret, c='#4CAF50', s=100, zorder=5, edgecolors='black')
    for i, b in enumerate(bias_levels):
        ax.annotate(f'bias={b}', (m0_replays[i], m0_ret[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel('M0 Replay Count', fontsize=12)
    ax.set_ylabel('M0 Retention', fontsize=12)
    ax.set_title('C. Replays predict retention', fontsize=13, fontweight='bold')

    # Add regression line if enough points
    if len(m0_replays) >= 3:
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(m0_replays, m0_ret)
        x_fit = np.linspace(min(m0_replays), max(m0_replays), 100)
        ax.plot(x_fit, slope * x_fit + intercept, '--', color='gray', alpha=0.7)
        ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.4f}', transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig_path = OUT_DIR / 'fig_Q5_dose_response.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(fig_path).replace('.png', '.pdf'), bbox_inches='tight')
    fig.savefig(str(fig_path).replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    print(f'Q5 figure saved: {fig_path}')

    # Print summary
    print('\nQ5 DOSE-RESPONSE SUMMARY:')
    print(f'{"Bias M0":<10} {"M0 replays":<12} {"M0 ret":<10} {"Mean ret":<10}')
    for i, b in enumerate(bias_levels):
        print(f'{b:<10.1f} {m0_replays[i]:<12} {m0_ret[i]:<10.4f} {mean_ret[i]:<10.4f}')
else:
    print(f'Q5 PKL not found at {Q5_PKL} -- run q5_dose_response.py first')

# ═══════════════════════════════════════════════════════════════════════════
# Q6: PARAMETER SENSITIVITY FIGURE
# ═══════════════════════════════════════════════════════════════════════════
Q6_PKL = Path(r'C:\Users\Admin\brain-organoid-rl\ablation_results\q6_sweep\Q6_sweep_results.pkl')
if Q6_PKL.exists():
    with open(Q6_PKL, 'rb') as f:
        q6 = pickle.load(f)

    BASELINE = {'gamma': 0.65, 'tau_slow': 4000, 'core_size': 20, 'w_max': 1.5}
    BASELINE_RET = 0.3064

    SWEEPS = {
        'gamma':     [0.3, 0.5, 0.65, 0.8, 0.95],
        'tau_slow':  [500, 2000, 4000, 8000, 16000],
        'core_size': [5, 10, 20, 30],
        'w_max':     [0.5, 1.0, 1.5, 2.0, 3.0],
    }

    LABELS = {
        'gamma': r'$\gamma$ (slow weight mixing)',
        'tau_slow': r'$\tau_{slow}$ (consolidation time constant)',
        'core_size': 'Schema core size (neurons)',
        'w_max': r'$W_{max}$ (weight ceiling)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (param, values) in enumerate(SWEEPS.items()):
        ax = axes[idx]
        sweep_vals = []
        sweep_ret = []

        for val in values:
            if val == BASELINE[param]:
                sweep_vals.append(val)
                sweep_ret.append(BASELINE_RET)
            else:
                cond = f'{param}_{val}'
                if cond in q6:
                    sweep_vals.append(val)
                    sweep_ret.append(q6[cond]['retention'])

        ax.plot(sweep_vals, sweep_ret, 'o-', color='#2196F3', linewidth=2, markersize=8)
        # Mark baseline
        ax.axvline(x=BASELINE[param], color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=BASELINE_RET, color='gray', linestyle='--', alpha=0.5)
        ax.plot(BASELINE[param], BASELINE_RET, 's', color='#F44336', markersize=12, zorder=10, label='Baseline')

        ax.set_xlabel(LABELS.get(param, param), fontsize=11)
        ax.set_ylabel('Mean Retention', fontsize=11)
        ax.set_title(f'{"ABCD"[idx]}. {param}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

        # Log scale for tau_slow
        if param == 'tau_slow':
            ax.set_xscale('log')

    plt.tight_layout()
    fig_path = OUT_DIR / 'fig_Q6_param_sensitivity.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(fig_path).replace('.png', '.pdf'), bbox_inches='tight')
    fig.savefig(str(fig_path).replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    print(f'\nQ6 figure saved: {fig_path}')

    # Summary
    print('\nQ6 PARAMETER SENSITIVITY SUMMARY:')
    for param, values in SWEEPS.items():
        print(f'\n  {param}:')
        for val in values:
            if val == BASELINE[param]:
                print(f'    {val} = {BASELINE_RET:.4f} (baseline)')
            else:
                cond = f'{param}_{val}'
                if cond in q6:
                    r = q6[cond]['retention']
                    pct = (r - BASELINE_RET) / BASELINE_RET * 100
                    print(f'    {val} = {r:.4f} ({pct:+.1f}%)')
                else:
                    print(f'    {val} = MISSING')
else:
    print(f'Q6 PKL not found at {Q6_PKL} -- run q6_param_sweep.py first')

print('\n[DONE] Q5+Q6 figures complete')
