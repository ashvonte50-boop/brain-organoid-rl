"""
ABLATION REPORT GENERATOR
Replay Distortion / Directional Schema Abstraction
===================================================

Generates ablation_report.pdf from ablation_results/ PKL files.
Uses matplotlib PdfPages — no external LaTeX or reportlab required.

Usage:
  python ablation_report.py
  python ablation_report.py --mode hyper
"""
import os, sys, pickle, argparse, datetime
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.stats import ttest_ind

# ── Paths ──────────────────────────────────────────────────────────────────────
IN_DIR   = r'C:\Users\Admin\brain-organoid-rl\ablation_results'
FIG_DIR  = os.path.join(IN_DIR, 'figures')
OUT_PATH = os.path.join(IN_DIR, 'ablation_report.pdf')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

C_TITLE   = '#1a3a5c'
C_FULL    = '#2E86AB'
C_ABLATE  = '#E84855'
C_NEUTRAL = '#8B8B8B'
C_BODY    = '#1a1a1a'
C_LIGHT   = '#f5f7fa'

MECH_NAMES = {
    'M1': 'Overlap-Sensitive Coherence', 'M2': 'Cross-Assembly LTD',
    'M3': 'Overlap-Weighted Prioritization', 'M4': 'Competitive Persistence Budget',
    'M5': 'Directional Drift', 'M6': 'Shared-Neuron Fatigue',
    'M7': 'Heterosynaptic LTD Tag', 'M8': 'Training-Time Decorrelation',
    'M9': 'Coherence WTA', 'M10': 'Reconsolidation Window', 'MB': 'Core Boost',
}

METRICS = ['retention_A', 'retention_B', 'retention_C', 'retention_D',
           'retention_mean', 'dai_core', 'real_schema', 'distortion']


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(name):
    p = os.path.join(IN_DIR, f'{name}.pkl')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def _agg(seed_list, mode='natural'):
    out = {}
    for k in METRICS:
        vals = [s[mode][k] for s in seed_list if mode in s and k in s.get(mode, {})]
        if vals:
            out[k + '_mean'] = float(np.mean(vals))
            out[k + '_sem']  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            out[k + '_vals'] = vals
            out[k + '_n']    = len(vals)
    return out


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def _cohen_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2: return 0.0
    s = np.sqrt(((len(a)-1)*a.std(ddof=1)**2+(len(b)-1)*b.std(ddof=1)**2)/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/(s+1e-9))


def _text_page(pdf, lines, fontsize_title=18, fontsize_body=11, bg=C_LIGHT):
    """Render a text-only page in the PDF."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(bg)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                boxstyle='round,pad=0.01', linewidth=1.5,
                                edgecolor='#cccccc', facecolor='white'))
    y = 0.92
    for kind, text in lines:
        if kind == 'title':
            ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top',
                    fontsize=fontsize_title, fontweight='bold', color=C_TITLE, wrap=True)
            y -= 0.08
        elif kind == 'subtitle':
            ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top',
                    fontsize=14, color='#444444', style='italic')
            y -= 0.06
        elif kind == 'h2':
            ax.text(0.07, y, text, transform=ax.transAxes, ha='left', va='top',
                    fontsize=13, fontweight='bold', color=C_TITLE)
            y -= 0.055
        elif kind == 'body':
            ax.text(0.07, y, text, transform=ax.transAxes, ha='left', va='top',
                    fontsize=fontsize_body, color=C_BODY, wrap=True,
                    multialignment='left')
            y -= 0.045
        elif kind == 'bullet':
            ax.text(0.09, y, f'• {text}', transform=ax.transAxes, ha='left', va='top',
                    fontsize=fontsize_body, color=C_BODY)
            y -= 0.04
        elif kind == 'space':
            y -= 0.025
        elif kind == 'hr':
            ax.axhline(y + 0.01, xmin=0.07, xmax=0.93, color='#dddddd', lw=1.0)
            y -= 0.02
        if y < 0.08:
            break
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _figure_page(pdf, img_path, caption, title=None):
    """Embed a pre-saved figure PNG into a PDF page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', color=C_TITLE, y=0.97)
    try:
        img = plt.imread(img_path)
        ax  = fig.add_axes([0.05, 0.10, 0.90, 0.82])
        ax.imshow(img, aspect='auto')
        ax.axis('off')
    except Exception:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'[Figure not found:\n{img_path}]',
                ha='center', va='center', fontsize=12, color='red',
                transform=ax.transAxes)
        ax.axis('off')
    fig.text(0.5, 0.03, caption, ha='center', fontsize=10, color='#555555',
             style='italic', wrap=True)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _results_table_page(pdf, single_data, mode='natural'):
    """Render a results table page."""
    MECH_ORDER = ['FULL', 'ABLATE_M1', 'ABLATE_M2', 'ABLATE_M3', 'ABLATE_M4',
                  'ABLATE_M5', 'ABLATE_M6', 'ABLATE_M7', 'ABLATE_M8',
                  'ABLATE_M9', 'ABLATE_M10', 'ABLATE_MB']
    XLABELS    = ['Full', '-M1', '-M2', '-M3', '-M4', '-M5',
                  '-M6', '-M7', '-M8', '-M9', '-M10', '-MB']
    col_labels = ['Condition', 'n', 'DAI_core', 'REAL_SCHEMA', 'Distortion', 'Ret(mean)', 'p(DAI)', 'Sig']

    rows = []
    full_agg  = _agg(single_data.get('FULL', []), mode)
    full_vals = full_agg.get('dai_core_vals', [])
    for cname, xlabel in zip(MECH_ORDER, XLABELS):
        sl  = single_data.get(cname, [])
        if not sl:
            continue
        agg  = _agg(sl, mode)
        vals = agg.get('dai_core_vals', [])
        n    = agg.get('dai_core_n', 0)
        if cname != 'FULL' and full_vals and vals:
            _, p = ttest_ind(full_vals, vals, equal_var=False)
            sig  = _sig_stars(p)
        else:
            p, sig = np.nan, '—'
        rows.append([
            xlabel,
            str(n),
            f'{agg.get("dai_core_mean", np.nan):+.4f} ± {agg.get("dai_core_sem",0):.4f}',
            f'{agg.get("real_schema_mean", np.nan):+.4f} ± {agg.get("real_schema_sem",0):.4f}',
            f'{agg.get("distortion_mean", np.nan):.4f} ± {agg.get("distortion_sem",0):.4f}',
            f'{agg.get("retention_mean_mean", np.nan):.4f} ± {agg.get("retention_mean_sem",0):.4f}',
            f'{p:.4f}' if not np.isnan(p) else '—',
            sig,
        ])

    fig, ax = plt.subplots(figsize=(14, max(5, len(rows) * 0.5 + 2)))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(C_TITLE)
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate rows
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor('#f0f4f8' if i % 2 == 0 else 'white')
    ax.set_title(f'Table 1: Single-Mechanism Ablation Results  (mode={mode})',
                 fontsize=13, fontweight='bold', color=C_TITLE, pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _importance_table_page(pdf, imp_data, mode='natural'):
    """Render mechanism importance ranking table."""
    if not imp_data:
        return
    ranked = sorted(imp_data.items(),
                    key=lambda x: abs(x[1]['delta_dai']) + abs(x[1]['delta_rs']),
                    reverse=True)
    col_labels = ['Rank', 'Mechanism', 'ΔDAI_core', 'ΔREAL_SCHEMA', 'ΔDistortion',
                  'ΔRetention', "Cohen's d", 'p-value', 'Sig']
    rows = []
    for rank, (mid, imp) in enumerate(ranked, 1):
        sig = _sig_stars(imp['p_dai'])
        rows.append([
            str(rank), f'{mid}: {MECH_NAMES.get(mid, mid)}',
            f'{imp["delta_dai"]:+.4f}',
            f'{imp["delta_rs"]:+.4f}',
            f'{imp["delta_di"]:+.4f}',
            f'{imp["delta_ret"]:+.4f}',
            f'{imp["cohens_d_dai"]:+.3f}',
            f'{imp["p_dai"]:.4f}',
            sig,
        ])

    fig, ax = plt.subplots(figsize=(14, max(5, len(rows) * 0.6 + 2)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.15, 1.5)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(C_TITLE)
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor('#f0f4f8' if i % 2 == 0 else 'white')
    ax.set_title('Table 2: Mechanism Importance Ranking  (sorted by |ΔDAI| + |ΔRS|)',
                 fontsize=13, fontweight='bold', color=C_TITLE, pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _mini_bar(ax, conds, metric, mode, title, full_agg):
    MECH_ORDER = ['FULL', 'ABLATE_M1', 'ABLATE_M2', 'ABLATE_M3', 'ABLATE_M4',
                  'ABLATE_M5', 'ABLATE_M6', 'ABLATE_M7', 'ABLATE_M8',
                  'ABLATE_M9', 'ABLATE_M10', 'ABLATE_MB']
    XLABELS    = ['Full', '-M1', '-M2', '-M3', '-M4', '-M5',
                  '-M6', '-M7', '-M8', '-M9', '-M10', '-MB']
    full_vals = full_agg.get(f'{metric}_vals', [])
    means, sems, colors = [], [], []
    for cname in MECH_ORDER:
        sl  = conds.get(cname, [])
        if not sl:
            means.append(np.nan); sems.append(0); colors.append(C_NEUTRAL); continue
        agg = _agg(sl, mode)
        m   = agg.get(f'{metric}_mean', np.nan)
        s   = agg.get(f'{metric}_sem', 0)
        v   = agg.get(f'{metric}_vals', [])
        if cname == 'FULL':
            c = C_FULL
        elif full_vals and v:
            _, p = ttest_ind(full_vals, v, equal_var=False)
            c = C_ABLATE if p < 0.05 else C_NEUTRAL
        else:
            c = C_NEUTRAL
        means.append(m); sems.append(s); colors.append(c)
    x = np.arange(len(MECH_ORDER))
    ax.bar(x, means, yerr=sems, capsize=3, color=colors,
           error_kw=dict(elinewidth=1.2), zorder=3)
    fm = full_agg.get(f'{metric}_mean', np.nan)
    if not np.isnan(fm):
        ax.axhline(fm, color=C_FULL, lw=1.2, ls='--', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(XLABELS, fontsize=6.5, rotation=45, ha='right')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


# ── Report pages ──────────────────────────────────────────────────────────────

def generate_report(mode='natural'):
    print(f'Generating report: {OUT_PATH}', flush=True)
    today    = datetime.date.today().isoformat()
    n_seeds  = 10

    single_data   = _load('single_ablations')
    cum_data      = _load('cumulative_ablations')
    interact_data = _load('interaction_ablations')
    imp_pkl       = _load('importance_analysis')
    imp_data      = imp_pkl.get('importance', {}) if imp_pkl else {}

    full_agg = _agg(single_data.get('FULL', []), mode) if single_data else {}

    # Collect key stats for the report text
    full_dai  = full_agg.get('dai_core_mean', np.nan)
    full_rs   = full_agg.get('real_schema_mean', np.nan)
    full_di   = full_agg.get('distortion_mean', np.nan)
    full_ret  = full_agg.get('retention_mean_mean', np.nan)

    # Identify most important mechanism from importance data
    if imp_data:
        ranked = sorted(imp_data.items(),
                        key=lambda x: abs(x[1]['delta_dai']) + abs(x[1]['delta_rs']),
                        reverse=True)
        top_mid   = ranked[0][0]  if len(ranked) > 0 else 'N/A'
        top_d     = ranked[0][1]['delta_dai'] if len(ranked) > 0 else np.nan
        top2_mid  = ranked[1][0]  if len(ranked) > 1 else 'N/A'
        top2_d    = ranked[1][1]['delta_dai'] if len(ranked) > 1 else np.nan
        negl_mids = [mid for mid, imp in imp_data.items()
                     if abs(imp['delta_dai']) < 0.01 and imp['p_dai'] > 0.1]
    else:
        top_mid = top2_mid = 'N/A'
        top_d = top2_d = np.nan
        negl_mids = []

    with PdfPages(OUT_PATH) as pdf:
        # ── Page 1: Title ─────────────────────────────────────────────────────
        _text_page(pdf, [
            ('title',    'ABLATION STUDY REPORT'),
            ('subtitle', 'Replay Distortion as Directional Schema Abstraction'),
            ('space', ''),
            ('hr', ''),
            ('space', ''),
            ('body', f'Date: {today}'),
            ('body', f'Replay mode analysed: {mode}'),
            ('body', f'Seeds per condition: {n_seeds}  (BASE_SEED=42, step=1000)'),
            ('body', f'Mechanisms tested: M1–M10 + MB (11 mechanisms)'),
            ('body', f'Ablation conditions: 12 single + 11 cumulative + 5 interactions'),
            ('space', ''),
            ('hr', ''),
            ('space', ''),
            ('h2', 'Key Findings at a Glance'),
            ('bullet', f'Full model: DAI_core={full_dai:.4f},  REAL_SCHEMA={full_rs:.4f},  Distortion={full_di:.4f},  Retention={full_ret:.4f}'),
            ('bullet', f'Most important mechanism: {top_mid} ({MECH_NAMES.get(top_mid,"")})  ΔDAI={top_d:+.4f}'),
            ('bullet', f'Second most important:    {top2_mid} ({MECH_NAMES.get(top2_mid,"")})  ΔDAI={top2_d:+.4f}'),
            ('bullet', f'Negligible contribution:  {", ".join(negl_mids) if negl_mids else "None identified"}'),
            ('space', ''),
            ('body', 'This report accompanies the ablation_results/ directory, which contains'),
            ('body', 'structured PKL/CSV data files and publication-grade figures (PDF/SVG/PNG).'),
        ])

        # ── Page 2: Experimental Design ───────────────────────────────────────
        _text_page(pdf, [
            ('h2', '1. Experimental Design'),
            ('hr', ''),
            ('body', 'The ablation study removes one mechanism at a time from the full model'),
            ('body', '(M1–M10 + MB) and measures the resulting drop in four outcome metrics.'),
            ('space', ''),
            ('h2', 'Network'),
            ('bullet', 'IzhikevichNetwork: 1000 neurons (750 exc / 250 inh), sparse_modular arch'),
            ('bullet', 'Memories: 4 hierarchical schema assemblies (core=20 + unique=20 neurons each)'),
            ('bullet', 'Protocol: Train A→B→C→D sequentially, then replay, then probe retention'),
            ('space', ''),
            ('h2', 'Conditions'),
            ('bullet', 'A. SINGLE ABLATIONS: FULL + ABLATE_M1 through ABLATE_MB (12 conditions)'),
            ('bullet', 'B. IMPORTANCE ANALYSIS: ΔDAI, ΔRS, ΔDist, ΔRet ranked per mechanism'),
            ('bullet', 'C. CUMULATIVE ABLATIONS: Mechanisms added one-by-one (11 steps)'),
            ('bullet', 'D. INTERACTIONS: 5 pairs — M2+M7, M2+M10, M5+M8, M6+M7, M7+M10'),
            ('space', ''),
            ('h2', 'Seed Protocol'),
            ('bullet', 'BASE_SEED=42, seeds = [42, 1042, 2042, ..., 9042]  (n=10 per condition)'),
            ('bullet', 'Each trial: torch.manual_seed(seed) + np.random.seed(seed)'),
            ('bullet', 'Independent across conditions (no shared random state)'),
        ])

        # ── Page 3: Methods ───────────────────────────────────────────────────
        _text_page(pdf, [
            ('h2', '2. Methods'),
            ('hr', ''),
            ('h2', 'Mechanism Ablation'),
            ('body', 'Each mechanism is ablated by passing ablation_dict = {key: False} to'),
            ('body', 'run_sequential_experiment(). The MB (core boost) mechanism is ablated by'),
            ('body', 'setting boost_scale=1.0 in the replay wrapper.'),
            ('space', ''),
            ('h2', 'Metrics'),
            ('bullet', 'Retention_A/B/C/D — I_syn differential probe score per memory'),
            ('bullet', 'DAI_core — Directional Alignment Index (cosine similarity, core neurons)'),
            ('bullet', 'REAL_SCHEMA — (core-core − unique-core) / (core-core + unique-core) weight ratio'),
            ('bullet', 'Distortion — Mean L2 norm of centroid displacement per replay event'),
            ('space', ''),
            ('h2', 'Statistics'),
            ('bullet', "Independent two-sample t-test (Welch's): ablated vs full model"),
            ('bullet', "Cohen's d: effect size with pooled standard deviation"),
            ('bullet', '95% CI on effect size: SE = sqrt[(N1+N2)/(N1*N2) + d²/(2*(N-2))]'),
            ('bullet', 'Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. p≥0.05'),
            ('space', ''),
            ('h2', 'Synergy Analysis'),
            ('body', 'Synergy = additive_expected − joint_ablated, where:'),
            ('body', '  additive_expected = DAI(abl_A) + DAI(abl_B) − DAI(full)'),
            ('body', 'Positive synergy: removing A+B together hurts more than removing A and B independently.'),
        ])

        # ── Page 4: Results overview ─────────────────────────────────────────
        _text_page(pdf, [
            ('h2', '3. Results'),
            ('hr', ''),
            ('h2', 'Full Model Performance'),
            ('bullet', f'DAI_core:     {full_dai:.4f} ± {full_agg.get("dai_core_sem", 0):.4f}  (n={full_agg.get("dai_core_n", 0)})'),
            ('bullet', f'REAL_SCHEMA:  {full_rs:.4f} ± {full_agg.get("real_schema_sem", 0):.4f}'),
            ('bullet', f'Distortion:   {full_di:.4f} ± {full_agg.get("distortion_sem", 0):.4f}'),
            ('bullet', f'Retention:    {full_ret:.4f} ± {full_agg.get("retention_mean_sem", 0):.4f}'),
            ('space', ''),
            ('h2', 'Mechanism Summary'),
        ] + ([
            ('bullet', f'{mid}: ΔDAI={v["delta_dai"]:+.4f},  ΔRS={v["delta_rs"]:+.4f},  '
                       f"Cohen's d={v['cohens_d_dai']:+.2f},  p={v['p_dai']:.4f}  {_sig_stars(v['p_dai'])}")
            for mid, v in (sorted(imp_data.items(), key=lambda x: abs(x[1]['delta_dai']), reverse=True)
                           if imp_data else [])
        ]))

        # ── Page 5: Results table ─────────────────────────────────────────────
        if single_data:
            _results_table_page(pdf, single_data, mode)

        # ── Page 6: Importance table ──────────────────────────────────────────
        if imp_data:
            _importance_table_page(pdf, imp_data, mode)

        # ── Page 7: All-metrics bar chart ─────────────────────────────────────
        if single_data:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()
            for axi, (mkey, mlabel) in enumerate([
                    ('dai_core', 'DAI_core'), ('real_schema', 'REAL_SCHEMA'),
                    ('distortion', 'Distortion'), ('retention_mean', 'Retention (mean)')]):
                _mini_bar(axes[axi], single_data, mkey, mode, mlabel, full_agg)
            fig.suptitle('Results: Single-Mechanism Ablations (all metrics)',
                         fontsize=13, fontweight='bold', color=C_TITLE)
            legend_patches = [
                Patch(color=C_FULL, label='Full model'),
                Patch(color=C_ABLATE, label='Ablated (p<0.05)'),
                Patch(color=C_NEUTRAL, label='Ablated (n.s.)'),
            ]
            axes[0].legend(handles=legend_patches, loc='upper right', fontsize=8)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # ── Pages 8+: Embed figures from FIG_DIR ─────────────────────────────
        fig_configs = [
            ('fig01_single_dai.png',       'Figure 1: Single ablations — DAI_core'),
            ('fig02_single_rs.png',        'Figure 2: Single ablations — REAL_SCHEMA'),
            ('fig03_single_dist.png',      'Figure 3: Single ablations — Distortion'),
            ('fig04_retention_heatmap.png','Figure 4: Retention heatmap'),
            ('fig05_importance_ranking.png','Figure 5: Mechanism importance ranking'),
            ('fig06_cumulative_dai.png',   'Figure 6: Cumulative emergence — DAI_core'),
            ('fig07_cumulative_rs.png',    'Figure 7: Cumulative emergence — REAL_SCHEMA'),
            ('fig08_synergy_matrix.png',   'Figure 8: Synergy interaction matrix'),
            ('fig09_forest_plot.png',      'Figure 9: Effect-size forest plot'),
            ('fig10_summary.png',          'Figure 10: Summary (all metrics)'),
        ]
        for fname, caption in fig_configs:
            fpath = os.path.join(FIG_DIR, fname)
            _figure_page(pdf, fpath, caption)

        # ── Statistical analysis page ─────────────────────────────────────────
        _text_page(pdf, [
            ('h2', '4. Statistical Analysis'),
            ('hr', ''),
            ('body', 'Primary comparison: each ablated condition vs full model (Welch t-test).'),
            ('body', 'Degrees of freedom: Welch–Satterthwaite approximation.'),
            ('body', 'Multiple comparison note: Bonferroni threshold for 11 tests → α=0.0045.'),
            ('space', ''),
        ] + ([] if not imp_data else [
            ('h2', 'Statistically Significant Effects (p<0.05, DAI_core):'),
        ] + [
            ('bullet', f'{mid}: t={v["t_dai"]:.3f}, p={v["p_dai"]:.4f}, d={v["cohens_d_dai"]:.3f}  {_sig_stars(v["p_dai"])}')
            for mid, v in imp_data.items() if v['p_dai'] < 0.05
        ] + [
            ('space', ''),
            ('h2', 'Non-significant Effects (p≥0.05):'),
        ] + [
            ('bullet', f'{mid}: t={v["t_dai"]:.3f}, p={v["p_dai"]:.4f}  n.s.')
            for mid, v in imp_data.items() if v['p_dai'] >= 0.05
        ]))

        # ── Mechanism ranking page ────────────────────────────────────────────
        ranking_lines = [('h2', '5. Mechanism Ranking'), ('hr', '')]
        if imp_data:
            ranked = sorted(imp_data.items(),
                            key=lambda x: abs(x[1]['delta_dai']) + abs(x[1]['delta_rs']),
                            reverse=True)
            ranking_lines += [
                ('body', 'Mechanisms ranked by combined importance |ΔDAI| + |ΔRS|:'),
                ('space', ''),
            ] + [
                ('bullet',
                 f'#{i+1}. {mid} — {MECH_NAMES.get(mid,"")}: '
                 f'ΔDAI={v["delta_dai"]:+.4f}, ΔRS={v["delta_rs"]:+.4f}, '
                 f"d={v['cohens_d_dai']:+.2f}, {_sig_stars(v['p_dai'])}")
                for i, (mid, v) in enumerate(ranked)
            ] + [
                ('space', ''),
                ('h2', 'Classification:'),
                ('bullet', f'Critical  (|ΔDAI|>0.05): ' +
                 str([mid for mid, v in ranked if abs(v['delta_dai']) > 0.05])),
                ('bullet', f'Moderate  (|ΔDAI|=0.01–0.05): ' +
                 str([mid for mid, v in ranked if 0.01 <= abs(v['delta_dai']) <= 0.05])),
                ('bullet', f'Negligible (|ΔDAI|<0.01): ' +
                 str([mid for mid, v in ranked if abs(v['delta_dai']) < 0.01])),
            ]
        else:
            ranking_lines.append(('body', 'Importance analysis not yet run. Execute --part importance.'))
        _text_page(pdf, ranking_lines)

        # ── Interpretation page ───────────────────────────────────────────────
        _text_page(pdf, [
            ('h2', '6. Interpretation'),
            ('hr', ''),
            ('body', 'The ablation results reveal the mechanistic substrate of replay-driven'),
            ('body', 'schema abstraction.  Key insights:'),
            ('space', ''),
            ('bullet', 'Mechanisms producing the largest ΔDAI when ablated are the primary'),
            ('bullet', 'drivers of directional centroid movement toward the schema attractor.'),
            ('space', ''),
            ('bullet', 'Large ΔREAL_SCHEMA signals that a mechanism shapes the weight structure'),
            ('bullet', 'directly (core-to-core strengthening vs. unique-to-core weakening).'),
            ('space', ''),
            ('bullet', 'ΔDistortion reveals whether a mechanism controls replay fidelity'),
            ('bullet', '(low distortion = faithful replay; high = distorted/schematic replay).'),
            ('space', ''),
            ('bullet', 'Synergistic pairs (positive synergy matrix) suggest mechanistically'),
            ('bullet', 'coupled processes — removing either disrupts the other\'s contribution.'),
            ('space', ''),
            ('bullet', 'Mechanisms with negligible contribution in isolation may still play'),
            ('bullet', 'modulatory roles (revealed by cumulative and interaction analyses).'),
        ])

        # ── Conclusions page ──────────────────────────────────────────────────
        _text_page(pdf, [
            ('h2', '7. Key Conclusions'),
            ('hr', ''),
            ('body', 'Based on the ablation study, the following conclusions are drawn:'),
            ('space', ''),
            ('bullet', f'1. The full model achieves DAI_core={full_dai:.4f}, confirming directional schema abstraction.'),
            ('bullet', f'2. {top_mid} ({MECH_NAMES.get(top_mid,"")}) is the single most critical mechanism.'),
            ('bullet', f'3. {top2_mid} ({MECH_NAMES.get(top2_mid,"")}) is the second most critical mechanism.'),
            ('bullet', f'4. Negligible mechanisms (individual ablation has p>0.1): {negl_mids or "None"}'),
            ('space', ''),
            ('body', 'Cumulative analysis reveals which mechanisms are necessary vs sufficient'),
            ('body', 'for schema abstraction emergence (see Figures 6–7).'),
            ('space', ''),
            ('body', 'Interaction analysis reveals synergistic mechanism pairs (see Figure 8).'),
            ('space', ''),
            ('hr', ''),
            ('body', f'Report generated: {today}'),
            ('body', f'Output: {OUT_PATH}'),
            ('body', 'All data: ablation_results/  |  Figures: ablation_results/figures/'),
        ])

        # ── PDF metadata ──────────────────────────────────────────────────────
        d = pdf.infodict()
        d['Title']   = 'Ablation Study Report — Replay Distortion / Schema Abstraction'
        d['Author']  = 'Ablation Pipeline (automated)'
        d['Subject'] = 'Publication-grade ablation study of M1–M10 mechanisms'
        d['Keywords'] = 'ablation, schema, replay, DAI, REAL_SCHEMA, distortion'
        d['CreationDate'] = datetime.datetime.now()

    print(f'Report saved -> {OUT_PATH}', flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='natural', choices=['natural', 'hyper'])
    args = parser.parse_args()
    generate_report(mode=args.mode)
