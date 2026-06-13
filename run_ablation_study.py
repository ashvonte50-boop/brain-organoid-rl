"""
MASTER ABLATION STUDY RUNNER
Replay Distortion / Directional Schema Abstraction
===================================================

Runs the complete publication-grade ablation study in sequence:
  Step 1: ablation_pipeline.py  -- all experimental conditions (A+B+C+D)
  Step 2: ablation_figures.py   -- generate all 10 publication figures
  Step 3: ablation_report.py    -- compile ablation_report.pdf

Usage:
  python run_ablation_study.py                    # full study (10 seeds, ~40-50 hr)
  python run_ablation_study.py --seeds 3          # quick dev run (~12 hr)
  python run_ablation_study.py --seeds 3 --fast   # single only, skip cumul/interact
  python run_ablation_study.py --figures-only     # regenerate figures from saved results
  python run_ablation_study.py --report-only      # regenerate report from saved results
  python run_ablation_study.py --mode hyper       # analyse hyper-replay results

Output directory: ablation_results/
  single_ablations.pkl/.csv
  single_ablations_per_seed.csv
  cumulative_ablations.pkl/.csv
  interaction_ablations.pkl/.csv
  importance_analysis.pkl/.csv
  figures/fig01_*.{pdf,svg,png} ... fig10_*
  ablation_report.pdf
"""
import argparse, sys, time, os

def main():
    parser = argparse.ArgumentParser(description='Run complete ablation study')
    parser.add_argument('--seeds',        type=int, default=10)
    parser.add_argument('--mode',         default='natural', choices=['natural','hyper'])
    parser.add_argument('--fast',         action='store_true',
                        help='Run single ablations only (skip cumulative + interactions)')
    parser.add_argument('--figures-only', action='store_true',
                        help='Skip experiments; regenerate figures from saved results')
    parser.add_argument('--report-only',  action='store_true',
                        help='Skip experiments + figures; regenerate report only')
    parser.add_argument('--no-report',    action='store_true',
                        help='Skip PDF report generation')
    args = parser.parse_args()

    t0 = time.time()

    # ── Step 1: Experiments ───────────────────────────────────────────────────
    if not args.figures_only and not args.report_only:
        from ablation_pipeline import (
            run_single_ablations, run_cumulative_ablations,
            run_interaction_ablations, run_importance_analysis, print_summary,
        )

        part = 'single' if args.fast else 'all'
        print(f'\n{"="*65}', flush=True)
        print(f'ABLATION STUDY — seeds={args.seeds}  mode={args.mode}', flush=True)
        print(f'{"="*65}\n', flush=True)

        print('▶ STEP 1/3: Running experiments', flush=True)

        single_conds = run_single_ablations(n_seeds=args.seeds)
        print_summary(single_conds, mode=args.mode)
        imp_data = run_importance_analysis(single_conditions=single_conds, mode=args.mode)

        if not args.fast:
            run_cumulative_ablations(n_seeds=args.seeds)
            run_interaction_ablations(n_seeds=args.seeds)

        elapsed = time.time() - t0
        print(f'\nExperiments complete in {elapsed:.0f}s ({elapsed/3600:.2f} hr)', flush=True)

    # ── Step 2: Figures ───────────────────────────────────────────────────────
    if not args.report_only:
        print('\n▶ STEP 2/3: Generating figures', flush=True)
        from ablation_figures import generate_all_figures
        generate_all_figures(mode=args.mode)
        elapsed = time.time() - t0
        print(f'Figures complete (+{elapsed:.0f}s total)', flush=True)

    # ── Step 3: Report ────────────────────────────────────────────────────────
    if not args.no_report:
        print('\n▶ STEP 3/3: Generating PDF report', flush=True)
        from ablation_report import generate_report
        generate_report(mode=args.mode)
        elapsed = time.time() - t0
        print(f'Report complete (+{elapsed:.0f}s total)', flush=True)

    total = time.time() - t0
    print(f'\n{"="*65}', flush=True)
    print(f'ABLATION STUDY COMPLETE  |  Total: {total:.0f}s ({total/3600:.2f} hr)', flush=True)
    print(f'Results: ablation_results/', flush=True)
    print(f'Report:  ablation_results/ablation_report.pdf', flush=True)
    print(f'Figures: ablation_results/figures/', flush=True)
    print(f'{"="*65}', flush=True)


if __name__ == '__main__':
    main()
