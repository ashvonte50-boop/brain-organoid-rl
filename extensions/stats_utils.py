"""
extensions/stats_utils.py — Statistical rigor utilities (Task 7).

Provides:
  bootstrap_ci       — percentile bootstrap confidence intervals
  permutation_test   — two-sample permutation test
  fdr_correction     — Benjamini-Hochberg FDR correction
  cohens_d           — standardised mean difference
  rank_biserial_r    — nonparametric effect size for MWU
  wilcoxon_mwu       — Wilcoxon MWU (unpaired) / signed-rank (paired)
  stats_table        — build per-condition stats rows vs a reference
  export_stats_csv   — write stats_table rows to CSV
  print_stats_table  — console-formatted stats table

All functions are pure (no side effects) and importable in worker processes.
"""
import csv
import os
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "bootstrap_ci",
    "permutation_test",
    "fdr_correction",
    "cohens_d",
    "rank_biserial_r",
    "wilcoxon_mwu",
    "stats_table",
    "export_stats_csv",
    "print_stats_table",
]

# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    stat: Callable = np.mean,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval.

    Returns (lo, hi, estimate) where estimate = stat(data).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    estimate = float(stat(data))
    boots = np.array([
        stat(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1.0 - alpha / 2)))
    return lo, hi, estimate


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 5000,
    alternative: str = "two-sided",
    stat: Callable = None,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """
    Two-sample permutation test on the difference of means.

    alternative: 'two-sided' | 'greater' (a > b) | 'less' (a < b)

    Returns (p_value, observed_stat, null_distribution).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if stat is None:
        stat = lambda x, y: float(np.mean(x) - np.mean(y))
    observed = stat(a, b)
    pooled = np.concatenate([a, b])
    na = len(a)
    null = np.array([
        stat(pooled[perm[:na]], pooled[perm[na:]])
        for perm in [rng.permutation(len(pooled)) for _ in range(n_perm)]
    ])
    if alternative == "two-sided":
        p = float(np.mean(np.abs(null) >= np.abs(observed)))
    elif alternative == "greater":
        p = float(np.mean(null >= observed))
    else:  # less
        p = float(np.mean(null <= observed))
    return p, float(observed), null


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns (reject_mask, adjusted_p_values).
    reject_mask[i] is True if hypothesis i is rejected at level alpha.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    adjusted = np.minimum(1.0, p * n / ranked)
    # Enforce monotonicity from right
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    reject = adjusted <= alpha
    return reject, adjusted


def cohens_d(
    a: np.ndarray,
    b: np.ndarray,
    pooled: bool = True,
) -> float:
    """
    Cohen's d effect size: (mean_a - mean_b) / pooled_std.
    Uses pooled SD when pooled=True, else uses SD of b (Glass's delta).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = float(np.mean(a) - np.mean(b))
    if pooled:
        na, nb = len(a), len(b)
        var_p = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
        sd = float(np.sqrt(max(var_p, 1e-12)))
    else:
        sd = float(np.std(b, ddof=1))
        sd = max(sd, 1e-12)
    return diff / sd


def rank_biserial_r(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Rank-biserial correlation — effect size for Mann-Whitney U.
    r = 1 - 2U / (n_a * n_b)   where U is the Wilcoxon statistic for a.
    Returns value in [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    # U_a = # pairs where a_i > b_j
    u_a = float(np.sum(
        a[:, None] > b[None, :]
    )) + 0.5 * float(np.sum(a[:, None] == b[None, :]))
    r = 1.0 - 2.0 * u_a / (na * nb)
    return float(r)


def wilcoxon_mwu(
    a: np.ndarray,
    b: np.ndarray,
    paired: bool = False,
) -> Tuple[float, float]:
    """
    Wilcoxon Mann-Whitney U (unpaired) or signed-rank test (paired).

    Returns (statistic, p_value).
    Falls back to scipy.stats; if unavailable returns (nan, nan) with warning.
    """
    try:
        from scipy import stats as sps
    except ImportError:
        warnings.warn("scipy not available — wilcoxon_mwu returns nan")
        return float("nan"), float("nan")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if paired:
        result = sps.wilcoxon(a - b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    else:
        result = sps.mannwhitneyu(a, b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------

def stats_table(
    conditions: List[dict],
    reference_idx: int = 0,
    ci: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> List[dict]:
    """
    Build per-condition statistics rows versus a reference condition.

    Each entry in `conditions` must be a dict with keys:
      "label"  : str
      "scores" : array-like of per-trial scalar retention values

    Returns list of row dicts with keys:
      label, n, mean, sem, ci_lo, ci_hi,
      cohens_d_vs_ref, rank_r_vs_ref,
      perm_p, perm_p_adj, reject_fdr
    """
    if not conditions:
        return []

    rows = []
    ref_scores = np.asarray(conditions[reference_idx]["scores"], dtype=float)

    raw_p = []
    pre_rows = []
    for i, cond in enumerate(conditions):
        scores = np.asarray(cond["scores"], dtype=float)
        lo, hi, est = bootstrap_ci(scores, n_boot=n_boot, ci=ci, seed=seed + i)
        n = len(scores)
        mean = float(np.mean(scores))
        sem = float(np.std(scores, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        if i == reference_idx:
            d = 0.0
            r = 0.0
            p = 1.0
        else:
            d = cohens_d(scores, ref_scores)
            r = rank_biserial_r(scores, ref_scores)
            p, _, _ = permutation_test(scores, ref_scores, n_perm=5000, seed=seed + i)
        raw_p.append(p)
        pre_rows.append({
            "label": cond["label"],
            "n": n,
            "mean": mean,
            "sem": sem,
            "ci_lo": lo,
            "ci_hi": hi,
            "cohens_d_vs_ref": d,
            "rank_r_vs_ref": r,
            "perm_p": p,
        })

    reject, adj_p = fdr_correction(np.array(raw_p))
    for i, row in enumerate(pre_rows):
        row["perm_p_adj"] = float(adj_p[i])
        row["reject_fdr"] = bool(reject[i])
        rows.append(row)
    return rows


def export_stats_csv(rows: List[dict], path: str) -> None:
    """Write stats_table rows to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats_table(rows: List[dict], title: str = "Statistics") -> None:
    """Print a formatted stats table to stdout."""
    if not rows:
        print(f"[{title}] No rows.")
        return
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    hdr = (
        f"{'Condition':<26} {'N':>3} {'Mean':>8} {'SEM':>7} "
        f"{'95% CI':>15} {'d':>6} {'r':>6} {'p':>7} {'p_adj':>7} {'FDR':>5}"
    )
    print(hdr)
    print("-" * 72)
    for r in rows:
        ci_str = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
        fdr_str = "*" if r["reject_fdr"] else " "
        p_str = f"{r['perm_p']:.4f}" if not np.isnan(r['perm_p']) else "  n/a"
        pa_str = f"{r['perm_p_adj']:.4f}" if not np.isnan(r['perm_p_adj']) else "  n/a"
        print(
            f"  {r['label']:<24} {r['n']:>3} {r['mean']:>8.4f} {r['sem']:>7.4f} "
            f"{ci_str:>15} {r['cohens_d_vs_ref']:>+6.2f} {r['rank_r_vs_ref']:>+6.3f} "
            f"{p_str:>7} {pa_str:>7} {fdr_str:>5}"
        )
    print("=" * 72)
    print("  d = Cohen's d vs reference; r = rank-biserial r; * = FDR-significant")
    print()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os
    _os.environ.setdefault("PYTHONUNBUFFERED", "1")

    rng = np.random.default_rng(42)
    a = rng.normal(1.0, 0.3, 15)
    b = rng.normal(0.0, 0.3, 15)
    c = rng.normal(0.5, 0.3, 15)

    print("bootstrap_ci(a):", bootstrap_ci(a))
    p, obs, _ = permutation_test(a, b)
    print(f"permutation_test(a vs b): p={p:.4f}, obs={obs:.4f}")
    print(f"cohens_d(a,b): {cohens_d(a,b):.3f}")
    print(f"rank_biserial_r(a,b): {rank_biserial_r(a,b):.3f}")
    stat, pval = wilcoxon_mwu(a, b)
    print(f"wilcoxon_mwu(a,b): stat={stat:.1f}, p={pval:.4f}")

    conds = [
        {"label": "Reference", "scores": b},
        {"label": "Condition A", "scores": a},
        {"label": "Condition C", "scores": c},
    ]
    rows = stats_table(conds, reference_idx=0, n_boot=500, seed=0)
    print_stats_table(rows, title="Self-test Stats Table")
    export_stats_csv(rows, "/tmp/stats_test.csv")
    print("CSV written to /tmp/stats_test.csv")
