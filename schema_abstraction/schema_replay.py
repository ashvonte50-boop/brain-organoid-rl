"""Advanced replay mechanisms: uncertainty-weighted prioritisation and
Bayesian-filtering interpretation of reverse replay.

These extend the behavioural analysis of replay without modifying the
core replay engine.  They read replay_metrics from the experiment results
and attach additional analytical layers.
"""

import numpy as np

from compare_catastrophic_forgetting import _safe_mean


# ═════════════════════════════════════════════════════════════════════════
# 1.  UNCERTAINTY-WEIGHTED REPLAY (Active Inference)
# ═════════════════════════════════════════════════════════════════════════

def compute_uncertainty_weights(prior_scores, replay_metrics):
    """Compute uncertainty-weighted replay priority from prior probe scores.

    Uncertainty = variance of probe scores across recent trials.
    High-uncertainty memories get higher replay priority (active inference:
    the network replays what it is *most uncertain about*, not what it
    knows best).

    Args:
        prior_scores: list of (n_mem,) arrays — probe scores for each assembly
            at each encoding step.
        replay_metrics: list of dicts from replay events.

    Returns:
        dict with per-assembly uncertainty scores and replay counts.
    """
    if not prior_scores or not replay_metrics:
        return {}
    n_mem = len(prior_scores[0]) if prior_scores else 0
    scores_by_mem = [[] for _ in range(n_mem)]
    for ps in prior_scores:
        for ai in range(min(len(ps), n_mem)):
            s = ps[ai]
            if np.isfinite(s):
                scores_by_mem[ai].append(s)
    uncertainties = []
    means = []
    for ai in range(n_mem):
        arr = np.array(scores_by_mem[ai])
        if len(arr) >= 2:
            uncertainties.append(float(np.std(arr, ddof=1)))
            means.append(float(np.mean(arr)))
        else:
            uncertainties.append(1.0)
            means.append(0.0)
    # Count replay events per assembly
    replay_counts = np.zeros(n_mem, dtype=int)
    for rm in replay_metrics:
        seq = rm.get("sequence", [])
        for sid in seq:
            if isinstance(sid, int) and sid < n_mem:
                replay_counts[sid] += 1
    return {
        "uncertainty": uncertainties,
        "mean_score": means,
        "replay_counts": replay_counts.tolist(),
        "uncertainty_weighted_priority": [
            u / (m + 0.01) if m > 0 else u
            for u, m in zip(uncertainties, means)
        ],
    }


# ═════════════════════════════════════════════════════════════════════════
# 2.  REVERSE REPLAY AS BAYESIAN FILTERING
# ═════════════════════════════════════════════════════════════════════════

def analyze_reverse_replay_bayesian(replay_metrics):
    """Analyse reverse replay events as Bayesian filtering updates.

    Theory: reverse replay corresponds to the backward pass of a Bayesian
    filter (smoothing).  Forward replay propagates activity from current
    state to predicted next state (prediction).  Reverse replay propagates
    from the later state back to correct the earlier one (update/correction).

    This analysis measures:
      - ``reverse_fraction``: proportion of replay events that are reverse.
      - ``coherence_improvement``: does the network *gain* coherence after
        reverse events (indicating successful correction)?
      - ``predictive_ratio``: forward:reverse imbalance.

    Returns dict with per-condition summary.
    """
    if not replay_metrics:
        return {}
    n_events = len(replay_metrics)
    is_reverse = []
    coherence_after = []
    for rm in replay_metrics:
        is_reverse.append(rm.get("is_reverse", False))
        coherence_after.append(rm.get("mean_coherence", 0.0))
    if n_events == 0:
        return {"reverse_fraction": 0.0, "n_events": 0}
    rev_mask = np.array(is_reverse, dtype=bool)
    fwd_mask = ~rev_mask
    n_rev = int(rev_mask.sum())
    n_fwd = int(fwd_mask.sum())
    rev_coh = None
    fwd_coh = None
    if n_rev > 0:
        rev_coh = float(np.mean([coherence_after[i] for i in range(n_events) if is_reverse[i]]))
    if n_fwd > 0:
        fwd_coh = float(np.mean([coherence_after[i] for i in range(n_events) if not is_reverse[i]]))
    # Coherence improvement: compare coherence before vs after reverse events
    coh_improvement = 0.0
    if n_rev > 0 and n_events > 1:
        diffs = []
        for i in range(1, n_events):
            if is_reverse[i] and not is_reverse[i - 1]:
                diffs.append(coherence_after[i] - coherence_after[i - 1])
        if diffs:
            coh_improvement = _safe_mean(diffs)
    return {
        "reverse_fraction": n_rev / max(1, n_events),
        "n_reverse": n_rev,
        "n_forward": n_fwd,
        "n_events": n_events,
        "mean_coherence_reverse": rev_coh,
        "mean_coherence_forward": fwd_coh,
        "coherence_improvement_after_reverse": coh_improvement,
        "predictive_ratio": n_fwd / max(1, n_rev),
    }


# ═════════════════════════════════════════════════════════════════════════
# 3.  POST-HOC ANALYSIS WRAPPER
# ═════════════════════════════════════════════════════════════════════════

def analyze_all_replay_metrics(all_results):
    """Run uncertainty and Bayesian analyses on all conditions.

    Returns dict keyed by condition label.
    """
    out = {}
    for res in all_results:
        label = res["cond"]["label"]
        trial_analyses = []
        for t in res.get("trials", []):
            rp = t.get("replay_metrics", [])
            ps = t.get("baseline_scores", None)
            # Build prior scores from available data
            prior = [t.get("baseline_scores", np.array([]))]
            for snaps in t.get("centroid_snapshots", []):
                if "label" in snaps and "post_encode" in snaps["label"]:
                    prior.append(snaps.get("scores", np.array([])))
            uw = compute_uncertainty_weights(prior, rp)
            bf = analyze_reverse_replay_bayesian(rp)
            trial_analyses.append({
                "uncertainty_weights": uw,
                "bayesian_filtering": bf,
            })
        out[label] = trial_analyses
    return out
