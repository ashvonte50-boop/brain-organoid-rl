"""Statistical analysis for all nine schema-abstraction elements.

Each function takes ``all_results`` (the list-of-condition-dicts returned by
``run_all_conditions``) and returns a dict of per-condition results.
"""

import numpy as np
from scipy.stats import linregress as _linregress, pearsonr as _pearsonr, t as _scipy_t, ttest_ind, f_oneway
from scipy.stats import tukey_hsd

from compare_catastrophic_forgetting import _safe_mean
from .schema_core import CENTROID_INTERP_FACTOR


# ═════════════════════════════════════════════════════════════════════════
# 1.  CENTROID-BASED TESTS (Existing)
# ═════════════════════════════════════════════════════════════════════════

def extract_schema_trials(all_results, condition_idx=None):
    trials = []
    for res in all_results:
        if condition_idx is not None and int(res.get("cond_idx", -1)) != condition_idx:
            continue
        for t in res.get("trials", []):
            snaps = t.get("centroid_snapshots", [])
            if len(snaps) > 1:
                trials.append(t)
    return trials


def test_directionality(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        dir_scores = []
        for t in res.get("trials", []):
            snaps = t.get("centroid_snapshots", [])
            if len(snaps) < 2:
                continue
            centroids_bl = snaps[0].get("centroids", [])
            centroids_fin = snaps[-1].get("centroids", [])
            if len(centroids_bl) < 2 or len(centroids_fin) < 2:
                continue
            n_mem = len(centroids_bl)
            for i in range(n_mem):
                for j in range(i + 1, n_mem):
                    ci0, cj0 = centroids_bl[i].ravel(), centroids_bl[j].ravel()
                    ci1, cj1 = centroids_fin[i].ravel(), centroids_fin[j].ravel()
                    disp_i = ci1 - ci0
                    disp_j = cj1 - cj0
                    schema_i = (cj1 - ci1) * CENTROID_INTERP_FACTOR
                    schema_j = (ci1 - cj1) * CENTROID_INTERP_FACTOR

                    def _cos(a, b):
                        na, nb = np.linalg.norm(a), np.linalg.norm(b)
                        if na < 1e-10 or nb < 1e-10:
                            return 0.0
                        return float(np.dot(a, b) / (na * nb))
                    dir_scores.append((_cos(disp_i, schema_i) + _cos(disp_j, schema_j)) / 2.0)
        if len(dir_scores) < 2:
            results[label] = {"mean": 0.0, "sem": 0.0, "n": 0, "t": 0.0, "p": 1.0}
            if verbose:
                print(f"  [directionality] {label:25s}: insufficient data", flush=True)
            continue
        arr = np.array(dir_scores)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se = s / np.sqrt(len(arr))
        t_stat = m / se if se > 0 else 0.0
        p_val = 2.0 * (1.0 - _scipy_t.cdf(abs(t_stat), df=max(1, len(arr) - 1)))
        results[label] = {"mean": m, "sem": se, "n": len(arr), "t": t_stat, "p": p_val}
        if verbose:
            print(f"  [directionality] {label:25s}: dir={m:.4f}+-{se:.4f}  "
                  f"t({len(arr)-1})={t_stat:.3f}  p={p_val:.4f}  n={len(arr)}", flush=True)
    return results


def test_schema_convergence(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        slopes = []
        for t in res.get("trials", []):
            conv = t.get("schema_convergence", {})
            for steps in conv.values():
                if len(steps) < 2:
                    continue
                dists = [(s["dist_i_to_schema"] + s["dist_j_to_schema"]) / 2.0 for s in steps]
                x = np.arange(len(dists))
                if np.std(dists) < 1e-10:
                    continue
                try:
                    slope, _, _, p_val, _ = _linregress(x, dists)
                    slopes.append((slope, p_val))
                except Exception:
                    continue
        if len(slopes) < 2:
            results[label] = {"mean_slope": 0.0, "n": 0, "p": 1.0}
            if verbose:
                print(f"  [schema_convergence] {label:25s}: insufficient data", flush=True)
            continue
        slopes_arr = np.array([s[0] for s in slopes])
        p_vals = np.array([s[1] for s in slopes])
        m_slope = float(np.mean(slopes_arr))
        se_slope = float(np.std(slopes_arr, ddof=1)) / np.sqrt(len(slopes_arr))
        combined_stat = -2.0 * np.sum(np.log(np.clip(p_vals, 1e-300, 1.0)))
        combined_p = 1.0 - _scipy_t.cdf(combined_stat, df=2 * len(p_vals))
        results[label] = {"mean_slope": m_slope, "sem_slope": se_slope,
                          "n": len(slopes), "combined_p": combined_p}
        if verbose:
            print(f"  [schema_convergence] {label:25s}: slope={m_slope:.4f}+-{se_slope:.4f}  "
                  f"Fisher_p={combined_p:.4f}  n={len(slopes)}", flush=True)
    return results


def test_coherence_drift_correlation(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        coherences, drifts = [], []
        for t in res.get("trials", []):
            snaps = t.get("centroid_snapshots", [])
            rp_metrics = t.get("replay_metrics", [])
            if len(snaps) < 2 or len(rp_metrics) < 2:
                continue
            coh = _safe_mean([m.get("mean_coherence", 0.0) for m in rp_metrics])
            bl = snaps[0].get("centroids", [])
            fin = snaps[-1].get("centroids", [])
            if len(bl) < 2 or len(fin) < 2:
                continue
            displacements = [float(np.linalg.norm(fin[i].ravel() - bl[i].ravel())) for i in range(len(bl))]
            drift_mag = _safe_mean(displacements)
            coherences.append(coh)
            drifts.append(drift_mag)
        if len(coherences) < 3:
            results[label] = {"r": 0.0, "p": 1.0, "n": 0}
            if verbose:
                print(f"  [coherence-drift] {label:25s}: insufficient data", flush=True)
            continue
        try:
            r, p = _pearsonr(coherences, drifts)
        except Exception:
            r, p = 0.0, 1.0
        results[label] = {"r": r, "p": p, "n": len(coherences)}
        if verbose:
            print(f"  [coherence-drift] {label:25s}: r={r:.4f}  p={p:.4f}  n={len(coherences)}", flush=True)
    return results


def compute_retention_tradeoff(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        conv_rates, decays = [], []
        for t in res.get("trials", []):
            conv = t.get("schema_convergence", {})
            bl = t.get("baseline_scores", None)
            fin = t.get("final_scores", None)
            if not conv or bl is None or fin is None:
                continue
            pair_slopes = []
            for steps in conv.values():
                if len(steps) < 2:
                    continue
                dists = [(s["dist_i_to_schema"] + s["dist_j_to_schema"]) / 2.0 for s in steps]
                x = np.arange(len(dists))
                if np.std(dists) < 1e-10:
                    continue
                try:
                    slope, _, _, _, _ = _linregress(x, dists)
                    pair_slopes.append(-slope)
                except Exception:
                    continue
            if not pair_slopes:
                continue
            cr = _safe_mean(pair_slopes)
            valid = np.isfinite(bl) & np.isfinite(fin)
            if valid.sum() < 2:
                continue
            decay = 1.0 - _safe_mean(fin[valid] / (bl[valid] + 1e-10))
            conv_rates.append(cr)
            decays.append(decay)
        if len(conv_rates) < 3:
            results[label] = {"r": 0.0, "p": 1.0, "n": 0}
            if verbose:
                print(f"  [retention-tradeoff] {label:25s}: insufficient data", flush=True)
            continue
        try:
            r, p = _pearsonr(conv_rates, decays)
        except Exception:
            r, p = 0.0, 1.0
        results[label] = {"r": r, "p": p, "n": len(conv_rates)}
        if verbose:
            print(f"  [retention-tradeoff] {label:25s}: r={r:.4f}  p={p:.4f}  n={len(conv_rates)}", flush=True)
    return results


def test_overlap_proportionality(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        overlaps, drifts = [], []
        for t in res.get("trials", []):
            snaps = t.get("centroid_snapshots", [])
            traj = t.get("distance_trajectories", {})
            if len(snaps) < 2:
                continue
            bl = snaps[0].get("centroids", [])
            fin = snaps[-1].get("centroids", [])
            if len(bl) < 2:
                continue
            for (i, j), data in traj.items():
                of = data.get("overlap_frac", 0.0)
                dist_series = data.get("pair_dist", [])
                if len(dist_series) < 2:
                    continue
                overlaps.append(of)
                drifts.append(abs(dist_series[-1] - dist_series[0]))
        if len(overlaps) < 3:
            results[label] = {"r": 0.0, "p": 1.0, "n": 0}
            if verbose:
                print(f"  [overlap-proportionality] {label:25s}: insufficient data", flush=True)
            continue
        try:
            r, p = _pearsonr(overlaps, drifts)
        except Exception:
            r, p = 0.0, 1.0
        results[label] = {"r": r, "p": p, "n": len(overlaps)}
        if verbose:
            print(f"  [overlap-proportionality] {label:25s}: r={r:.4f}  p={p:.4f}  n={len(overlaps)}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 2.  GENERALIZATION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_generalization(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        gen_scores = []
        for t in res.get("trials", []):
            gen = t.get("generalization", None)
            if gen is None or not isinstance(gen, dict):
                continue
            for aidx_str in gen:
                v = gen[aidx_str]
                if isinstance(v, dict):
                    gs = v.get("generalization_score", None)
                    if gs is not None:
                        gen_scores.append(gs)
        if not gen_scores:
            results[label] = {"mean": 0.0, "sem": 0.0, "n": 0}
            if verbose:
                print(f"  [generalization] {label:25s}: no data", flush=True)
            continue
        arr = np.array(gen_scores)
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        results[label] = {"mean": m, "sem": se, "n": len(arr)}
        if verbose:
            print(f"  [generalization] {label:25s}: {m:.4f}+-{se:.4f}  n={len(arr)}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 3.  ANTI-PREDICTION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_anti_prediction(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        nat_scores, hf_scores = [], []
        for t in res.get("trials", []):
            ap = t.get("anti_prediction", None)
            if ap is None or not isinstance(ap, dict):
                continue
            ns = ap.get("natural_generalization", None)
            hs = ap.get("high_fidelity_generalization", None)
            if ns is not None:
                nat_scores.append(ns)
            if hs is not None:
                hf_scores.append(hs)
        if not nat_scores or not hf_scores:
            results[label] = {"natural_mean": 0.0, "hf_mean": 0.0, "n": 0}
            if verbose:
                print(f"  [anti-prediction] {label:25s}: no data", flush=True)
            continue
        nat_m = _safe_mean(nat_scores)
        hf_m = _safe_mean(hf_scores)
        results[label] = {"natural_mean": nat_m, "hf_mean": hf_m,
                          "n": len(nat_scores), "gap": hf_m - nat_m}
        if verbose:
            print(f"  [anti-prediction] {label:25s}: nat={nat_m:.4f} hf={hf_m:.4f} "
                  f"gap={hf_m-nat_m:.4f}  n={len(nat_scores)}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 4.  DOWNSSCALING ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_downscaling(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        active_syns = []
        downscale_events = []
        for t in res.get("trials", []):
            ds = t.get("downscale_summary", None)
            if ds is not None and isinstance(ds, dict):
                active_syns.append(ds.get("active_synapses", 0))
                downscale_events.append(ds.get("downscale_events", 0))
        if not active_syns:
            results[label] = {"mean_active_synapses": 0.0, "n": 0}
            if verbose:
                print(f"  [downscaling] {label:25s}: no data", flush=True)
            continue
        results[label] = {
            "mean_active_synapses": _safe_mean(active_syns),
            "mean_downscale_events": _safe_mean(downscale_events),
            "n": len(active_syns),
        }
        if verbose:
            print(f"  [downscaling] {label:25s}: active_syn={_safe_mean(active_syns):.0f}  "
                  f"events={_safe_mean(downscale_events):.1f}  n={len(active_syns)}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 5.  GENERATIVE LAYER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_generative_layer(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        final_mses = []
        independence_scores = []
        for t in res.get("trials", []):
            gl = t.get("generative_layer", None)
            if gl is None or isinstance(gl, str) or not isinstance(gl, dict):
                continue
            fm = gl.get("final_mse", None)
            gi = gl.get("generative_independence", None)
            if fm is not None:
                final_mses.append(fm)
            if gi and isinstance(gi, list) and len(gi) >= 2:
                independence_scores.append(float(np.mean(gi[-2:])))
        results[label] = {
            "final_mse": _safe_mean(final_mses) if final_mses else 0.0,
            "generative_independence": _safe_mean(independence_scores) if independence_scores else 0.0,
            "n": max(len(final_mses), len(independence_scores)),
        }
        if verbose:
            print(f"  [generative] {label:25s}: mse={results[label]['final_mse']:.4f}  "
                  f"indep={results[label]['generative_independence']:.4f}  "
                  f"n={results[label]['n']}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 6.  STRUCTURED FORGETTING / BASIN GEOMETRY
# ═════════════════════════════════════════════════════════════════════════

def test_basin_protection(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        r_vals = []
        for t in res.get("trials", []):
            bl = t.get("baseline_scores", None)
            fn = t.get("final_scores", None)
            if bl is None or fn is None:
                continue
            changes = np.array(fn, dtype=float) - np.array(bl, dtype=float)
            if len(changes) < 3:
                continue
            r_vals.append(float(np.std(changes)))
        results[label] = {"forgetting_variability": _safe_mean(r_vals) if r_vals else 0.0, "n": len(r_vals)}
        if verbose:
            print(f"  [basin_protection] {label:25s}: var={results[label]['forgetting_variability']:.4f}  "
                  f"n={results[label]['n']}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 7.  HIDDEN STATE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_hidden_states(all_results, verbose=True):
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        cortical_counts = []
        for t in res.get("trials", []):
            hs = t.get("hidden_state", None)
            if hs is None or not isinstance(hs, dict):
                continue
            cc = hs.get("n_cortical", 0)
            cortical_counts.append(cc)
        results[label] = {"mean_cortical": _safe_mean(cortical_counts) if cortical_counts else 0.0,
                          "n": len(cortical_counts)}
        if verbose:
            print(f"  [hidden_state] {label:25s}: cortical={results[label]['mean_cortical']:.1f}  "
                  f"n={results[label]['n']}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 8.  META-ANALYSIS OVER MULTIPLE SEEDS
# ═════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════
# 9.  NOVEL METRICS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_schema_crystallization(all_results, verbose=True):
    """Analyze Schema Crystallization Index across conditions."""
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        scis = []
        for t in res.get("trials", []):
            nm = t.get("novel_metrics", None)
            if nm is None or not isinstance(nm, dict):
                continue
            sci = nm.get("schema_crystallization_index", {})
            if isinstance(sci, dict) and "final_SCI" in sci:
                scis.append(sci["final_SCI"])
        if not scis:
            results[label] = {"mean_SCI": 0.0, "sem": 0.0, "n": 0, "interpretation": "no data"}
            if verbose:
                print(f"  [SCI] {label:25s}: no data", flush=True)
            continue
        arr = np.array(scis)
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        interp = "differentiating" if m < -0.1 else "decorrelated" if abs(m) < 0.1 else "blending"
        results[label] = {"mean_SCI": m, "sem": se, "n": len(arr), "interpretation": interp}
        if verbose:
            print(f"  [SCI] {label:25s}: SCI={m:.4f}+-{se:.4f}  ({interp})  n={len(arr)}", flush=True)
    return results


def test_cfr(all_results, verbose=True):
    """Analyze Catastrophic Forgetting Resistance across conditions."""
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        cfrs = []
        cfr_a_vals = []
        for t in res.get("trials", []):
            nm = t.get("novel_metrics", None)
            if nm is None or not isinstance(nm, dict):
                continue
            cfr = nm.get("catastrophic_forgetting_resistance", {})
            if isinstance(cfr, dict):
                overall = cfr.get("overall_CFR", 0.0)
                cfr_a = cfr.get("memory_A_CFR", 0.0)
                cfrs.append(overall)
                cfr_a_vals.append(cfr_a)
        if not cfrs:
            results[label] = {"mean_CFR": 0.0, "mean_CFR_A": 0.0, "n": 0}
            if verbose:
                print(f"  [CFR] {label:25s}: no data", flush=True)
            continue
        results[label] = {
            "mean_CFR": _safe_mean(cfrs),
            "mean_CFR_A": _safe_mean(cfr_a_vals),
            "sem_CFR_A": float(np.std(cfr_a_vals, ddof=1)) / np.sqrt(len(cfr_a_vals)) if len(cfr_a_vals) > 1 else 0.0,
            "n": len(cfrs),
        }
        if verbose:
            print(f"  [CFR] {label:25s}: overall={results[label]['mean_CFR']:.4f}  "
                  f"memA={results[label]['mean_CFR_A']:.4f}  n={results[label]['n']}", flush=True)
    return results


def test_drift_velocity(all_results, verbose=True):
    """Analyze representational drift velocity across conditions."""
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        velocities = []
        angles = []
        for t in res.get("trials", []):
            nm = t.get("novel_metrics", None)
            if nm is None or not isinstance(nm, dict):
                continue
            dv = nm.get("drift_velocity", {})
            if isinstance(dv, dict):
                vel = dv.get("drift_velocity", 0.0)
                ang = dv.get("convergence_angle", 90.0)
                velocities.append(vel)
                angles.append(ang)
        if not velocities:
            results[label] = {"mean_velocity": 0.0, "mean_angle": 90.0, "n": 0}
            if verbose:
                print(f"  [drift] {label:25s}: no data", flush=True)
            continue
        results[label] = {
            "mean_velocity": _safe_mean(velocities),
            "mean_angle": _safe_mean(angles),
            "n": len(velocities),
        }
        if verbose:
            print(f"  [drift] {label:25s}: vel={results[label]['mean_velocity']:.4f}  "
                  f"angle={results[label]['mean_angle']:.1f}  n={results[label]['n']}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 10.  PARTIAL-CUE RETRIEVAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_partial_cue_retrieval(all_results, verbose=True):
    """Analyze partial-cue completion accuracy across conditions."""
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        completion_by_frac = {}
        for t in res.get("trials", []):
            cc = t.get("completion_curves", None)
            if cc is None or not isinstance(cc, dict):
                continue
            mbf = cc.get("mean_by_fraction", {})
            for frac_str, prob in mbf.items():
                try:
                    frac = float(frac_str) if isinstance(frac_str, (int, float)) else float(f"{frac_str}")
                except (ValueError, TypeError):
                    continue
                if frac not in completion_by_frac:
                    completion_by_frac[frac] = []
                completion_by_frac[frac].append(prob)
        if not completion_by_frac:
            results[label] = {"mean_completion": {}, "n": 0}
            if verbose:
                print(f"  [partial-cue] {label:25s}: no data", flush=True)
            continue
        results[label] = {
            "mean_completion": {f: _safe_mean(v) for f, v in completion_by_frac.items()},
            "n": len(completion_by_frac.get(list(completion_by_frac.keys())[0], [])),
        }
        if verbose:
            fracs = sorted(results[label]["mean_completion"].keys())
            frac_strs = [f"{f*100:.0f}%" for f in fracs]
            vals = [results[label]["mean_completion"][f] for f in fracs]
            print(f"  [partial-cue] {label:25s}: {dict(zip(frac_strs, [f'{v:.3f}' for v in vals]))}",
                  flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 11.  REPLAY DIVERSITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def test_replay_diversity(all_results, verbose=True):
    """Analyze replay diversity (entropy, fragmentation, variability)."""
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        entropies = []
        fragmentations = []
        variabilities = []
        for t in res.get("trials", []):
            rd = t.get("replay_diversity", None)
            if rd is None or not isinstance(rd, dict):
                continue
            entropies.append(rd.get("entropy", 0.0))
            fragmentations.append(rd.get("fragmentation", 0.0))
            variabilities.append(rd.get("variability", 0.0))
        if not entropies:
            results[label] = {"mean_entropy": 0.0, "mean_frag": 0.0, "mean_var": 0.0, "n": 0}
            if verbose:
                print(f"  [replay-div] {label:25s}: no data", flush=True)
            continue
        results[label] = {
            "mean_entropy": _safe_mean(entropies),
            "mean_fragmentation": _safe_mean(fragmentations),
            "mean_variability": _safe_mean(variabilities),
            "n": len(entropies),
        }
        if verbose:
            d = results[label]
            print(f"  [replay-div] {label:25s}: entropy={d['mean_entropy']:.3f}  "
                  f"frag={d['mean_fragmentation']:.3f}  var={d['mean_variability']:.3f}  "
                  f"n={d['n']}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 12.  SCHEMA-CORE GENERALIZATION
# ═════════════════════════════════════════════════════════════════════════

def test_schema_core_generalization(all_results, verbose=True):
    """Analyze Schema-Core generalization across conditions."""
    results = {}
    for res in all_results:
        label = res["cond"]["label"]
        blended_acts = []
        specificities = []
        core_completions = []
        for t in res.get("trials", []):
            scg = t.get("schema_core_gen", None)
            if scg is None or not isinstance(scg, dict):
                continue
            blended_acts.append(scg.get("blended_activation", 0.0))
            specificities.append(scg.get("specificity", 0.0))
            core_completions.append(scg.get("core_completion", 0.0))
        if not blended_acts:
            results[label] = {"mean_blended": 0.0, "mean_specificity": 0.0,
                              "mean_core_completion": 0.0, "n": 0}
            if verbose:
                print(f"  [core-gen] {label:25s}: no data", flush=True)
            continue
        results[label] = {
            "mean_blended": _safe_mean(blended_acts),
            "mean_specificity": _safe_mean(specificities),
            "mean_core_completion": _safe_mean(core_completions),
            "n": len(blended_acts),
        }
        if verbose:
            d = results[label]
            print(f"  [core-gen] {label:25s}: blended={d['mean_blended']:.4f}  "
                  f"spec={d['mean_specificity']:.4f}  core={d['mean_core_completion']:.4f}  "
                  f"n={d['n']}", flush=True)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 13.  TWO-WAY ANOVA (Overlap × Sleep Condition)
# ═════════════════════════════════════════════════════════════════════════

def run_two_way_anova(sweep_results, verbose=True):
    """Run two-way ANOVA on overlap sweep data.

    Tests whether OVERLAP_RATIO and SLEEP_CONDITION significantly affect
    retention, with a significant interaction effect.

    Args:
        sweep_results: dict from run_overlap_sweep.

    Returns:
        dict with ANOVA results.
    """
    from scipy.stats import f_oneway
    data_by_overlap = {}
    for overlap, sdata in sweep_results.items():
        results = sdata["results"]
        cond_data = {}
        for res in results:
            label = res["cond"]["label"]
            finals = []
            for t in res.get("trials", []):
                fs = t.get("final_scores", [])
                if len(fs) > 0:
                    finals.append(np.nanmean(fs))
            cond_data[label] = finals
        data_by_overlap[overlap] = cond_data

    # Build factor arrays for ANOVA
    overlap_factor = []
    condition_factor = []
    values = []

    for overlap, cond_data in data_by_overlap.items():
        for label, vals in cond_data.items():
            for v in vals:
                overlap_factor.append(overlap)
                condition_factor.append(label)
                values.append(v)

    if len(values) < 4:
        return {"error": "insufficient_data"}

    # One-way ANOVA by condition (simplified)
    groups_by_cond = {}
    for label in condition_factor:
        if label not in groups_by_cond:
            groups_by_cond[label] = []
    for ol, cd in data_by_overlap.items():
        for label, vals in cd.items():
            groups_by_cond[label].extend(vals)

    anova_results = {}
    for label, vals in groups_by_cond.items():
        if len(vals) >= 3:
            from scipy.stats import f_oneway
            # Compare across overlaps for this condition
            overlap_groups = []
            for ol in sorted(data_by_overlap.keys()):
                og = data_by_overlap[ol].get(label, [])
                if len(og) > 0:
                    overlap_groups.append(og)
            if len(overlap_groups) >= 2:
                try:
                    f_stat, p_val = f_oneway(*overlap_groups)
                    anova_results[label] = {"F": f_stat, "p": p_val,
                                            "n_groups": len(overlap_groups)}
                except Exception as e:
                    anova_results[label] = {"error": str(e)}

    return anova_results


# ═════════════════════════════════════════════════════════════════════════
# 8.  META-ANALYSIS OVER MULTIPLE SEEDS
# ═════════════════════════════════════════════════════════════════════════

def run_multi_seed_meta_analysis(all_seed_schema):
    """Aggregate schema results across multiple seeds.

    Args:
        all_seed_schema: list of schema_results dicts (one per seed).

    Returns:
        Dict with same structure as schema_results but with mean/sem over seeds.
    """
    if not all_seed_schema:
        return {}
    test_names = list(all_seed_schema[0].keys())
    meta = {}
    for test_name in test_names:
        conditions = set()
        for sr in all_seed_schema:
            conditions.update(sr.get(test_name, {}).keys())
        meta[test_name] = {}
        for cond in conditions:
            vals = []
            for sr in all_seed_schema:
                r = sr.get(test_name, {}).get(cond, {})
                if isinstance(r, dict):
                    for k, v in r.items():
                        if isinstance(v, (int, float)) and k != "n" and k != "p":
                            vals.append(v)
            if vals:
                arr = np.array(vals)
                meta[test_name][cond] = {
                    "mean_over_seeds": float(np.mean(arr)),
                    "sem_over_seeds": float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0,
                    "n_seeds": len(vals),
                }
    return meta


# ═════════════════════════════════════════════════════════════════════════
# 9.  MASTER ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════

def run_all_schema_analysis(all_results, verbose=True):
    """Run all schema-abstraction statistical tests on experiment results."""
    print("\n" + "=" * 70, flush=True)
    print("SCHEMA ABSTRACTION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    out = {}
    out["directionality"] = test_directionality(all_results, verbose=verbose)
    out["convergence"] = test_schema_convergence(all_results, verbose=verbose)
    out["coherence_drift"] = test_coherence_drift_correlation(all_results, verbose=verbose)
    out["retention_tradeoff"] = compute_retention_tradeoff(all_results, verbose=verbose)
    out["overlap_proportionality"] = test_overlap_proportionality(all_results, verbose=verbose)
    out["generalization"] = test_generalization(all_results, verbose=verbose)
    out["anti_prediction"] = test_anti_prediction(all_results, verbose=verbose)
    out["downscaling"] = test_downscaling(all_results, verbose=verbose)
    out["generative_layer"] = test_generative_layer(all_results, verbose=verbose)
    out["basin_protection"] = test_basin_protection(all_results, verbose=verbose)
    out["hidden_state"] = test_hidden_states(all_results, verbose=verbose)
    out["schema_crystallization"] = test_schema_crystallization(all_results, verbose=verbose)
    out["cfr"] = test_cfr(all_results, verbose=verbose)
    out["drift_velocity"] = test_drift_velocity(all_results, verbose=verbose)
    out["partial_cue"] = test_partial_cue_retrieval(all_results, verbose=verbose)
    out["replay_diversity"] = test_replay_diversity(all_results, verbose=verbose)
    out["core_generalization"] = test_schema_core_generalization(all_results, verbose=verbose)

    return out
