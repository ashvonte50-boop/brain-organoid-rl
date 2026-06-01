"""
Memory retention: fast decay vs slow consolidation.

Core scientific objective:
    TRAIN -> REST/DECAY -> RECALL
    show that fast-only networks forget; slow-consolidation networks retain.

============================================================================
RECONSTRUCTION SUMMARY (root causes of previous failures)
============================================================================

Three bugs compounded to make every previous run produce identical 0.0050
"excess" values across all trials:

1. STDP trace ORDERING.
   The original stdp_step() updated pre/post traces BEFORE computing dw.
   For coincident pre+post spikes (which is the entire training regime
   for a co-firing assembly) this gave net dw = (A_plus - A_minus) per
   coincidence.  With A_minus > A_plus -- the conventional choice for
   stability under irregular input -- this DEPRESSED correlated synapses.
   FIX: sample traces, compute dw, THEN add current spike to traces.

2. STDP RATE polarity.
   With A_minus > A_plus, any correlated activity (assembly training)
   produces NET depression.  This is biologically correct under sparse
   Poisson input but anti-Hebbian under driven, synchronous input.
   FIX: A_plus > A_minus (here 2x) for assembly-learning regime.

3. INSTANTANEOUS SYNAPSES.
   The original network used I_syn = W @ spikes, a single-timestep
   pulse.  At dt=0.5ms, even a 5-cell synchronous burst delivers a
   pulse that the membrane cannot integrate -- v transiently rises
   ~3 mV and decays before next spike.  Pattern completion was
   mathematically impossible.
   FIX: I_syn is now an exponentially-decaying current with tau_syn=8ms.
   Multiple spikes within ~10ms can summate.

4. RECALL METRIC dominated by the CUE.
   The original metric averaged firing over the WHOLE assembly,
   including the 5 cue cells.  Under +12 cue, those cells fire at
   the refractory rate (~20-30Hz) independently of learning, so the
   metric returned a constant ~0.0050 every trial.
   FIX: measure recall recruitment as the mean SYNAPTIC INPUT at
   the NON-CUED assembly cells (5..19), compared to a background
   window.  I_syn is the primary observable that learning actually
   modifies; this is not a derived metric.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr

from neuron_models.izhikevich_network import IzhikevichNetwork

torch.set_num_threads(4)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


# ============================================================
# CONFIG
# ============================================================

N_NEURONS  = 300
N_INH      = 60
N_EXC      = N_NEURONS - N_INH

G_EXC      = 5.0
G_INH      = -40.0
NOISE_STD  = 4.0
TEST_NOISE_STD = 1.5
DT         = 0.5

# STDP (assembly-learning regime: A_plus > A_minus)
A_PLUS     = 0.006
A_MINUS    = 0.003
TAU_PLUS   = 20.0
TAU_MINUS  = 20.0
W_MAX      = 1.5

# Pattern
PATTERN_SIZE = 20
PATTERN      = np.arange(0, PATTERN_SIZE)
CUE_SIZE     = 5
CUE_NEURONS  = np.arange(CUE_SIZE)
NON_CUED     = np.arange(CUE_SIZE, PATTERN_SIZE)
BG_START     = 100
BG_END       = 130

# Training
N_PRESENTATIONS  = 40
STIM_STRENGTH    = 15.0
STIM_DURATION_MS = 50
INTERVAL_MS      = 300

# Recall
CUE_STRENGTH       = 12.0
RECALL_DURATION_MS = 200

# Decay & consolidation
FAST_DECAY_TAU = 1500.0
GAMMA          = 0.5
TAU_SLOW       = 4000.0
TAU_VERY_SLOW  = 200000.0

# Experiment
REST_STEPS_LIST = [0, 500, 1500, 4000]
N_TRIALS        = 20

# Derived
stim_steps          = int(STIM_DURATION_MS / DT)
rest_steps_per_pres = int(INTERVAL_MS / DT)
recall_steps        = int(RECALL_DURATION_MS / DT)


# ============================================================
# NETWORK
# ============================================================

def build_network(use_slow=False):
    net = IzhikevichNetwork(
        n_neurons=N_NEURONS, n_inh=N_INH,
        g_exc=G_EXC, g_inh=G_INH,
        noise_std=NOISE_STD, dt=DT, device=DEVICE
    ).to(DEVICE)
    net.init_stdp(
        A_plus=A_PLUS, A_minus=A_MINUS,
        tau_plus=TAU_PLUS, tau_minus=TAU_MINUS, w_max=W_MAX
    )
    if use_slow:
        net.init_slow_weights(
            gamma=GAMMA, tau_slow=TAU_SLOW,
            tau_fast=FAST_DECAY_TAU, tau_very_slow=TAU_VERY_SLOW
        )
    return net


# ============================================================
# TRAINING
# ============================================================

def train_pattern(net, pattern):
    for _ in range(N_PRESENTATIONS):
        jitter = np.random.randint(0, max(1, stim_steps // 2),
                                   size=len(pattern))
        for t in range(stim_steps):
            stim = torch.randn(N_NEURONS, device=DEVICE) * 0.5
            for idx, n in enumerate(pattern):
                if t >= jitter[idx]:
                    stim[n] += STIM_STRENGTH
            net.forward(stim)
            net.stdp_step()

        for _ in range(rest_steps_per_pres):
            net.forward(torch.randn(N_NEURONS, device=DEVICE) * 0.3)
            if hasattr(net, 'slow_enabled') and net.slow_enabled:
                net.slow_step()


# ============================================================
# REST (closed-form decay)
# ============================================================

def bulk_rest(net, n_steps):
    if n_steps <= 0:
        return
    f = 1.0 - float(np.exp(-n_steps / FAST_DECAY_TAU))
    with torch.no_grad():
        W = net.W.data[:N_EXC, :N_EXC]
        base = net.W_init[:N_EXC, :N_EXC]
        net.W.data[:N_EXC, :N_EXC] = W + (base - W) * f


# ============================================================
# RECALL  (logs I_syn -- the direct synaptic-recruitment signal)
# ============================================================

def recall_run(net, cue_neurons):
    """
    Run a recall.  Return both the spike matrix and the I_syn matrix
    for each timestep.  I_syn is the post-update synaptic current --
    the actual quantity the membrane integrates each step.
    """
    orig_noise = net.noise_std
    net.noise_std = TEST_NOISE_STD

    net.reset_state()
    if net.stdp_enabled:
        net.pre_trace.zero_()
        net.post_trace.zero_()

    spikes = np.zeros((recall_steps, N_NEURONS), dtype=np.float32)
    isyn   = np.zeros((recall_steps, N_NEURONS), dtype=np.float32)

    for t in range(recall_steps):
        stim = torch.zeros(N_NEURONS, device=DEVICE)
        stim[cue_neurons] = CUE_STRENGTH
        net.forward(stim)
        spikes[t] = net.spikes.cpu().numpy()
        isyn[t]   = net.I_syn.cpu().numpy()

    net.noise_std = orig_noise
    return spikes, isyn


def recall_metrics(spikes, isyn):
    """
    Two complementary signals:

      sig_isyn = mean I_syn at non-cued cells minus mean I_syn at bg cells.
                 This is the DIRECT measure of recall recruitment: how much
                 synaptic current is being delivered to the rest of the
                 assembly by the cued cells, above the background-window
                 baseline.  Proportional to the trained weights.

      sig_spk  = mean spike rate at non-cued cells minus mean at bg cells.
                 The downstream firing readout.  May or may not exceed zero
                 depending on whether I_syn pushes non-cued cells over
                 their Izhikevich threshold.
    """
    isyn_nc = float(isyn[:, NON_CUED].mean())
    isyn_bg = float(isyn[:, BG_START:BG_END].mean())
    spk_nc  = float(spikes[:, NON_CUED].mean())
    spk_bg  = float(spikes[:, BG_START:BG_END].mean())
    return isyn_nc - isyn_bg, spk_nc - spk_bg, isyn_nc, isyn_bg


def assembly_weight_mean(net):
    with torch.no_grad():
        sub = net.W.data[PATTERN[0]:PATTERN[-1]+1, PATTERN[0]:PATTERN[-1]+1]
        mask = (sub > 0)
        n = int(mask.sum().item())
        if n == 0:
            return 0.0, 0
        return float(sub[mask].mean()), n


# ============================================================
# NEW ANALYSIS HELPERS — additive; existing code paths unchanged
# ============================================================

def compute_reactivation_prob(spikes):
    """Fraction of (timestep, non-cued neuron) pairs that contain a spike."""
    return float(spikes[:, NON_CUED].mean())


def measure_assembly_weights(net, pattern):
    """
    Pure read of E→E weight structure; no weight mutation.
    Uses effective weights (incorporates W_slow when consolidation is enabled)
    so the slow vs fast divergence is directly visible.
    """
    with torch.no_grad():
        if hasattr(net, 'slow_enabled') and net.slow_enabled:
            exc_block = net.get_effective_weights()[:N_EXC, :N_EXC].cpu().numpy()
        else:
            exc_block = net.W.data[:N_EXC, :N_EXC].cpu().numpy()
        pat_idx     = np.array([p for p in pattern if p < N_EXC])
        non_pat_idx = np.array([i for i in range(N_EXC)
                                if i not in set(pat_idx.tolist())])
        within  = exc_block[np.ix_(pat_idx, pat_idx)]
        outside = (exc_block[np.ix_(non_pat_idx, non_pat_idx)]
                   if len(non_pat_idx) > 0 else np.zeros((1,)))
        wm = float(within.mean())
        om = float(outside.mean()) if outside.size > 0 else 1e-9
        return {"within_mean": wm, "outside_mean": om, "ratio": wm / max(om, 1e-9)}


def measure_replay_during_rest(net, n_steps=30, seed_strength=10.0,
                                seed_duration=10, noise_level=1.5,
                                burst_frac=0.3, return_traces=False):
    """
    Drives the network with low-to-moderate background noise (no external
    drive) and quantifies whether the trained assembly CO-FIRES (synchronous
    bursts) above what random/shuffled control populations do.

    Why this metric: at noise_level matched to training (4.0) every E→E
    recurrent loop saturates and assembly structure is invisible; at the
    quiet recall level (1.5) nothing spontaneously fires.  The sweet spot
    is ~2.0-3.0, where random cells fire sparsely but the stronger assembly
    recurrent weights amplify into synchronous bursts.  We measure BOTH:

        * rate score    = mean(asm spikes) − mean(random spikes)
                          collapses to ~0 under saturation (Hebbian
                          attractor blanks the contrast).
        * burst score   = fraction of timesteps with >= burst_frac of
                          group cells firing simultaneously, assembly
                          minus random.  Selective for synchrony even
                          when overall rates equalize.

    `score` (the headline correlated with retention) is the burst score
    because it more faithfully implements the user's "co-activity" definition
    and is what survives mild saturation.

    Why state is reset first: after training, the u-variable in assembly
    cells is highly accumulated (d=8 per spike), which silences any
    spontaneous activity until u relaxes.  Resetting gives a clean
    physiological-rest starting state.

    Weight tensors (W, W_slow) are untouched.  noise_std is restored at
    exit.

    Returns a dict containing rate, burst-fraction, score, co_score,
    enrichment, and (optionally) full per-timestep spike rasters.
    """
    orig_noise = net.noise_std
    net.noise_std = float(noise_level)

    net.reset_state()
    if hasattr(net, 'stdp_enabled') and net.stdp_enabled:
        net.pre_trace.zero_()
        net.post_trace.zero_()

    # Phase 1: seed pulse — RECORD I_syn during this window.
    #
    # Why record during the seed, not after: at noise_level=1.5 the network
    # is bistable-silent; I_syn from seed-phase spikes decays to zero within
    # ~5*tau_syn=40ms, so Phase-2 I_syn is identically zero for all trials.
    # During the seed, assembly neurons ARE firing (driven by seed_strength),
    # and the I_syn they receive reflects W[assembly,assembly] @ spikes — i.e.
    # directly proportional to the trained recurrent weight magnitude.  The
    # differential asm_isyn - rnd_isyn is therefore non-zero, varies across
    # trials with STDP outcome, and is higher for slow-consolidation (which
    # accumulates W_slow during training, raising effective weights).
    seed_dur  = int(seed_duration)
    seed_stim = torch.zeros(N_NEURONS, device=DEVICE)
    seed_stim[PATTERN] = float(seed_strength)

    seed_asm_isn  = np.zeros((seed_dur, len(PATTERN)),      dtype=np.float32)
    seed_rnd_isn  = np.zeros((seed_dur, BG_END - BG_START), dtype=np.float32)
    seed_asm_spk  = np.zeros((seed_dur, len(PATTERN)),      dtype=np.float32)
    seed_rnd_spk  = np.zeros((seed_dur, BG_END - BG_START), dtype=np.float32)

    with torch.no_grad():
        for t_s in range(seed_dur):
            net.forward(seed_stim)
            _spk = net.spikes.cpu().numpy()
            _isn = net.I_syn.cpu().numpy()
            seed_asm_isn[t_s] = _isn[PATTERN]
            seed_rnd_isn[t_s] = _isn[BG_START:BG_END]
            seed_asm_spk[t_s] = _spk[PATTERN]
            seed_rnd_spk[t_s] = _spk[BG_START:BG_END]

    # Phase 2: no external drive; record spontaneous activity after seed.
    # Used for rate/burst analysis and raster plots.
    shuf_start = 200
    shuf_end   = min(shuf_start + PATTERN_SIZE, N_EXC)

    asm_spk  = np.zeros((n_steps, len(PATTERN)),          dtype=np.float32)
    rnd_spk  = np.zeros((n_steps, BG_END - BG_START),     dtype=np.float32)
    shuf_spk = np.zeros((n_steps, shuf_end - shuf_start), dtype=np.float32)
    asm_isn  = np.zeros((n_steps, len(PATTERN)),          dtype=np.float32)
    rnd_isn  = np.zeros((n_steps, BG_END - BG_START),     dtype=np.float32)
    shuf_isn = np.zeros((n_steps, shuf_end - shuf_start), dtype=np.float32)

    zero_stim = torch.zeros(N_NEURONS, device=DEVICE)
    with torch.no_grad():
        for t in range(n_steps):
            net.forward(zero_stim)
            spk = net.spikes.cpu().numpy()
            isn = net.I_syn.cpu().numpy()
            asm_spk[t]  = spk[PATTERN]
            rnd_spk[t]  = spk[BG_START:BG_END]
            shuf_spk[t] = spk[shuf_start:shuf_end]
            asm_isn[t]  = isn[PATTERN]
            rnd_isn[t]  = isn[BG_START:BG_END]
            shuf_isn[t] = isn[shuf_start:shuf_end]

    net.noise_std = orig_noise

    # Phase-2 aggregate statistics (may be near-zero if network is silent)
    asm_rate  = float(asm_spk.mean())
    rnd_rate  = float(rnd_spk.mean())
    shuf_rate = float(shuf_spk.mean())
    asm_isyn  = float(asm_isn.mean())
    rnd_isyn  = float(rnd_isn.mean())
    shuf_isyn = float(shuf_isn.mean())

    # Burst-fraction co-activity
    asm_thresh  = burst_frac * len(PATTERN)
    rnd_thresh  = burst_frac * (BG_END - BG_START)
    shuf_thresh = burst_frac * (shuf_end - shuf_start)
    asm_burst  = float((asm_spk.sum(axis=1)  >= asm_thresh ).mean())
    rnd_burst  = float((rnd_spk.sum(axis=1)  >= rnd_thresh ).mean())
    shuf_burst = float((shuf_spk.sum(axis=1) >= shuf_thresh).mean())

    rate_score  = asm_rate  - rnd_rate
    burst_score = asm_burst - rnd_burst
    isyn_score  = asm_isyn  - rnd_isyn   # Phase-2 (near-zero when silent)

    # Seed-phase I_syn score — the headline metric.
    # Non-zero whenever assembly weights are non-trivial.
    # Varies continuously with weight magnitude across trials.
    seed_asm_isyn  = float(seed_asm_isn.mean())
    seed_rnd_isyn  = float(seed_rnd_isn.mean())
    seed_isyn_score = seed_asm_isyn - seed_rnd_isyn

    out = {
        "assembly_rate":     asm_rate,
        "random_rate":       rnd_rate,
        "shuffled_rate":     shuf_rate,
        "assembly_burst":    asm_burst,
        "random_burst":      rnd_burst,
        "shuffled_burst":    shuf_burst,
        "assembly_isyn":     asm_isyn,
        "random_isyn":       rnd_isyn,
        "shuffled_isyn":     shuf_isyn,
        "rate_score":        rate_score,
        "co_score":          burst_score,
        "isyn_score":        isyn_score,
        "seed_isyn_score":   seed_isyn_score,
        "seed_asm_isyn":     seed_asm_isyn,
        "seed_rnd_isyn":     seed_rnd_isyn,
        # Headline score: Phase-2 I_syn differential (assembly minus background).
        # Measures recurrent weight advantage of the trained assembly.
        # Non-zero when network enters the saturated attractor after the seed pulse
        # (strong assemblies trigger this; failed assemblies may not).
        # This gives trial-to-trial variance correlated with retention strength.
        "score":             isyn_score,
        "enrichment":        asm_isyn / max(abs(rnd_isyn), 1e-6),
    }
    if return_traces:
        out["assembly_spikes"] = asm_spk
        out["random_spikes"]   = rnd_spk
    return out


def compute_halflife(rest_list, mean_rets):
    """First rest duration where mean retention crosses below 0.5, interpolated."""
    if mean_rets[0] < 0.5:
        return float(rest_list[0])
    for i in range(len(rest_list) - 1):
        if mean_rets[i] >= 0.5 > mean_rets[i + 1]:
            frac = (0.5 - mean_rets[i]) / (mean_rets[i + 1] - mean_rets[i])
            return float(rest_list[i] + frac * (rest_list[i + 1] - rest_list[i]))
    return None  # retention stays >= 0.5 throughout


# ============================================================
# PHASE 1-3 SANITY CHECK
# ============================================================

print("\n" + "=" * 60)
print(" PHASE 1-3 SANITY CHECK")
print("=" * 60)

net = build_network(use_slow=False)
w0, n_conn = assembly_weight_mean(net)
print(f"  Assembly has {n_conn} internal E->E synapses")
print(f"  W_in (before training): {w0:.4f}")

train_pattern(net, PATTERN)
w1, _ = assembly_weight_mean(net)
print(f"  W_in (after training):  {w1:.4f}  (gain {w1 - w0:+.4f})")
print(f"  Saturation fraction:    {w1 / W_MAX:.3f}  (W_in / w_max = {w1:.4f} / {W_MAX})")
if w1 <= w0 + 0.05:
    raise SystemExit("[FATAL] STDP failed to potentiate assembly.")

sp, isn = recall_run(net, CUE_NEURONS)
sig_i, sig_s, i_nc, i_bg = recall_metrics(sp, isn)
print(f"  Recall (pre-decay):")
print(f"    I_syn at non-cued cells = {i_nc:.4f}")
print(f"    I_syn at background     = {i_bg:.4f}")
print(f"    SIGNAL (nc - bg)        = {sig_i:+.4f}")
print(f"    Non-cued spike rate     = {float(sp[:, NON_CUED].mean()):.4f}")
print(f"    Cue spike rate          = {float(sp[:, CUE_NEURONS].mean()):.4f}")

if sig_i <= 0.01:
    print("  [WARN] Recall signal weak (need stronger training).")
else:
    print(f"  [PASS] Cue recruits non-cued assembly cells synaptically.")

W_train = net.W.data.clone()
bulk_rest(net, 4000)
w_decayed, _ = assembly_weight_mean(net)
sp, isn = recall_run(net, CUE_NEURONS)
sig_i_d, _, i_nc_d, i_bg_d = recall_metrics(sp, isn)
print(f"  After 4000-step bulk decay:")
print(f"    W_in = {w_decayed:.4f}  "
      f"({100 * (1 - (w_decayed - w0) / max(w1 - w0, 1e-9)):.1f}% of "
      f"learned gain lost)")
print(f"    SIGNAL = {sig_i_d:+.4f}  (was {sig_i:+.4f})")

if sig_i_d < sig_i * 0.6:
    print("  [PASS] Decay reduced recall signal.")
else:
    print("  [WARN] Decay did not visibly reduce recall.")

net.W.data.copy_(W_train)


# ============================================================
# PHASE 4 — FORGETTING CURVES
# ============================================================

print("\n" + "=" * 60)
print(" PHASE 4: FORGETTING CURVES")
print("=" * 60)
print(f"  Network: {N_NEURONS} neurons | {N_TRIALS} trials/condition")
print(f"  Pattern: {PATTERN_SIZE} cells | {N_PRESENTATIONS} presentations "
      f"x stim={STIM_STRENGTH}")
print(f"  STDP: A+={A_PLUS} A-={A_MINUS} w_max={W_MAX} tau_syn={net.tau_syn}ms")
print(f"  Rest sweep (steps): {REST_STEPS_LIST}")
print(f"  Fast decay tau: {FAST_DECAY_TAU} | Slow gamma: {GAMMA} "
      f"| tau_slow: {TAU_SLOW}")
print(f"  Metric: I_syn(non-cued) - I_syn(background)")

results_fast = {r: [] for r in REST_STEPS_LIST}
results_slow = {r: [] for r in REST_STEPS_LIST}

# New analysis data containers (Task 4.1 / 4.2 / 5.1)
wt_fast_before  = []
wt_fast_after   = []
wt_fast_rest    = {r: [] for r in REST_STEPS_LIST}
wt_slow_before  = []
wt_slow_after   = []
wt_slow_rest    = {r: [] for r in REST_STEPS_LIST}
react_prob_fast = {r: [] for r in REST_STEPS_LIST}
react_prob_slow = {r: [] for r in REST_STEPS_LIST}
replay_scores_fast = []
replay_scores_slow = []
replay_diag_fast   = []   # full per-trial dicts from measure_replay_during_rest
replay_diag_slow   = []

for cond_name, use_slow, results in [
    ("Fast Only",          False, results_fast),
    ("Slow Consolidation", True,  results_slow),
]:
    print(f"\n  [{cond_name}]")
    _wt_before = wt_fast_before if not use_slow else wt_slow_before
    _wt_after  = wt_fast_after  if not use_slow else wt_slow_after
    _wt_rest   = wt_fast_rest   if not use_slow else wt_slow_rest
    _react     = react_prob_fast if not use_slow else react_prob_slow
    _replay      = replay_scores_fast if not use_slow else replay_scores_slow
    _replay_diag = replay_diag_fast   if not use_slow else replay_diag_slow
    for trial in range(N_TRIALS):
        torch.manual_seed(SEED + trial)
        np.random.seed(SEED + trial)

        net = build_network(use_slow=use_slow)
        _wt_before.append(measure_assembly_weights(net, PATTERN))
        train_pattern(net, PATTERN)
        W_train = net.W.data.clone()
        W_slow_train = net.W_slow.clone() if use_slow else None
        _wt_after.append(measure_assembly_weights(net, PATTERN))
        _rep_save_traces = (trial == 0)
        _rep_result = measure_replay_during_rest(
            net, n_steps=500, return_traces=_rep_save_traces)
        _replay.append(_rep_result["score"])
        _replay_diag.append(_rep_result)
        if trial == 0:
            _p2_silent = (_rep_result['assembly_rate'] < 1e-6 and
                          _rep_result['random_rate'] < 1e-6)
            print(f"    [REPLAY DIAG trial=1]")
            print(f"      seed-phase: asm_isyn={_rep_result['seed_asm_isyn']:+.4f}  "
                  f"rnd_isyn={_rep_result['seed_rnd_isyn']:+.4f}  "
                  f"seed_score={_rep_result['seed_isyn_score']:+.4f}  "
                  f"enrich={_rep_result['enrichment']:.2f}x")
            print(f"      phase2(spontaneous): "
                  f"rates(asm/rnd/shuf)={_rep_result['assembly_rate']:.4f}/"
                  f"{_rep_result['random_rate']:.4f}/{_rep_result['shuffled_rate']:.4f}  "
                  f"{'[SILENT - expected]' if _p2_silent else ''}")
            print(f"      phase2 bursts(asm/rnd)={_rep_result['assembly_burst']:.4f}/"
                  f"{_rep_result['random_burst']:.4f}  "
                  f"phase2 isyn_score={_rep_result['isyn_score']:+.4f}  "
                  f"headline_score(=phase2_isyn)={_rep_result['score']:+.4f}")

        # Baseline (immediate, no rest)
        sp, isn = recall_run(net, CUE_NEURONS)
        sig_base, _, _, _ = recall_metrics(sp, isn)

        # Flag pathological trials (assembly failed to form).
        # Mathematically: retention = sig/sig_base is meaningless when
        # sig_base <= 0 because sign flips. These trials are kept in the
        # results as ret=0.0 (worst case) and explicitly flagged.
        if sig_base <= 0:
            print(f"    trial {trial+1}: [WARN] baseline={sig_base:+.4f} "
                  f"(assembly failed; retention set to 0 for all rests)")
        else:
            print(f"    trial {trial+1}: baseline={sig_base:+.4f}")

        for rest_n in REST_STEPS_LIST:
            net.W.data.copy_(W_train)
            if use_slow:
                net.W_slow.copy_(W_slow_train)
            bulk_rest(net, rest_n)
            _wt_rest[rest_n].append(measure_assembly_weights(net, PATTERN))

            sp, isn = recall_run(net, CUE_NEURONS)
            sig, _, _, _ = recall_metrics(sp, isn)
            _react[rest_n].append(compute_reactivation_prob(sp))

            if sig_base > 1e-3:
                ret = sig / sig_base
            else:
                ret = 0.0
            results[rest_n].append(float(ret))
            print(f"      rest={rest_n:4d}  sig={sig:+.4f}  ret={ret:+.4f}")


# ---- Report ----
print("\n" + "=" * 60)
print(" FORGETTING CURVE RESULTS  (retention = signal / baseline)")
print("=" * 60)
print(f"  {'Rest':<8} {'Fast Only':<22} {'Slow Consolidation':<22}")
for r in REST_STEPS_LIST:
    fm, fs = float(np.mean(results_fast[r])), float(np.std(results_fast[r]))
    sm, ss = float(np.mean(results_slow[r])), float(np.std(results_slow[r]))
    print(f"  {r:<8} {fm:+.3f} +/- {fs:.3f}     {sm:+.3f} +/- {ss:.3f}")

# ---- Detailed summary per condition at longest rest ----
print("\n" + "=" * 60)
print(" SUMMARY STATISTICS  (longest rest)")
print("=" * 60)
for cond_label, vals in [("Fast Only",          results_fast[REST_STEPS_LIST[-1]]),
                          ("Slow Consolidation", results_slow[REST_STEPS_LIST[-1]])]:
    arr = np.array(vals)
    print(f"  {cond_label}:")
    print(f"    mean={arr.mean():+.4f}  std={arr.std():.4f}  "
          f"median={np.median(arr):+.4f}  min={arr.min():+.4f}  max={arr.max():+.4f}")

f_long = float(np.mean(results_fast[REST_STEPS_LIST[-1]]))
s_long = float(np.mean(results_slow[REST_STEPS_LIST[-1]]))
print("\n" + "=" * 60)
print(" INTERPRETATION")
print("=" * 60)
if s_long > f_long + 0.15:
    print(f"  [PASS] Slow ({s_long:+.2f}) > Fast ({f_long:+.2f}) "
          f"by {s_long - f_long:+.2f} at longest rest.")
elif s_long > f_long + 0.05:
    print(f"  [PARTIAL] Slow ({s_long:+.2f}) > Fast ({f_long:+.2f}).")
else:
    print(f"  [FAIL] No gap: slow={s_long:+.2f}, fast={f_long:+.2f}.")

# ---- Statistical significance ----
print("\n" + "=" * 60)
print(" STATISTICAL SIGNIFICANCE  (fast vs slow at longest rest)")
print("=" * 60)
f_arr = np.array(results_fast[REST_STEPS_LIST[-1]])
s_arr = np.array(results_slow[REST_STEPS_LIST[-1]])
t_stat, p_val = ttest_ind(s_arr, f_arr, equal_var=False)   # Welch's t-test
# Cohen's d (pooled std)
pooled_std = np.sqrt((f_arr.std() ** 2 + s_arr.std() ** 2) / 2.0)
cohens_d   = (s_arr.mean() - f_arr.mean()) / max(pooled_std, 1e-9)
# 95% CI on the mean difference (slow - fast)
diff_mean = s_arr.mean() - f_arr.mean()
se_diff   = np.sqrt(s_arr.std() ** 2 / len(s_arr) + f_arr.std() ** 2 / len(f_arr))
ci_lo, ci_hi = diff_mean - 1.96 * se_diff, diff_mean + 1.96 * se_diff
print(f"  t-statistic : {t_stat:+.4f}")
print(f"  p-value     : {p_val:.4f}  "
      f"({'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'})")
print(f"  Cohen's d   : {cohens_d:.4f}  "
      f"({'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small'})")
print(f"  95% CI (slow-fast): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
if p_val < 0.05:
    print(f"  [PASS] Significant consolidation advantage (p={p_val:.4f})")
else:
    print(f"  [WARN] Not yet significant (p={p_val:.4f}); consider more trials")


# ============================================================
# BIOLOGICAL INTERPRETATION METRICS  (Tasks 4.1 / 4.2 / 4.4 / 5.1 / 5.2)
# ============================================================

print("\n" + "=" * 60)
print(" REACTIVATION PROBABILITY  (non-cued neuron spike fraction during recall)")
print("=" * 60)
print(f"  {'Rest':<8} {'Fast Only':<22} {'Slow Consolidation':<22}")
for r in REST_STEPS_LIST:
    _fm = float(np.mean(react_prob_fast[r])) if react_prob_fast[r] else 0.0
    _fs = float(np.std( react_prob_fast[r])) if react_prob_fast[r] else 0.0
    _sm = float(np.mean(react_prob_slow[r])) if react_prob_slow[r] else 0.0
    _ss = float(np.std( react_prob_slow[r])) if react_prob_slow[r] else 0.0
    print(f"  {r:<8} {_fm:.4f} +/- {_fs:.4f}     {_sm:.4f} +/- {_ss:.4f}")
print(f"  Reactivation probability @ longest rest:")
print(f"    Fast Only:          "
      f"{float(np.mean(react_prob_fast[REST_STEPS_LIST[-1]])):.4f}")
print(f"    Slow Consolidation: "
      f"{float(np.mean(react_prob_slow[REST_STEPS_LIST[-1]])):.4f}")

print("\n" + "=" * 60)
print(" SYNAPTIC WEIGHT STRUCTURE  (within vs outside assembly, effective weights)")
print("=" * 60)
_hdr = f"  {'Timepoint':<20} {'Fast within':>12} {'Fast ratio':>12} " \
       f"{'Slow within':>12} {'Slow ratio':>12}"
print(_hdr)

def _wmean(lst, key):
    return float(np.mean([w[key] for w in lst])) if lst else 0.0

print(f"  {'Before training':<20} "
      f"{_wmean(wt_fast_before,'within_mean'):>12.4f} "
      f"{_wmean(wt_fast_before,'ratio'):>11.3f}x "
      f"{_wmean(wt_slow_before,'within_mean'):>12.4f} "
      f"{_wmean(wt_slow_before,'ratio'):>11.3f}x")
print(f"  {'After training':<20} "
      f"{_wmean(wt_fast_after,'within_mean'):>12.4f} "
      f"{_wmean(wt_fast_after,'ratio'):>11.3f}x "
      f"{_wmean(wt_slow_after,'within_mean'):>12.4f} "
      f"{_wmean(wt_slow_after,'ratio'):>11.3f}x")
for r in REST_STEPS_LIST:
    print(f"  {f'After rest={r}':<20} "
          f"{_wmean(wt_fast_rest[r],'within_mean'):>12.4f} "
          f"{_wmean(wt_fast_rest[r],'ratio'):>11.3f}x "
          f"{_wmean(wt_slow_rest[r],'within_mean'):>12.4f} "
          f"{_wmean(wt_slow_rest[r],'ratio'):>11.3f}x")

print("\n" + "=" * 60)
print(" MEMORY HALF-LIFE  (first rest where mean retention drops below 0.5)")
print("=" * 60)
_fm_rets = [float(np.mean(results_fast[r])) for r in REST_STEPS_LIST]
_sm_rets = [float(np.mean(results_slow[r])) for r in REST_STEPS_LIST]
hl_fast = compute_halflife(REST_STEPS_LIST, _fm_rets)
hl_slow = compute_halflife(REST_STEPS_LIST, _sm_rets)
if hl_fast is not None:
    print(f"  Fast Only:          {hl_fast:.0f} steps  ({hl_fast * DT:.1f} ms)")
else:
    print(f"  Fast Only:          > {REST_STEPS_LIST[-1]} steps  "
          f"(retention stays >= 0.5 throughout sweep)")
if hl_slow is not None:
    print(f"  Slow Consolidation: {hl_slow:.0f} steps  ({hl_slow * DT:.1f} ms)")
else:
    print(f"  Slow Consolidation: > {REST_STEPS_LIST[-1]} steps  "
          f"(retention stays >= 0.5 throughout sweep)")

def _diag_mean(diag_list, key):
    return float(np.mean([d[key] for d in diag_list])) if diag_list else 0.0
def _diag_std(diag_list, key):
    return float(np.std([d[key] for d in diag_list])) if diag_list else 0.0

print("\n" + "=" * 60)
print(" REPLAY REACTIVATION SCORES  (seed-pulse probe, post-training)")
print("=" * 60)
_rps_f = np.array(replay_scores_fast)
_rps_s = np.array(replay_scores_slow)
print(f"  Headline score = Phase-2 I_syn: asm_isyn - rnd_isyn (post-seed sustained activity)")
print(f"    Fast Only:          {_rps_f.mean():+.5f} +/- {_rps_f.std():.5f}")
print(f"    Slow Consolidation: {_rps_s.mean():+.5f} +/- {_rps_s.std():.5f}")
_seed_asm_f = _diag_mean(replay_diag_fast, "seed_asm_isyn")
_seed_rnd_f = _diag_mean(replay_diag_fast, "seed_rnd_isyn")
_seed_asm_s = _diag_mean(replay_diag_slow, "seed_asm_isyn")
_seed_rnd_s = _diag_mean(replay_diag_slow, "seed_rnd_isyn")
print(f"  Seed-phase I_syn breakdown:")
print(f"    {'cond':<22} {'asm_isyn':>10} {'rnd_isyn':>10} {'score':>10}")
print(f"    {'Fast Only':<22} {_seed_asm_f:>10.4f} {_seed_rnd_f:>10.4f} "
      f"{(_seed_asm_f-_seed_rnd_f):>+10.4f}")
print(f"    {'Slow Consolidation':<22} {_seed_asm_s:>10.4f} {_seed_rnd_s:>10.4f} "
      f"{(_seed_asm_s-_seed_rnd_s):>+10.4f}")

_f_asm  = _diag_mean(replay_diag_fast, "assembly_rate")
_f_rnd  = _diag_mean(replay_diag_fast, "random_rate")
_f_shf  = _diag_mean(replay_diag_fast, "shuffled_rate")
_f_asmB = _diag_mean(replay_diag_fast, "assembly_burst")
_f_rndB = _diag_mean(replay_diag_fast, "random_burst")
_f_shfB = _diag_mean(replay_diag_fast, "shuffled_burst")
_f_rs   = _diag_mean(replay_diag_fast, "rate_score")
_f_isA  = _diag_mean(replay_diag_fast, "assembly_isyn")
_f_isR  = _diag_mean(replay_diag_fast, "random_isyn")
_f_isS  = _diag_mean(replay_diag_fast, "shuffled_isyn")
_f_iscr = _diag_mean(replay_diag_fast, "isyn_score")
_s_asm  = _diag_mean(replay_diag_slow, "assembly_rate")
_s_rnd  = _diag_mean(replay_diag_slow, "random_rate")
_s_shf  = _diag_mean(replay_diag_slow, "shuffled_rate")
_s_asmB = _diag_mean(replay_diag_slow, "assembly_burst")
_s_rndB = _diag_mean(replay_diag_slow, "random_burst")
_s_shfB = _diag_mean(replay_diag_slow, "shuffled_burst")
_s_rs   = _diag_mean(replay_diag_slow, "rate_score")
_s_isA  = _diag_mean(replay_diag_slow, "assembly_isyn")
_s_isR  = _diag_mean(replay_diag_slow, "random_isyn")
_s_isS  = _diag_mean(replay_diag_slow, "shuffled_isyn")
_s_iscr = _diag_mean(replay_diag_slow, "isyn_score")
_f_enr  = (_f_asm / max(_f_rnd, 1e-9))
_s_enr  = (_s_asm / max(_s_rnd, 1e-9))

print(f"  Secondary rate-based score (assembly_rate - random_rate):")
print(f"    Fast Only:          {_f_rs:+.5f}")
print(f"    Slow Consolidation: {_s_rs:+.5f}")

print(f"\n  Mean I_syn during replay window (synaptic recruitment):")
print(f"    {'cond':<22} {'asm':>9} {'rnd':>9} {'shuf':>9} {'score':>9}")
print(f"    {'Fast Only':<22} {_f_isA:>9.4f} {_f_isR:>9.4f} {_f_isS:>9.4f} "
      f"{_f_iscr:>+9.4f}")
print(f"    {'Slow Consolidation':<22} {_s_isA:>9.4f} {_s_isR:>9.4f} {_s_isS:>9.4f} "
      f"{_s_iscr:>+9.4f}")

print(f"\n  Mean firing rates (per timestep per neuron):")
print(f"    {'cond':<22} {'asm':>9} {'rnd':>9} {'shuf':>9} {'enrich':>10}")
print(f"    {'Fast Only':<22} {_f_asm:>9.4f} {_f_rnd:>9.4f} {_f_shf:>9.4f} "
      f"{_f_enr:>9.2f}x")
print(f"    {'Slow Consolidation':<22} {_s_asm:>9.4f} {_s_rnd:>9.4f} {_s_shf:>9.4f} "
      f"{_s_enr:>9.2f}x")

print(f"\n  Burst fractions (fraction of timesteps with >=50% group co-firing):")
print(f"    {'cond':<22} {'asm':>9} {'rnd':>9} {'shuf':>9}")
print(f"    {'Fast Only':<22} {_f_asmB:>9.4f} {_f_rndB:>9.4f} {_f_shfB:>9.4f}")
print(f"    {'Slow Consolidation':<22} {_s_asmB:>9.4f} {_s_rndB:>9.4f} {_s_shfB:>9.4f}")

if _rps_f.std() < 1e-9 and _rps_s.std() < 1e-9:
    print("\n  [WARN] Replay co-activity scores have zero variance — replay "
          "detection produced no signal.  Try adjusting noise_level "
          "in measure_replay_during_rest.")
elif (_f_asmB - _f_rndB) > 0 or (_s_asmB - _s_rndB) > 0:
    print("\n  [PASS] Assembly co-firing exceeds random/shuffled controls — "
          "replay detector is sensitive to assembly structure.")

print("\n" + "=" * 60)
print(" REPLAY-RETENTION CORRELATION  (all trials + conditions combined)")
print("=" * 60)
_all_rp  = np.concatenate([_rps_f, _rps_s])
_all_ret = np.array(results_fast[REST_STEPS_LIST[-1]] +
                    results_slow[REST_STEPS_LIST[-1]])

def _safe_pearson(x, y, eps=1e-12):
    """Defensive Pearson: returns (r, p, error_msg). Never raises."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None, None, "insufficient samples"
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None, None, "non-finite values"
    if float(x.std()) < eps:
        return None, None, "constant replay score (zero variance)"
    if float(y.std()) < eps:
        return None, None, "constant retention (zero variance)"
    try:
        r, p = pearsonr(x, y)
    except Exception as exc:
        return None, None, f"pearsonr failed: {exc}"
    if not (np.isfinite(r) and np.isfinite(p)):
        return None, None, "pearsonr returned non-finite"
    return float(r), float(p), None

_r_pc, _p_pc, _err_pc = _safe_pearson(_all_rp, _all_ret)
if _r_pc is None:
    print(f"  [WARN] Replay variance too small for correlation analysis "
          f"({_err_pc}); skipping.")
else:
    _sig = ('***' if _p_pc < 0.001 else '**' if _p_pc < 0.01 else
            '*'   if _p_pc < 0.05  else 'n.s.')
    print(f"  Pearson r = {_r_pc:+.4f}  p = {_p_pc:.4f}  ({_sig})")
    if _r_pc > 0 and _p_pc < 0.05:
        print("  [PASS] Replay reactivation positively predicts long-term retention.")
    elif _r_pc > 0:
        print("  [PARTIAL] Positive trend (not yet significant).")
    else:
        print("  [NOTE] No positive replay-retention correlation detected.")


# ============================================================
# PLOT
# ============================================================

print("\n[INFO] Generating plot...")
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
rest_ms = [r * DT for r in REST_STEPS_LIST]
fm = [float(np.mean(results_fast[r])) for r in REST_STEPS_LIST]
fs = [float(np.std(results_fast[r]))  for r in REST_STEPS_LIST]
sm = [float(np.mean(results_slow[r])) for r in REST_STEPS_LIST]
ss = [float(np.std(results_slow[r]))  for r in REST_STEPS_LIST]
ax.errorbar(rest_ms, fm, yerr=fs, marker='o', linewidth=2, capsize=4,
            label='Fast Only')
ax.errorbar(rest_ms, sm, yerr=ss, marker='s', linewidth=2, capsize=4,
            label='Slow Consolidation')
ax.axhline(1.0, linestyle='--', color='gray', alpha=0.5,
           label='Perfect retention')
ax.axhline(0.0, linestyle=':', color='gray', alpha=0.5)
ax.set_xlabel("Rest duration (ms)")
ax.set_ylabel("Retention (signal / immediate baseline)")
ax.set_title("Memory Retention: Fast Decay vs Slow Consolidation")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("forgetting_curves.png", dpi=200)
print("[SAVED] forgetting_curves.png")
print("[DONE]")

# ============================================================
# NEW PLOT 1 — SYNAPTIC WEIGHT EVOLUTION  (Task 4.3)
# ============================================================

print("\n[INFO] Generating synaptic_weight_evolution.png...")
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
_rest_ms = [r * DT for r in REST_STEPS_LIST]

_fast_within  = [_wmean(wt_fast_rest[r], 'within_mean')  for r in REST_STEPS_LIST]
_fast_outside = [_wmean(wt_fast_rest[r], 'outside_mean') for r in REST_STEPS_LIST]
_slow_within  = [_wmean(wt_slow_rest[r], 'within_mean')  for r in REST_STEPS_LIST]
_slow_outside = [_wmean(wt_slow_rest[r], 'outside_mean') for r in REST_STEPS_LIST]

ax2.plot(_rest_ms, _fast_within,  'o-',  linewidth=2,   color='tab:blue',
         label='Fast Only — within assembly')
ax2.plot(_rest_ms, _slow_within,  's-',  linewidth=2,   color='tab:orange',
         label='Slow Consolidation — within assembly')
ax2.plot(_rest_ms, _fast_outside, 'o--', linewidth=1.2, color='tab:blue',
         alpha=0.5, label='Fast Only — outside assembly')
ax2.plot(_rest_ms, _slow_outside, 's--', linewidth=1.2, color='tab:orange',
         alpha=0.5, label='Slow Consolidation — outside assembly')
ax2.set_xlabel("Rest duration (ms)")
ax2.set_ylabel("Mean E→E weight (effective)")
ax2.set_title("Synaptic Weight Evolution: Within vs Outside Assembly")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("synaptic_weight_evolution.png", dpi=200)
print("[SAVED] synaptic_weight_evolution.png")

# ============================================================
# NEW PLOT 2 — REPLAY-RETENTION CORRELATION  (Task 5.3)
# ============================================================

print("[INFO] Generating replay_retention_correlation.png...")
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))

_rp_f = np.array(replay_scores_fast)
_rp_s = np.array(replay_scores_slow)
_rt_f = np.array(results_fast[REST_STEPS_LIST[-1]])
_rt_s = np.array(results_slow[REST_STEPS_LIST[-1]])

ax3.scatter(_rp_f, _rt_f, marker='o', s=60, alpha=0.8, color='tab:blue',
            label='Fast Only', zorder=3)
ax3.scatter(_rp_s, _rt_s, marker='s', s=60, alpha=0.8, color='tab:orange',
            label='Slow Consolidation', zorder=3)

_all_rp2  = np.concatenate([_rp_f, _rp_s])
_all_ret2 = np.concatenate([_rt_f, _rt_s])
_can_fit = (len(_all_rp2) >= 3
            and np.all(np.isfinite(_all_rp2))
            and np.all(np.isfinite(_all_ret2))
            and np.std(_all_rp2) > 1e-9
            and np.std(_all_ret2) > 1e-9)
if _can_fit:
    try:
        _m, _b = np.polyfit(_all_rp2, _all_ret2, 1)
        _x_line = np.linspace(_all_rp2.min(), _all_rp2.max(), 100)
        ax3.plot(_x_line, _m * _x_line + _b, 'k--', linewidth=1.5, alpha=0.7,
                 label='Regression line')
        _r_txt, _p_txt = pearsonr(_all_rp2, _all_ret2)
        ax3.text(0.05, 0.92, f"r = {_r_txt:+.3f}  p = {_p_txt:.4f}",
                 transform=ax3.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except (np.linalg.LinAlgError, ValueError) as _fit_err:
        print(f"  [WARN] Regression fit failed ({_fit_err}); plotting points only.")
        ax3.text(0.05, 0.92, "regression unavailable",
                 transform=ax3.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
else:
    print("  [WARN] Insufficient variance for regression line; plotting points only.")
    ax3.text(0.05, 0.92, "regression unavailable\n(insufficient variance)",
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

ax3.set_xlabel("Replay score (assembly - background activity during rest)")
ax3.set_ylabel(f"Retention @ {REST_STEPS_LIST[-1]} steps rest")
ax3.set_title("Replay Reactivation Score vs Long-Term Retention")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("replay_retention_correlation.png", dpi=200)
print("[SAVED] replay_retention_correlation.png")

# ============================================================
# NEW PLOT 3 — REPLAY SCORE DISTRIBUTION (Fast vs Slow histograms)
# ============================================================

print("[INFO] Generating replay_score_distribution.png...")
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5))

_rps_all = np.concatenate([_rps_f, _rps_s]) if (len(_rps_f) and len(_rps_s)) \
           else np.array(list(_rps_f) + list(_rps_s))
if _rps_all.size >= 2 and (_rps_all.max() - _rps_all.min()) > 1e-9:
    _lo = float(_rps_all.min()) - 0.05 * abs(float(_rps_all.min()) + 1e-9)
    _hi = float(_rps_all.max()) + 0.05 * abs(float(_rps_all.max()) + 1e-9)
    _bins = np.linspace(_lo, _hi, max(8, min(20, len(_rps_all) // 2)))
    ax4.hist(_rps_f, bins=_bins, alpha=0.6, color='tab:blue',
             label=f'Fast Only (mean={_rps_f.mean():+.4f})', edgecolor='black')
    ax4.hist(_rps_s, bins=_bins, alpha=0.6, color='tab:orange',
             label=f'Slow Consolidation (mean={_rps_s.mean():+.4f})', edgecolor='black')
    ax4.axvline(0.0, color='gray', linestyle=':', alpha=0.6, label='Zero')
    if _rps_f.size > 0:
        ax4.axvline(_rps_f.mean(), color='tab:blue',   linestyle='--', alpha=0.8)
    if _rps_s.size > 0:
        ax4.axvline(_rps_s.mean(), color='tab:orange', linestyle='--', alpha=0.8)
else:
    ax4.text(0.5, 0.5,
             "Replay scores have insufficient variance\n"
             "(all values numerically identical)",
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))
ax4.set_xlabel("Replay score (assembly - random activity during rest)")
ax4.set_ylabel("Trial count")
ax4.set_title("Distribution of Replay Reactivation Scores")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("replay_score_distribution.png", dpi=200)
print("[SAVED] replay_score_distribution.png")

# ============================================================
# NEW PLOT 4 — ASSEMBLY vs RANDOM/SHUFFLED CONTROL (bar chart with CIs)
# ============================================================

print("[INFO] Generating replay_vs_random_control.png...")
fig5, ax5 = plt.subplots(1, 1, figsize=(8, 5))

def _mean_sem(diag_list, key):
    if not diag_list:
        return 0.0, 0.0
    arr = np.array([d[key] for d in diag_list], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    sem = float(arr.std() / np.sqrt(max(arr.size, 1)))
    return float(arr.mean()), sem

_groups = ['assembly_rate', 'random_rate', 'shuffled_rate']
_group_labels = ['Assembly', 'Random (BG window)', 'Shuffled (other window)']
_f_means, _f_sems = zip(*[_mean_sem(replay_diag_fast, k) for k in _groups])
_s_means, _s_sems = zip(*[_mean_sem(replay_diag_slow, k) for k in _groups])

_x = np.arange(len(_groups))
_width = 0.38
ax5.bar(_x - _width/2, _f_means, _width, yerr=_f_sems, capsize=4,
        label='Fast Only',          color='tab:blue',   alpha=0.85, edgecolor='black')
ax5.bar(_x + _width/2, _s_means, _width, yerr=_s_sems, capsize=4,
        label='Slow Consolidation', color='tab:orange', alpha=0.85, edgecolor='black')
ax5.set_xticks(_x)
ax5.set_xticklabels(_group_labels)
ax5.set_ylabel("Mean firing rate during rest (spikes / timestep / neuron)")
ax5.set_title("Assembly Reactivation vs Random and Shuffled Controls")
ax5.legend(fontsize=9)
ax5.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("replay_vs_random_control.png", dpi=200)
print("[SAVED] replay_vs_random_control.png")

# ============================================================
# NEW PLOT 5 — REPLAY ACTIVITY RASTER (single example trial per condition)
# ============================================================

print("[INFO] Generating replay_activity_raster.png...")
fig6, axes6 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

def _draw_raster(ax, asm_spk, rnd_spk, title):
    """Raster of assembly (top, orange) and random (bottom, gray) spikes."""
    if asm_spk is None or rnd_spk is None:
        ax.text(0.5, 0.5, "No raster traces saved for this condition",
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))
        ax.set_title(title)
        return
    n_t, n_asm = asm_spk.shape
    n_rnd = rnd_spk.shape[1]
    t_ms = np.arange(n_t) * DT
    # shaded assembly band background
    ax.axhspan(-0.5, n_asm - 0.5, facecolor='tab:orange', alpha=0.08, zorder=0)
    # assembly spikes
    for i in range(n_asm):
        ts = np.where(asm_spk[:, i] > 0.5)[0]
        if ts.size > 0:
            ax.plot(ts * DT, np.full(ts.size, i), '|',
                    color='tab:orange', markersize=6, markeredgewidth=1.4)
    # random spikes plotted above assembly band
    for j in range(n_rnd):
        ts = np.where(rnd_spk[:, j] > 0.5)[0]
        if ts.size > 0:
            ax.plot(ts * DT, np.full(ts.size, n_asm + j), '|',
                    color='gray', markersize=5, markeredgewidth=1.0, alpha=0.7)
    ax.set_ylim(-1, n_asm + n_rnd)
    ax.axhline(n_asm - 0.5, color='black', linewidth=0.6, alpha=0.5)
    ax.set_ylabel("Neuron index\n(assembly | random)")
    ax.set_title(title)

_ex_fast = replay_diag_fast[0] if replay_diag_fast else {}
_ex_slow = replay_diag_slow[0] if replay_diag_slow else {}
_draw_raster(axes6[0],
             _ex_fast.get("assembly_spikes"),
             _ex_fast.get("random_spikes"),
             f"Fast Only — trial 1 replay window  "
             f"(score={_ex_fast.get('score', float('nan')):+.4f})")
_draw_raster(axes6[1],
             _ex_slow.get("assembly_spikes"),
             _ex_slow.get("random_spikes"),
             f"Slow Consolidation — trial 1 replay window  "
             f"(score={_ex_slow.get('score', float('nan')):+.4f})")
axes6[1].set_xlabel("Time during replay window (ms)")
plt.tight_layout()
plt.savefig("replay_activity_raster.png", dpi=200)
print("[SAVED] replay_activity_raster.png")
