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
N_TRIALS        = 5

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

for cond_name, use_slow, results in [
    ("Fast Only",          False, results_fast),
    ("Slow Consolidation", True,  results_slow),
]:
    print(f"\n  [{cond_name}]")
    for trial in range(N_TRIALS):
        torch.manual_seed(SEED + trial)
        np.random.seed(SEED + trial)

        net = build_network(use_slow=use_slow)
        train_pattern(net, PATTERN)
        W_train = net.W.data.clone()
        W_slow_train = net.W_slow.clone() if use_slow else None

        # Baseline (immediate, no rest)
        sp, isn = recall_run(net, CUE_NEURONS)
        sig_base, _, _, _ = recall_metrics(sp, isn)
        print(f"    trial {trial+1}: baseline signal = {sig_base:+.4f}")

        for rest_n in REST_STEPS_LIST:
            net.W.data.copy_(W_train)
            if use_slow:
                net.W_slow.copy_(W_slow_train)
            bulk_rest(net, rest_n)

            sp, isn = recall_run(net, CUE_NEURONS)
            sig, _, _, _ = recall_metrics(sp, isn)

            if sig_base > 1e-3:
                ret = sig / sig_base
            else:
                ret = 0.0
            results[rest_n].append(float(ret))


# ---- Report ----
print("\n" + "=" * 60)
print(" FORGETTING CURVE RESULTS  (retention = signal / baseline)")
print("=" * 60)
print(f"  {'Rest':<8} {'Fast Only':<22} {'Slow Consolidation':<22}")
for r in REST_STEPS_LIST:
    fm, fs = float(np.mean(results_fast[r])), float(np.std(results_fast[r]))
    sm, ss = float(np.mean(results_slow[r])), float(np.std(results_slow[r]))
    print(f"  {r:<8} {fm:+.3f} +/- {fs:.3f}     {sm:+.3f} +/- {ss:.3f}")

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
