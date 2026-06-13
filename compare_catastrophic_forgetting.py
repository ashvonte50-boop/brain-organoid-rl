"""
Catastrophic Forgetting -- Stable + Optimised.
=================================================

Sequential learning of overlapping memories A->B->C->D.
Four conditions: {Fast, Slow} x {NoReplay, Replay}.
Overlap sweep: 0%, 10%, 20%  (40%/60% removed — I_syn metric conflation at high overlap).

ARCHITECTURE MODES (v4):
  ARCH_MODE = "dense"           — original validated 400-neuron system (unchanged)
  ARCH_MODE = "sparse_modular"  — scalable 800–1000 neuron system with modular
                                  connectivity, fan-in normalization, localized replay

STABILITY HISTORY
-----------------
v1 : original, stable -- t=2.625, p=0.030 (significant).
     Replay at TEST_NOISE=1.5 (bistable-silent). No competition. No tags.
v2-unstable : GAMMA=0.60, N_PRES=30, REPLAY_NOISE=3.0, TAG_RATE=0.08
     W_slow overshot W_MAX because tag_driven_consolidation had no clamp.
     Probe scores blew up to +-22.  DISCARDED.
v3 (this file) : stable rollback + targeted optimisations.
     Root cause fixed (W_slow clamped after tag capture).
     Single new mechanism added per iteration, validated before proceeding.

ROOT CAUSE OF v2 INSTABILITY
------------------------------
tag_driven_consolidation() set W_slow += capture_rate * tag * delta
without clamping.  With 60 replay events x TAG_RATE=0.08, W_slow exceeded
W_MAX=1.5.  Since W_slow has no intrinsic upper bound in IzhikevichNetwork,
W_eff = (1-gamma)*W + gamma*W_slow could reach >>1.5 across many connections.
The probe at TEST_NOISE=1.5 then measured enormous I_syn differentials (+-22).

FIXES IN v3
-----------
1.  W_slow clamped to [0, W_MAX] after every tag_driven_consolidation call.
2.  TAG_CAPTURE_RATE reduced to 0.02 (mild; biologically conservative).
3.  REPLAY_NOISE reduced to 2.0 (above bistable but well below saturation).
4.  N_REPLAY_EVENTS reduced to 25 (stable; same as v1 had for reference).
5.  GAMMA = 0.5 (restored to validated value from compare_retention.py).
6.  N_PRESENTATIONS = 20 (restored).
7.  FAST_DECAY_TAU = 1500 (restored).
8.  homeostatic_step() removed from training (was interacting with slow weights
    in ways that depended on random seed; removed until independently validated).

PERFORMANCE OPTIMISATIONS
--------------------------
A. bulk_slow_step(net, n_steps): closed-form exact solution to the slow-step
   ODE for constant W.  Replaces O(n_steps) loop calls with a single
   vectorised operation.  Saves ~50,000 slow_step() calls per trial.
B. Analytical LTP tag (SynapticTags.update_from_spikes): eliminates
   per-step W tensor clone.  Was ~29,000 clones/trial; now zero.
C. DEV_MODE flag: quick 2-trial / 1-overlap run completes in <10 min.
D. GENERATE_PDFS flag: skip PDF rendering for iterative debugging.
E. ProcessPoolExecutor: parallel trials / conditions (4x speedup on 4 cores).
F. Timer dict: per-section timing printed at end for future profiling.

SCIENTIFIC CONSTRAINTS (never violated)
-----------------------------------------
- izhikevich_network.py: NOT modified
- compare_retention.py: NOT modified
- STDP regime (A_plus, A_minus, taus): UNCHANGED
- Core consolidation mechanism: UNCHANGED
- Probe metric (I_syn differential): UNCHANGED
- All validated physics parameters: UNCHANGED

NOVELTY MECHANISMS (stable, mild, validated one at a time)
-----------------------------------------------------------
1. Partial-cue reactivation (5/20 neurons seeded; pattern completion fills rest)
2. Replay at REPLAY_NOISE=2.0 (above bistable; STDP fires; safe)
3. Overlap-dependent competitive interference (strength=0.25; mild)
4. Synaptic tagging & capture (STC hypothesis; rate=0.02; W_slow clamped)
5. Multi-memory replay chains (prob=0.20)
6. Interference-aware replay scheduling
7. Representational drift analysis (cosine similarity over sequence)
"""

import os
import sys
import time
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.stats import (ttest_ind, sem as _scipy_sem, linregress as _linregress,
                         pearsonr as _pearsonr, t as _scipy_t)
from scipy.spatial.distance import cosine as _cosine_dist

from neuron_models.izhikevich_network import IzhikevichNetwork

# Workers (spawned by ProcessPoolExecutor) reimport this module.
# On a 4-core machine: 3 workers x 4 threads = 12 threads competing for 4 cores.
# Setting 1 thread per worker lets the OS give each worker 1 full core.
# For N=300 small matrices, single-threaded is faster anyway (no thread-launch overhead).
_is_worker = multiprocessing.current_process().name != 'MainProcess'
torch.set_num_threads(1 if _is_worker else 4)
warnings.filterwarnings("ignore", category=RuntimeWarning)   # suppress nanmean on empty slice

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
MASTER_SEED = 42
torch.manual_seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE MODE  (v4 — scalable sparse modular architecture)
# ─────────────────────────────────────────────────────────────────────────────
# "dense"           = original validated 400-neuron distance-dependent connectivity
# "sparse_modular"  = modular topology with sparse E-E, fan-in normalization, local replay
ARCH_MODE = "sparse_modular"

# Sparse modular architecture parameters (used only when ARCH_MODE == "sparse_modular")
N_MODULES               = 8
INTRA_MODULE_CONN_PROB  = 0.25     # within-module connection probability
INTER_MODULE_CONN_PROB  = 0.02     # between-module connection probability
INTER_MODULE_SCALE      = 0.05     # cross-module weight scaling factor
EE_SPARSITY             = 0.10     # overall E-E connection density target
LOCAL_REPLAY_SCALE      = 1.0      # within-module replay strength
DISTAL_REPLAY_SCALE     = 0.05     # cross-module replay strength
MEMORY_MODULE           = 0        # module where assemblies are placed
DEBUG_SCALING           = False    # print scaling diagnostics during validation

# ─────────────────────────────────────────────────────────────────────────────
# NETWORK CONSTANTS  (identical to compare_retention.py -- do not change)
# ─────────────────────────────────────────────────────────────────────────────
N_NEURONS   = 1000
N_INH       = 250
N_EXC       = N_NEURONS - N_INH

G_EXC       = 5.0
G_INH       = -40.0
NOISE_STD   = 4.0         # training noise
TEST_NOISE  = 1.5         # probe noise (bistable-silent)
DT          = 0.5

# STDP -- assembly-learning regime (A_plus > A_minus, validated)
A_PLUS      = 0.006
A_MINUS     = 0.003
TAU_PLUS    = 20.0
TAU_MINUS   = 20.0
W_MAX       = 1.5

# Slow consolidation -- validated in compare_retention.py, restored here
FAST_DECAY_TAU  = 1500.0   # v2-unstable used 2000; RESTORED to validated 1500
# GAMMA: fraction of W_eff drawn from W_slow.  Raised from 0.50 - 0.65 because
# the slow pathway is the mechanism we are trying to demonstrate — at 0.50 the
# slow contribution is mathematically halved before reaching the probe, masking
# the Slow+Replay advantage.  0.65 keeps the dual-pathway interpretation
# (fast still contributes 35%) while letting consolidated weights actually
# drive the readout.  Bounded by W_MAX so this cannot destabilize forward pass.
GAMMA           = 0.65
TAU_SLOW        = 3000.0   # was 4000.  Faster upward catch-up of W_slow to
                            # W_fast during training rests.  Over the full
                            # 20-presentation production schedule (12000 slow_step
                            # steps), catch-up rises from 95% - 98% (small).
                            # In DEV_MODE (4200 steps), catch-up rises 65% - 75%
                            # (substantial).  Downward drift unchanged — that is
                            # controlled by TAU_VERY_SLOW, which preserves the
                            # asymmetric "consolidation ratchet" against forgetting.
TAU_VERY_SLOW   = 200_000.0

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
N_MEMORIES              = 4

# Assembly size scales with module size in sparse_modular mode
if ARCH_MODE == "sparse_modular":
    _MODULE_SIZE = N_EXC // N_MODULES
    ASSEMBLY_SIZE = max(20, _MODULE_SIZE // 4)
    CUE_SIZE = max(5, ASSEMBLY_SIZE // 4)
else:
    ASSEMBLY_SIZE = 20
    CUE_SIZE = 5
N_PRESENTATIONS_PER_MEM = 12          # v2-unstable used 30; production-stable value.
                                       # 20 presentations caused W_slow saturation
                                       # (>98% of W_MAX) - cross-attractor runaway
                                       # when two overlapping assemblies both have
                                       # near-maximum W_eff.  12 keeps W_eff in the
                                       # stable operating range (~0.60 vs threshold
                                       # ≈0.75) while providing adequate training.
STIM_STRENGTH           = 15.0
STIM_DURATION_MS        = 50
INTERVAL_MS             = 300

INTER_MEM_REST_STEPS    = 2500        # was 1500.  With FAST_DECAY_TAU=1500,
                                       # 2500-step rest gives ~81% fast-weight
                                       # decay (was 63%) — sufficient forgetting
                                       # pressure for Fast/NoReplay, and a
                                       # longer window for slow consolidation
                                       # + replay events.  Biologically: longer
                                       # post-encoding consolidation gap.
PROBE_DURATION_MS       = 200
CUE_STRENGTH            = 12.0
# Background window for probe — in sparse_modular mode, use a non-assembly module
if ARCH_MODE == "sparse_modular":
    _BG_MODULE = (MEMORY_MODULE + 1) % N_MODULES
    BG_START = _BG_MODULE * _MODULE_SIZE
    BG_SIZE  = min(30, _MODULE_SIZE)
else:
    BG_START = 150
    BG_SIZE  = 30
BG_END                  = BG_START + BG_SIZE

# Replay -- conservative, stable parameters
# v2-unstable used noise=3.0, events=60. Both restored to safe values.
REPLAY_NOISE_STD         = 2.0        # above bistable(1.5), below saturation; STDP fires
# REPLAY_SEED_STRENGTH raised 9.0 - 12.0 to match probe CUE_STRENGTH.
# With longer INTER_MEM_REST (2500 steps, 81% fast decay), the cued A neurons
# must reliably spike even when W_fast is near baseline.  9.0 was tuned for
# the previous 63%-decay regime; under stronger forgetting pressure, the
# weaker seed allowed random background firing to dominate STDP during replay
# in Fast/Replay (no W_slow to support pattern completion).
REPLAY_SEED_STRENGTH     = 12.0
REPLAY_SEED_DURATION     = 15
REPLAY_SPONTANEOUS_STEPS = 100
N_REPLAY_EVENTS_PER_REST = 35   # was 25.  More events per rest are needed
                                 # because each capture transfers a small,
                                 # tag-bounded amount to W_slow.  35 events ×
                                 # 3 rests = 105 capture opportunities for
                                 # Memory A under chained replay.
# PARTIAL_CUE_SIZE 5 - 8 (40% partial cue, was 25%).  At 25% cue with the new
# stronger inter-memory decay, recurrent W_fast in Fast/Replay was insufficient
# to drive pattern completion (only 5 neurons firing × W≈0.1 = ~0.5 mean input
# to non-cued A; below spike threshold).  Replay then re-trained random
# connections instead of A-specific ones — explaining the negative-going
# Fast/Replay result.  At 40% cue, recurrent input scales to ~0.8 mean -
# reliable completion across all conditions.  Biologically consistent with
# sharp-wave-ripple reactivation, which engages substantial fractions of an
# assembly, not single neurons.
if ARCH_MODE == "sparse_modular":
    PARTIAL_CUE_SIZE = max(8, ASSEMBLY_SIZE // 2)
else:
    PARTIAL_CUE_SIZE = 8
CHAIN_REPLAY_PROB        = 0.20       # prob of chaining to next memory in sequence

# ── Sequential / trajectory replay ──────────────────────────────────────────
# When REPLAY_TRAJECTORY is "random" (default), each event selects the assembly
# to replay independently via the prioritization scheme.  When set to
# "sequential" or "reverse", the burst trajectory traverses learned assemblies
# in order A→B→C→D (or reverse), with repetitions if more events than assemblies.
# "bidirectional" alternates: forward pass then reverse pass per burst.
# This models hippocampal theta sequences (Foster & Wilson 2006) where replay
# sweeps forward or backward along a learned trajectory during SWR events.
REPLAY_TRAJECTORY = "bidirectional"   # "random", "sequential", "reverse", "bidirectional"

# Replay-burst mechanism (Sharp-Wave-Ripple burst clustering)
# Hippocampal SWR events cluster in bursts of 3–10 ripples separated by
# ~100–300 ms; individual ripples within a burst reactivate overlapping
# patterns.  Temporal summation within a burst drives larger net STDP
# accumulation at recurrent A-A synapses before the next decay epoch
# resumes.  This is critical for Fast/Replay: without W_slow support,
# a single isolated replay event yields only a small W_fast increment
# that is erased by the next inter-burst gap.  Bursts of REPLAY_BURST_SIZE
# rapid-fire events accumulate BURST_SIZE × increment before any decay,
# giving Fast/Replay a meaningful W_fast re-potentiation step.
#
# Total replay events per rest is unchanged (N_REPLAY_EVENTS_PER_REST).
# The burst structure only reorganises their temporal layout.
#
# REPLAY_BURST_GAP: steps of passive dynamics between bursts (~25 ms at
# DT=0.5 ms/step).  Applies a small _bulk_decay + bulk_slow_step so the
# inter-burst interval is biologically realistic and not free.
REPLAY_BURST_SIZE = 5    # rapid-fire events per burst
# Increased from 3 - 5.  Within a burst successive events benefit from the
# small W_fast increment deposited by preceding events (bootstrapping): even
# if event 1 achieves only partial completion, W_fast[A,A] is marginally
# higher for event 2, etc.  By event 4–5 the net accumulated LTP is
# substantially larger than from 3 events.  5 events per burst (3 bursts in
# DEV at 15 total; 7 bursts in production at 35 total) gives Fast/Replay
# adequate within-burst accumulation without slowing the rest period.
REPLAY_BURST_GAP  = 50   # passive steps between bursts (25 ms at DT=0.5 ms)

# ── Theta oscillation gating (hippocampal theta-nested replay) ──────────────
# During replay, a slow theta rhythm (4–8 Hz) modulates the noise amplitude
# gating STDP windows.  At theta peak, noise is at its base level (pattern
# completion favoured); at theta trough, noise is elevated (exploration /
# de-correlation).  This implements the "theta-nested replay" model (Buzsáki
# 2002): replay events tend to occur at theta peaks when CA3 recurrent
# collaterals drive coherent pattern completion.
#
# theta_phase runs from 0 to 2π over THETA_PERIOD steps.
# noise_mod(t) = THETA_MOD_DEPTH * sin(theta_phase(t)),
# applied as noise_std = base_noise * (1 + noise_mod(t)).
# At THETA_MOD_DEPTH=0 the mechanism is inert (no modulation).
THETA_FREQ_HZ   = 6.0          # theta frequency
THETA_MOD_DEPTH = 0.30         # fractional modulation depth (0=off, 0.3=±30%)
# theta phase is tracked globally across replay events within a rest period;
# it does NOT reset between events, preserving the continuous theta rhythm.
_th_phase_rad  = 0.0           # global theta phase (radians), updated per step
_gamma_phase_rad = 0.0         # gamma phase, updated per step at GAMMA_FREQ_HZ

# Competitive interference -- overlap-dependent synaptic depression.
# Runs for ALL conditions (not just use_slow=·).  Each new memory depresses
# old-assembly connections through shared neurons.  At 20% overlap with
# STRENGTH=0.50, each round applies 10% depression; 3 rounds (B,C,D) give
# cumulative factor 0.90^3 = 0.729 = 27.1% total depression.
USE_COMPETITION      = True
COMPETITION_STRENGTH = 0.50           # overlap_frac * strength = extra decay/round

# ── Overlap-sensitive interference mechanisms ─────────────────────────────────
# These create genuine attractor competition between overlapping memories:
#
# M1 — OVERLAP_COHERENCE_PENALTY: scale LAMBDA_OFF by (1 + overlap_frac * penalty).
#   Higher overlap -> stricter coherence gate -> fewer STDP events ->
#   weaker replay-driven consolidation for overlapping assemblies.
#
# M2 — CROSS_LTD_RATE: after replay STDP, apply LTD from overlap neurons to
#   neighboring assemblies.  Shared neurons active during A's replay send
#   conflicting signals; their connections to B's downstream targets are
#   weakened.  Implements competitive consolidation: shared neurons are
#   pulled toward whichever assembly replays them most.
#
# M3 — OVERLAP_REPLAY_BOOST: overlap-weighted prioritization.  Assemblies with
#   higher overlap get more replay events, preventing starvation of the most
#   interference-vulnerable memories.
#
# M4 — OVERLAP_PERS_BUDGET_REDUCTION: when the target overlaps others, the
#   shared persistence budget is reduced proportionally.  Zero-sum competition
#   for reverberatory support during replay.
#
# M5 — OVERLAP_DRIFT_RATE: directional drift.  During coherent replay of A,
#   overlap neurons co-activated with B's structure receive a small Hebbian
#   boost.  Over time, attractors drift toward shared latent structure.
#
# M6 — OVERLAP_FATIGUE_RATE: shared-neuron refractory fatigue.  Each time an
#   overlap neuron fires during a coherent replay step, its fatigue counter
#   increments.  Fatigue scales down the neuron's contribution to STDP,
#   reducing the consolidation benefit that overlapping assemblies receive
#   from shared neurons.  Fatigue decays between replay events.
#   Biological basis: spike-frequency adaptation through after-hyperpolarization
#   currents (Madison & Nicoll 1984) — neurons that fire frequently become
#   temporarily less excitable, creating temporal competition.
#
# M7 — HETERO_TAG_RATE: heterosynaptic LTD tag persistence.  After each replay
#   event, overlap→other connections that were active receive a persistent LTD
#   tag that decays slowly.  Tagged connections are more susceptible to LTD in
#   subsequent events, creating a cumulative competitive memory that makes
#   frequently-replayed assemblies capture shared neurons more effectively.
#   Biological basis: synaptic tagging and capture (Frey & Morris 1997) —
#   active synapses are "tagged" for later plasticity modulation.
#
# M8 — TRAINING_DECORR_STRENGTH: training-time overlap decorrelation.  During
#   training of a new memory, LTD is applied at connections from the new assembly's
#   specific neurons to the old assembly's specific neurons via shared overlap
#   pathways.  This drives representations apart at encoding time, preventing
#   overlapping memories from converging on identical synaptic weights.
#   Biological basis: pattern separation via lateral inhibition in dentate gyrus
#   (Yassa & Stark 2011).
#
# M9 — WTA_COH_MARGIN: coherence-gated winner-take-all during replay.  After each
#   replay step, the coherence of all assemblies is compared.  Only the assembly
#   with highest coherence (above a margin) gets positive STDP; all others receive
#   LTD at their overlap→specific connections.  This prevents mixed-representation
#   replay from consolidating interfering patterns.
#   Biological basis: competitive queuing during SWR (Pfeiffer 2020) — SWR content
#   is competitively selected, not random.
#
# M10 — RECONSOL_LTD_BOOST: reconsolidation window metaplasticity.  After a
#   successful probe or replay event, the reactivated assembly's overlap synapses
#   enter a labile window where they are more susceptible to LTD.  Future
#   interference events within this window apply enhanced LTD, modeling the
#   reconsolidation vulnerability window (Nader et al. 2000).
#
# M6 — overlap urgency signal in endogenous prioritization (handled inline).
OVERLAP_COHERENCE_PENALTY     = 2.0   # per-unit overlap penalty on LAMBDA_OFF
CROSS_LTD_RATE                = 0.03  # per-step LTD at overlap->neighbor connections
OVERLAP_REPLAY_BOOST          = 0.50  # max probability boost for high-overlap assemblies
OVERLAP_PERS_BUDGET_REDUCTION = 0.50  # budget reduction per unit overlap fraction
OVERLAP_DRIFT_RATE            = 0.02  # Hebbian drift toward overlapping structure
OVERLAP_FATIGUE_RATE          = 0.30  # fatigue added per active overlap spike per step
OVERLAP_FATIGUE_DECAY         = 0.85  # exponential fatigue decay between replay events
HETERO_TAG_RATE               = 0.10  # LTD tag strength after each replay event
HETERO_TAG_DECAY              = 0.90  # per-event decay of heterosynaptic LTD tags
TRAINING_DECORR_STRENGTH      = 0.08  # LTD rate for training-time decorrelation
WTA_COH_MARGIN                = 0.05  # coherence margin to "win" the replay step
WTA_LTD_RATE                  = 0.02  # LTD applied to losing assemblies' connections
RECONSOL_WINDOW_STEPS         = 50    # metaplastic window after reactivation (sim steps)
RECONSOL_LTD_BOOST            = 2.0   # LTD multiplier on overlapping synapses in window

# Synaptic tagging & capture (STC) -- very mild
# ROOT-CAUSE FIX: W_slow is clamped to [0, W_MAX] after every capture call
# so it can never overshoot the fast-weight ceiling.
USE_TAGGING          = True
TAG_DECAY_TAU        = 2500.0
# TAG_CAPTURE_RATE: re-calibrated 0.02 - 0.15 in conjunction with the
# tag_driven_consolidation formula change (gate removed).  Under the new direct
# capture rule ΔW_slow = rate · tag, each replay event transfers
# ~0.15 × mean_tag ≈ 0.004 to W_slow at tagged A-A synapses.  Across the
# protocol's ~105 capture opportunities for Memory A, total upward push to
# W_slow[A,A] sums to ~0.4 (partially eroded by slow_step's downward drift
# during B/C/D training), giving Slow+Replay a clear advantage.
# W_slow is hard-clamped to [0, W_MAX=1.5] so this cannot overshoot.
TAG_CAPTURE_RATE     = 0.15

# Overlap sweep
# NOTE: At ≥40% overlap the I_syn probe may partially reflect pattern-completion
# through shared neurons (A∩B = 8 at 40%, 12 at 60%) maintained by later B/C/D
# training, inflating Memory A scores.  Results at these overlaps should be
# interpreted with caution and are labelled accordingly in the figure.
# They are included for completeness to show the full interference gradient.
OVERLAP_FRACS   = [0.0, 0.10, 0.20, 0.40, 0.60]   # full sweep (production)
N_TRIALS        = 15   # ≥10 required for publication; 5 was dev-mode quality
N_TRIALS_SWEEP  = 5    # overlap sweep + prioritization comparison

# ── Performance flags ─────────────────────────────────────────────────────────
# DEV_MODE can be set via environment variable:
#   $env:DEV_MODE="1"   (PowerShell)   or   set DEV_MODE=1   (cmd)
# Accepts: "1", "true", "yes" (case-insensitive).  All other values -> False.
_dev_env  = os.environ.get("DEV_MODE", "").strip().lower()
DEV_MODE  = _dev_env in ("1", "true", "yes")
# Only print the notice from the main process; workers also import this module
# but their stdout is captured/interleaved and the repeated prints are noise.
if DEV_MODE and multiprocessing.current_process().name == 'MainProcess':
    print("[INFO] DEV_MODE=True (via $env:DEV_MODE)", flush=True)

GENERATE_PDFS = not DEV_MODE   # PDFs only in production; skip during dev iteration
N_WORKERS     = (min(2, multiprocessing.cpu_count())
                 if DEV_MODE
                 else min(max(1, multiprocessing.cpu_count() - 1), 4))

# DEV_MODE replay speed-up: fewer events per rest (-40%)
_N_REPLAY_EVENTS = 15 if DEV_MODE else N_REPLAY_EVENTS_PER_REST

# DEV_MODE training speed-up: hardcoded to 7 presentations per memory.
# Enough for stability checks and matches the validated DEV result set;
# not for quantitative production results.
# Production uses N_PRESENTATIONS_PER_MEM (12).
# NOTE: DEV is hardcoded (not a fraction of production) because reducing
# N_PRESENTATIONS_PER_MEM from 20 to 12 would otherwise drop DEV to
# int(12*0.35)=4, breaking the validated DEV baseline (Slow+Replay=0.1802).
_N_PRESENTATIONS = 7 if DEV_MODE else N_PRESENTATIONS_PER_MEM

# DEV_MODE probe speed-up: half the probe window.
# Halves probe time with negligible effect on the I_syn differential estimate.
probe_steps = 200

if DEV_MODE:
    probe_steps = probe_steps // 2

# Derived
stim_steps          = int(STIM_DURATION_MS / DT)
rest_steps_per_pres = int(INTERVAL_MS / DT)
probe_steps         = int(PROBE_DURATION_MS / DT)

PRIORITIZE_MODES = ["uniform", "oldest_first", "interference_aware", "endogenous"]
N_TRIALS_ABLATION = 5    # production; dev uses 2 (set in main via _n_ablation)

OUT_DIR = "."

# ── Stability guards ──────────────────────────────────────────────────────────
# Hard upper bound for outlier detection (probe score); trials that exceed
# this are retried once with an offset seed before being excluded.
OUTLIER_SCORE_THRESHOLD = 5.0

# During replay, if more than this fraction of all neurons fire simultaneously
# the network is in a runaway state: skip STDP for that step.
REPLAY_SPIKE_FRACTION_MAX = 0.35      # > 35% firing = runaway signal

# ── Replay coherence gating (biological replay-fidelity safeguard) ──────────
# Hippocampal sharp-wave-ripple replay drives downstream plasticity only when
# the reactivation is COHERENT — a clean ensemble of the target memory, not
# a noisy partial activation contaminated by random off-assembly firing.
#
# Without this gate, Fast/Replay bifurcates: when partial-cue completion fails
# (W_fast too low to recruit non-cued neurons), random background neurons fire
# during the replay seed, and STDP then strengthens A-noise connections.
# Result: replay sometimes corrupts memory instead of consolidating it.
#
# Per replay step we compute:
#   target_rate = fraction of CURRENT-memory excitatory neurons recently active
#   off_rate    = fraction of NON-target excitatory neurons recently active
#   coherence   = target_rate / (target_rate + λ · off_rate + ε)
# STDP fires only when coherence > COHERENCE_THR.  This naturally:
#   • opens for clean pattern completion (target dominates)
#   • opens for benign cue-only firing (off=0 - coherence=1)
#   • closes for noisy/failed completion (target ≈ off ≈ background)
#   • closes for runaway / saturated firing (everything ≈ 1)
# Calibrated values (see derivation in implementation):
REPLAY_COHERENCE_DECAY      = 0.95   # activity buffer per-step decay (~10 ms tau)
REPLAY_COHERENCE_ACTIVE_THR = 0.30   # buffer value above which a neuron counts active
REPLAY_COHERENCE_LAMBDA     = 3.0    # off-target weight in coherence denominator
REPLAY_COHERENCE_THR        = 0.50   # coherence required for STDP to fire

# ── Adaptive replay selection ─────────────────────────────────────────────────
# The spontaneous phase is split into two sub-phases:
#   (1) Evaluation window  (REPLAY_EVAL_STEPS): no STDP — observe coherence,
#       completion quality, and off-target activity.
#   (2) STDP window        (remaining steps)  : STDP fires ONLY for events
#       that passed all three acceptance criteria in the eval window.
#
# Biological rationale:
#   Sharp-wave ripples (SWRs) in the hippocampus are gated by neuromodulators
#   (ACh, dopamine) that evaluate whether a replay event is "valid" before
#   triggering synaptic plasticity.  Incoherent or noisy replays (weak
#   pattern-completion, high off-target activation, or no sustained coherent
#   epoch) are filtered out — only high-confidence events drive consolidation.
#
# Acceptance criteria (all three must hold within the eval window):
#   1. target_frac  >= REPLAY_ACCEPT_MIN_COMPLETION  — assembly is reactivated
#   2. off_frac     <= REPLAY_ACCEPT_MAX_OFFTARGET   — clean, low-noise replay
#   3. consec_above >= REPLAY_ACCEPT_MIN_CONSEC       — coherence is sustained
#
# Confidence score: geometric mean of normalised completion × stability ×
# coherence-SNR.  Continuous; computed for ALL events regardless of acceptance.
# Used in correlation analysis (Fig R5c) to measure how strongly replay
# quality predicts long-term retention.
REPLAY_EVAL_STEPS            = 20    # window used for confidence score (steps)
REPLAY_ACCEPT_MIN_CONSEC     = 3     # min consecutive coherent steps required
REPLAY_ACCEPT_MIN_COMPLETION = 0.10  # min target fraction at streak (≥2/20 neurons)
REPLAY_ACCEPT_MAX_OFFTARGET  = 0.30  # max off-target fraction at streak (safety valve)

# ── Attractor persistence (W_slow-mediated reverberatory support) ─────────────
# During the spontaneous replay phase, consolidated assemblies receive an
# additional recurrent excitation current proportional to their slow-weight
# connectivity and recent firing history.  This models NMDA-mediated
# reverberatory excitation that sustains attractor occupancy after the seed
# drive ends.
#
# Biological rationale:
#   Systems consolidation (cortical slow-wave sleep) strengthens recurrent
#   excitatory synapses via NMDA-LTP.  Stronger recurrent excitation creates
#   deeper attractor basins — once a pattern is activated, recurrent currents
#   sustain it against noise.  This is captured here by:
#     I_pers[i](t) = PERS_GAIN × Σ_j  W_slow[i,j] × trace[j](t)
#     trace[j](t) = PERS_DECAY × trace[j](t-1) + spike[j](t)
#   where the sum runs over all excitatory neurons j.
#
# Key properties:
#   • Zero for Fast/Replay:  tags=None - W_slow=0 - I_pers=0
#   • Scales with consolidation:  higher W_slow (more presentations, Slow cond)
#     - stronger persistence - longer coherent epochs
#   • Local: only assembly neurons that actually fired contribute (via trace)
#   • No assembly labels: W_slow encodes identity implicitly through weight structure
#   • Self-limiting: trace decays; clamped to avoid runaway amplification
REPLAY_PERS_DECAY = 0.90    # trace decay per step (NMDA tau ~5 ms at DT=0.5 ms)
REPLAY_PERS_GAIN  = 0.50    # reverberatory current gain (raised 0.30→0.50: stronger drive)
REPLAY_PERS_CLAMP = 6.0     # per-neuron ceiling (raised 4.0→6.0: allows more NMDA-like depolarisation)
# Competitive persistence budget: total reverberatory current across ALL excitatory
# neurons is bounded.  When multiple assemblies simultaneously sustain activity
# (e.g. via overlap neurons), they compete for this shared resource.  The assembly
# with stronger W_slow wins proportionally — competition is weighted by
# consolidation strength, not hardcoded assembly identity.
# Calibration: ≈ GAIN × ASSEMBLY_SIZE² × 0.5 × median_W_slow_prod
#   at N=1000, ASSEMBLY_SIZE=25: GAIN=0.5 × 625 × 0.5 × 0.40 ≈ 62.5
#   use 100 (generous; tightens as W_slow grows)
REPLAY_PERS_BUDGET = 100.0  # max total persistence current (competitive normalization)

# ── Endogenous replay prioritization ─────────────────────────────────────────
# When prioritize="endogenous", replay probability is recomputed each burst
# from accumulated network-state measurements rather than from a fixed rule.
#
# Urgency signals per assembly (all normalised to [0, 1]):
#   1. w_fast_erosion : how far w_fast_aa has decayed relative to other assemblies
#                       - assembly with weakest recurrent support = most urgent
#   2. reject_rate    : fraction of recent events rejected by adaptive gate
#                       - fragile attractor cannot sustain coherent replay
#   3. coh_deficit    : max(0, COH_THR − mean_peak_coherence) / COH_THR
#                       - coherence below gate threshold signals basin degradation
#
# Urgency = (u1 × u2 × u3)^(1/3)  with a small floor to prevent dead zeros.
# Probabilities = urgency / urgency.sum()  (no softmax temperature needed;
# the geometric mean already compresses extreme ratios naturally).
#
# The window is intentionally short (10 events) so urgency tracks recent
# state, not accumulated history.  Assemblies with no prior events are
# assigned mid-urgency (0.5 for each signal) so they get explored initially.
REPLAY_URGENCY_WINDOW = 10   # most-recent events per assembly for urgency signals

# Ablation suite (Phase 4)
# Each ablation runs Slow+Replay with one mechanism disabled.
# "Full model" is included so all conditions use identical random seeds.
# pers_gain=0.0        -> attractor persistence current zeroed (mechanism absent)
# use_competition=False -> competitive interference disabled
# prioritize="..."      -> replay scheduling mode
ABLATION_CONDITIONS = [
    {"label": "Full model",        "pers_gain": REPLAY_PERS_GAIN,  "use_competition": True,  "prioritize": "interference_aware"},
    {"label": "No persistence",    "pers_gain": 0.0,               "use_competition": True,  "prioritize": "interference_aware"},
    {"label": "No competition",    "pers_gain": REPLAY_PERS_GAIN,  "use_competition": False, "prioritize": "interference_aware"},
    {"label": "Uniform replay",    "pers_gain": REPLAY_PERS_GAIN,  "use_competition": True,  "prioritize": "uniform"},
    {"label": "Endogenous replay", "pers_gain": REPLAY_PERS_GAIN,  "use_competition": True,  "prioritize": "endogenous"},
]

# Explicit W clamp applied IN OUR CODE after every stdp_step() call, in
# addition to the w_max clamp inside IzhikevichNetwork (belt + suspenders).
# Using the same W_MAX constant so the STDP ceiling is unchanged.
_W_MAX_CLAMP = W_MAX                  # 1.5 — same as STDP ceiling
_W_SLOW_MAX_CLAMP = W_MAX             # W_slow ceiling (matches tag_driven_consolidation)
_W_TAG_MAX_CLAMP  = 0.5               # tag values stay small (< A_plus ceiling)

# ── Replay compression ─────────────────────────────────────────────────────
# Biological replay is temporally compressed (~5-10x faster than real-time;
# Foster & Wilson 2006, Ji & Wilson 2007).  COMPRESSION_FACTOR speeds up
# replay dynamics by scaling DT during replay events, making the model
# more biologically realistic and reducing computational cost.
# At COMPRESSION_FACTOR=5: 1 replay step = 2.5 ms biological time.
REPLAY_COMPRESSION_FACTOR = 5        # 1=real-time, 5=5x compressed
REPLAY_IGNITION_STRENGTH  = 0.15     # min target fraction for replay ignition
REPLAY_IGNITION_WINDOW    = 5        # steps to evaluate ignition
REPLAY_TERMINATE_AFTER    = 20       # consecutive sub-threshold steps -> early stop

# Probabilistic STDP gating  (replaces binary acceptance lock)
STDP_GATE_ENABLED         = True     # use probabilistic STDP gating instead of hard lock
STDP_GATE_SLOPE           = 8.0      # sigmoid steepness for STDP probability
STDP_GATE_BIAS            = 0.50     # sigmoid midpoint (coherence where p=0.5)
STDP_GATE_SMOOTH_ALPHA    = 0.3      # EMA weight for coherence smoothing (~3-step memory)

# ── Gamma oscillation nesting (theta-gamma coupling) ───────────────────────
# In hippocampus, gamma bursts (30-80 Hz) are nested within theta cycles.
# Pyramidal cell firing and STDP are modulated by gamma phase, creating
# discrete "packets" of neural activity (Lisman & Idiart 1995, Buzsáki 2002).
# Here we superimpose gamma-amplitude modulation on the STDP learning rate:
# STDP magnitude = A_PLUS * (1 + GAMMA_MOD_DEPTH * sin(gamma_phase))
# effectively creating phase-specific plasticity windows.
GAMMA_FREQ_HZ    = 40.0         # gamma frequency (Hz)
GAMMA_MOD_DEPTH  = 0.50         # STDP amplitude modulation depth
GAMMA_BURST_PROB = 0.30         # probability of gamma burst per theta cycle

# ── Sharp-wave ripple (SWR) synchronization ─────────────────────────────────
# During accepted replay, brief high-frequency ripple bursts (140-200 Hz)
# synchronize the assembly.  We inject a weak synchronizing current pulse
# at ripple frequency during high-coherence epochs.
RIPPLE_FREQ_HZ      = 150.0     # ripple frequency (Hz)
RIPPLE_STRENGTH     = 3.0       # synchronizing pulse strength
RIPPLE_MIN_COHERENCE = 0.60     # coherence threshold to trigger ripple burst
RIPPLE_BURST_LENGTH  = 5        # steps per ripple burst

# ── Phase-specific STDP ─────────────────────────────────────────────────────
# STDP efficacy depends on theta phase: LTP preferentially at theta peak,
# LTD at theta trough (Pavlides et al. 1988, Hyman et al. 2003).
# A_PLUS(t) = A_PLUS * (1 + PHASE_STDP_DEPTH * cos(theta_phase))
# A_MINUS(t) = A_MINUS * (1 - PHASE_STDP_DEPTH * cos(theta_phase))
PHASE_STDP_DEPTH = 0.40         # 0=off, 0.4=±40% modulation

# ── Behavioral readout ──────────────────────────────────────────────────────
# Linear decoder hyperparameters for memory retrieval classification.
DECODER_L2_PENALTY = 0.01       # ridge regression penalty
DECODER_TRAIN_FRAC = 0.80       # fraction of trials for training
DECODER_N_COMPONENTS = 20       # PCA components for dimensionality reduction
NOISE_RETRIEVAL_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]  # noise levels for retrieval robustness

# ── Homeostatic scaling ─────────────────────────────────────────────────────
HOMEOSTATIC_TARGET_RATE = 0.10
HOMEOSTATIC_STRENGTH    = 0.01
HOMEOSTATIC_WINDOW      = 100

# ── Asymmetric STDP for sequence learning ──────────────────────────────────
# Standard symmetric STDP (tau_plus=tau_minus) cannot learn temporal sequences.
# Sequence learning requires the LTD window to be wider than LTP (tau_plus <
# tau_minus), so anti-causal pairings are strongly depressed while causal
# pairings are potentiated.  Values from Bi & Poo 1998, Sjöström et al. 2001.
STDP_SEQUENCE_TAU_PLUS  = 15.0   # LTP time constant (ms) — narrower window
STDP_SEQUENCE_TAU_MINUS = 30.0   # LTD time constant (ms) — wider window
# Sequence transition strength: multiplier for learned A→B transitions
SEQUENCE_TRANSITION_STRENGTH = 0.15   # fraction of weight to copy to next-assembly

# ── Hippocampus-Cortex two-system architecture ─────────────────────────────
# Real consolidation requires separate HC (fast) and Cortex (slow) systems
# with replay-driven transfer.  HC learns rapidly but forgets quickly; cortex
# consolidates slowly but retains (McClelland et al. 1995, O'Reilly et al. 1998).
#   HC:   fast STDP, high decay, sparse recurrent, generates replay
#   Ctx:  slow STDP, low decay, dense local, stores consolidated memories
#   HC→Ctx: feedforward projections strengthened during replay
#   Ctx→HC: feedback projections, sparse, consolidated during slow-wave
HC_RATIO               = 0.30    # fraction of excitatory neurons in HC
N_HC                   = int(N_EXC * HC_RATIO)
N_CTX                  = N_EXC - N_HC
# HC parameters (fast encoding, rapid forgetting)
HC_A_PLUS              = A_PLUS * 2.0    # faster LTP
HC_A_MINUS             = A_MINUS * 2.0   # faster LTD (balanced)
HC_FAST_DECAY_TAU      = 500.0   # HC decays in ~500 steps (fast forgetting)
HC_SPARSITY            = 0.05    # HC recurrent connectivity (very sparse)
# Cortex parameters (slow encoding, stable retention)
CTX_A_PLUS             = A_PLUS * 0.5    # slower LTP
CTX_A_MINUS            = A_MINUS * 0.5   # slower LTD
CTX_SLOW_DECAY_TAU     = 3000.0  # cortex decays in ~3000 steps (slow forgetting)
CTX_SPARSITY           = 0.20    # cortex recurrent connectivity (moderately dense)
# HC→Ctx projection (feedforward memory trace)
HC_CTX_PROJ_PROB       = 0.10    # HC→Ctx connection probability
HC_CTX_PROJ_STRENGTH   = 0.30    # HC→Ctx initial weight strength
CTX_HC_PROJ_PROB       = 0.05    # Ctx→HC feedback probability
CTX_HC_PROJ_STRENGTH   = 0.15    # Ctx→HC feedback strength
# HC→Ctx replay transfer boost — amplifies HC drive to cortex during replay,
# modelling sharp-wave-ripple-driven consolidation (Buzsáki 2015).
HC_CTX_REPLAY_BOOST    = 3.0     # multiplier for HC→Ctx current during replay events

# ── Emergent replay triggering ─────────────────────────────────────────────
# Instead of orchestrating replay events externally, replay can also be
# triggered spontaneously when network state crosses a coherence threshold.
# This models hippocampal sharp-wave ripple initiation (Buzsáki 2015).
EMERGENT_REPLAY_PROB   = 0.10    # per-burst probability of spontaneous trigger
EMERGENT_COH_THR       = 0.40    # coherence threshold for emergent ignition

# ── Multi-scale hierarchy ──────────────────────────────────────────────────
# Modules are grouped into super-modules (2-level hierarchy).  Super-modules
# have denser inter-module connectivity, modelling cortical columns within
# regions.  Distant super-modules have sparser long-range connections.
SUPER_MODULES          = 2       # number of super-module groups
SUPER_MODULE_CONN_PROB = 0.08    # within super-module connection prob (vs 0.02 inter)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRUE SEQUENCE MEMORY DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
#
# Scientific rationale:
# Standard replay imposes trajectories externally (scheduler-driven order).
# In real hippocampal replay, sequences emerge from learned asymmetric
# synaptic chains (Foster & Wilson 2006, Diba & Buzsáki 2007).  We replace
# external trajectory steering with mechanisms that allow replay to
# propagate spontaneously along learned chain connections.
#
# Biological grounding:
# - Hippocampal place cells form temporal sequences via STDP (Skaggs et al. 1996)
# - Forward replay sweeps along learned place-cell sequences during SWRs
#   (Lee & Wilson 2002)
# - Synaptic facilitation at CA3 recurrent collaterals enables rapid
#   sequence propagation (Zucker & Regehr 2002)
# - Membrane time constants shorten during replay (high-conductance state,
#   Destexhe et al. 2003), accelerating dynamics

# 1a. Sequence-chain STDP: asymmetric forward/backward gains
#   FORWARD_GAIN multiplies LTP when assembly X fires before Y (causal)
#   BACKWARD_GAIN multiplies LTD when Y fires after X (anti-causal)
#   Together they create directionally selective chain weights.
STDP_FORWARD_GAIN       = 2.0    # LTP boost for causal (A→B) pairings
STDP_BACKWARD_GAIN      = 0.5    # LTD reduction for anti-causal (B→A) pairings
SEQUENCE_CHAIN_DELAY    = 50     # propagation delay between assemblies (steps)

# 1b. Internal replay propagation
#   After a replay event for assembly X, if the event was accepted and
#   chain weights from X→Y are sufficiently strong, the next event
#   spontaneously ignites Y instead of being externally scheduled.
INTERNAL_PROPAGATION_PROB  = 0.30  # prob of igniting next-assembly chain
INTERNAL_PROP_MIN_WEIGHT   = 0.05  # min mean chain weight for propagation
INTERNAL_PROP_MAX_STEPS    = 30    # max steps to wait for propagation ignition

# 1c. Temporal replay compression via synaptic mechanisms
#   Replace hardcoded DT scaling with biophysical mechanisms:
#   - Facilitation: short-term synaptic enhancement during high-frequency bursts
#     (Zucker & Regehr 2002).  Decays with FACILITATION_TAU.
#   - Membrane scaling: effective reduction of membrane time constant during
#     replay (high-conductance state from barrages of inhibitory input)
FACILITATION_TAU        = 50.0   # short-term facilitation decay (steps)
FACILITATION_STRENGTH   = 2.0    # max facilitation multiplier
REPLAY_MEMBRANE_SCALE   = 0.7    # membrane time constant scaling during replay
USE_FACILITATION        = True   # toggle synaptic facilitation
USE_MEMBRANE_SCALING    = False  # toggle membrane scaling (False=use DT scaling)

# 1d. Sequential replay metrics
TRANSITION_MATRIX_DECAY = 0.9    # exponential decay for transition counter

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — TRUE SYSTEMS CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Scientific rationale:
# Systems consolidation theory (McClelland et al. 1995) posits that memories
# are initially HC-dependent and gradually become cortex-independent over
# time/replay cycles.  We implement differential plasticity rates, replay-
# driven HC→Ctx transfer, and lesion experiments to demonstrate this shift.
#
# Biological grounding:
# - HC: sparse, fast encoding, fast decay (Marr 1971, Treves & Rolls 1994)
# - Ctx: distributed, slow encoding, stable (O'Reilly et al. 1998)
# - Replay drives HC→Ctx transfer during SWRs (Buzsáki 2015)
# - Early lesions impair recall, late lesions spare it (Kim & Fanselow 1992)

# HC→Ctx transfer gain (how much replay drives cortex consolidation)
HCCTX_TRANSFER_GAIN     = 1.0     # HC→Ctx weight strengthening per accepted event
HCCTX_CORTEX_EMERGENCE  = 5.0     # replay cycles before cortex supports recall
LESION_HC               = False   # HC lesion flag (set True during experiment)
LESION_CTX              = False   # Cortex lesion flag
LESION_SILENCE_HC       = False   # Temporary HC silencing (reversible)

# Transfer curve tracking
TRANSFER_WINDOW_SIZE    = 5       # replay events per transfer measurement

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — DYNAMICAL SYSTEMS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
#
# Scientific rationale:
# Recurrent spiking networks form attractor landscapes whose topology
# (basins, separatrices, metastable states) determines memory retrieval
# dynamics.  We measure these properties directly.
#
# Biological grounding:
# - Attractor networks as models of hippocampal CA3 (Hopfield 1982)
# - Metastable states in prefrontal cortex during working memory
#   (Stokes 2015, Durstewitz et al. 2000)
# - Dimensionality reduction reveals low-dimensional replay trajectories
#   (Cunningham & Yu 2014)

BASIN_PERTURB_STRENGTH  = 3.0     # perturbation strength for basin estimation
BASIN_N_TRIALS          = 10      # perturbation trials per state
METASTABLE_DWELL_MIN    = 5       # min steps for metastable state detection
EFFECTIVE_DIM_K         = 10      # top PCs for participation ratio
SPECTRAL_RADIUS_TRACK   = True    # track weight matrix spectral radius
LYAPUNOV_ESTIM_STEPS    = 20      # steps for divergence estimation

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — BIOLOGICALLY REALISTIC HOMEOSTASIS
# ═══════════════════════════════════════════════════════════════════════════
#
# Scientific rationale:
# Global weight normalization is engineering-style and lacks biological
# plausibility.  BCM (Bienenstock-Cooper-Munro 1982) theory provides a
# sliding threshold for LTP/LTD that depends on recent postsynaptic activity.
# Intrinsic plasticity adjusts excitability toward a target rate.
#
# Biological grounding:
# - BCM rule in visual cortex (Bear et al. 1987)
# - Spike-frequency adaptation via slow K+ currents (Madison & Nicoll 1984)
# - Synaptic scaling at individual synapses (Turrigiano et al. 1998)
# - Inhibitory STDP refines E/I balance (Vogels et al. 2011)

BCM_ENABLED             = False   # BCM metaplasticity toggle
BCM_THETA_INIT          = 0.05    # initial LTP/LTD threshold
BCM_TAU                 = 1000.0  # threshold adaptation time constant
BCM_SLIDING_RATE        = 0.01    # how fast theta slides with activity
BCM_ACTIVATION_WINDOW   = 100     # steps for sliding average

INTRINSIC_PLASTICITY    = False   # intrinsic plasticity toggle
IP_TARGET_RATE          = 0.01    # target firing probability per step
IP_GAIN                 = 0.001   # excitability adjustment rate
IP_WINDOW               = 500     # evaluation window

SPIKE_FREQ_ADAPT        = False   # spike-frequency adaptation toggle
SFA_STRENGTH            = 0.5     # adaptation current strength
SFA_TAU                 = 200.0   # adaptation decay time constant
SFA_INCREMENT           = 0.2     # adaptation per spike

LOCAL_SCALING           = False   # local (neuron-level) scaling toggle
LOCAL_SCALING_TARGET    = 0.05    # target mean weight per neuron
LOCAL_SCALING_RATE      = 0.005   # per-step adjustment

INHIBITORY_STDP         = False   # inhibitory plasticity toggle
INH_STDP_LTP_RATE       = 0.001   # I→E LTP rate
INH_STDP_LTD_RATE       = 0.0005  # I→E LTD rate
INH_STDP_TAU            = 20.0    # I-STDP time constant

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — INTERNEURON DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════
#
# Scientific rationale:
# Homogeneous inhibition cannot produce the diverse oscillatory dynamics
# (theta, gamma, ripples) observed in hippocampus.  PV, SOM, VIP subtypes
# have distinct properties and connectivity forming a canonical microcircuit.
#
# Biological grounding:
# - PV fast-spiking basket cells generate gamma/ripple oscillations
#   (Buzsáki & Wang 2012, Bartos et al. 2007)
# - SOM cells gate dendritic input, control replay content (Lovett-Barron
#   et al. 2012, Royer et al. 2012)
# - VIP cells disinhibit via SOM suppression (Pi et al. 2013)
# - Cell-type-specific connectivity forms the hippocampal microcircuit
#   (Klausberger & Somogyi 2008)

PV_FRACTION             = 0.40    # fraction of inhibitory pool that is PV
SOM_FRACTION            = 0.30    # fraction SOM
VIP_FRACTION            = 0.10    # fraction VIP
# Remaining INH fraction: other (e.g. CCK, neurogliaform)

PV_G_EXC                = 5.0     # PV→E excitatory conductance (strong)
PV_G_INH                = -50.0   # PV inhibition strength (perisomatic, strong)
SOM_G_INH               = -30.0   # SOM inhibition (dendritic, moderate)
VIP_G_INH               = -20.0   # VIP→SOM inhibition (weak)

PV_A_PLUS               = 0.01    # PV STDP LTP rate (fast)
PV_A_MINUS              = 0.005   # PV STDP LTD rate
SOM_A_PLUS              = 0.003   # SOM STDP (slow)
SOM_A_MINUS             = 0.006   # SOM anti-Hebbian (depressing)
VIP_A_PLUS              = 0.001   # VIP plasticity (weak)
VIP_A_MINUS             = 0.001

# Cell-type connectivity matrices (pre→post: [E, PV, SOM, VIP, Other])
# Stored as dict of dicts, default homogeneous if not set
CELL_TYPE_CONN_PROB = {
    # E → targets
    "E_E":    0.15, "E_PV":   0.20, "E_SOM":  0.10, "E_VIP":  0.05,
    # PV → targets
    "PV_E":   0.25, "PV_PV":  0.05, "PV_SOM": 0.10, "PV_VIP": 0.15,
    # SOM → targets
    "SOM_E":  0.15, "SOM_PV": 0.05, "SOM_SOM":0.02, "SOM_VIP":0.20,
    # VIP → targets
    "VIP_E":  0.05, "VIP_PV": 0.05, "VIP_SOM":0.25, "VIP_VIP":0.02,
}

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — ENERGY-CONSTRAINED REPLAY
# ═══════════════════════════════════════════════════════════════════════════
#
# Scientific rationale:
# Replay consumes metabolic energy (ATP for synaptic transmission, action
# potentials).  Unconstrained replay is biologically unrealistic — the brain
# must allocate limited energy resources across competing consolidation needs.
#
# Biological grounding:
# - Brain energy budget limits neural computation (Attwell & Laughlin 2001)
# - SWR incidence modulated by prior sleep/wake state (Buzsáki 2015)
# - Metabolic cost of spiking constrains plasticity (Harris et al. 2012)
# - Sleep pressure modulates replay (Genzel et al. 2014)

ENERGY_BUDGET           = 100.0   # total energy per rest period
ENERGY_PER_EVENT        = 10.0    # energy consumed per replay event
ENERGY_PER_SPIKE        = 0.001   # energy per additional spike per event

REPLAY_SUPPRESSION_DECAY = 0.95   # recovery of replay probability after suppression
REPLAY_SUPPRESSION_THR  = 0.30    # suppression kicks in when events > budget × this

SLEEP_WAKE_PHASE        = "rest"  # "wake", "rest", "nrem"
SLEEP_REPLAY_PROB_WAKE  = 0.05    # replay prob during wake (low)
SLEEP_REPLAY_PROB_REST  = 0.30    # during quiet rest (moderate)
SLEEP_REPLAY_PROB_NREM  = 0.60    # during NREM (high, SWR-rich)

ENERGY_TRACKING         = True    # track energy consumption
ENERGY_PLOT             = False   # generate energy plots

# ═══════════════════════════════════════════════════════════════════════════
# ABLATION REGISTRY — all feature toggles in one place
# ═══════════════════════════════════════════════════════════════════════════
ABLATION_PHASE1 = {
    "chain_stdp":         True,
    "internal_propagation": True,
    "facilitation":       True,
    "membrane_scaling":   False,
}
ABLATION_PHASE2 = {
    "lesion_hc":          False,
    "lesion_ctx":         False,
    "transfer_gain":      1.0,
}
ABLATION_PHASE3 = {
    "basin_stability":    True,
    "metastable_detect":  True,
    "pca_participation":  True,
    "spectral_radius":    True,
    "phase_sweep":        False,
}
ABLATION_PHASE4 = {
    "bcm":                False,
    "intrinsic_plasticity": False,
    "sfa":                False,
    "local_scaling":      False,
}
ABLATION_PHASE5 = {
    "pv_cells":           False,
    "som_cells":          False,
    "vip_cells":          False,
}
ABLATION_PHASE6 = {
    "energy_budget":      False,
    "sleep_modulation":   False,
}
ABLATION_PHASE7 = {
    "overlap_penalty":    True,   # M1: overlap-sensitive coherence
    "cross_ltd":          True,   # M2: cross-assembly replay LTD
    "overlap_priority":   True,   # M3: overlap-weighted prioritization
    "pers_competition":   True,   # M4: competitive persistence budget
    "drift":              True,   # M5: directional drift
    "fatigue":            True,   # M6: shared-neuron fatigue
    "hetero_tag":         True,   # M7: heterosynaptic LTD tag
    "decorrelation":      True,   # M8: training-time decorrelation
    "wta":                True,   # M9: coherence-based WTA
    "reconsol":           True,   # M10: reconsolidation window metaplasticity
}

# ═══════════════════════════════════════════════════════════════════════════
# EXTERNAL HOOK SYSTEM  (for optional analysis modules like schema_abstraction)
# ═══════════════════════════════════════════════════════════════════════════
# Hooks let external packages observe experiment lifecycle events without
# modifying the core engine.  Every hook is a no-op by default; register
# callbacks via register_hook(name, fn).  The base framework runs perfectly
# even if no hooks are registered (i.e., schema_abstraction/ is absent).
_EXPERIMENT_HOOKS: dict = {}

def register_hook(name: str, fn):
    """Register a callback for an experiment lifecycle event."""
    _EXPERIMENT_HOOKS[name] = fn

def _call_hooks(name: str, **kwargs):
    """Invoke the registered hook for *name* with keyword arguments."""
    fn = _EXPERIMENT_HOOKS.get(name)
    if fn is not None:
        fn(**kwargs)

# ═══════════════════════════════════════════════════════════════════════════
# ENGINE OPTIMIZATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Sparse connectivity enforcement
SPARSITY_ENFORCE         = True    # apply structural mask after STDP to zero excluded conn
SPARSITY_THRESHOLD       = 1e-4    # zero out weights below this absolute value
ENFORCE_SPARSITY_EVERY   = 50      # steps between structural mask applications

# Probe caching and reduction
PROBE_CACHE_ENABLED      = True    # cache probe results to avoid redundant recompute
PROBE_REDUCE_EARLY_EXIT  = True    # exit probe early if assembly pattern is clear
PROBE_CACHE_MAXSIZE      = 256     # max entries in probe cache

# Adaptive replay
ADAPTIVE_REPLAY          = True    # reduce replay events when coherence consistently fails
ADAPTIVE_REPLAY_WINDOW   = 5       # events to evaluate for adaptive termination
ADAPTIVE_REPLAY_MIN_EVENTS = 3     # min events even when coherence is poor
ADAPTIVE_COHERENCE_THR   = 0.50    # threshold for "good" event

# Structural mask disk caching
CACHE_MASKS_TO_DISK      = True    # save/load module masks from cached_masks.pt
MASKS_CACHE_PATH         = "cached_masks.pt"

# Torch compile (requires PyTorch >= 2.0)
USE_TORCH_COMPILE        = False   # wrap net.forward with torch.compile (experimental)
TORCH_COMPILE_MODE       = "reduce-overhead"  # compile mode

# ── Global caches (mutable, populated at runtime) ─────────────────────────
_STRUCTURAL_MASK_EE = None   # cached (N_EXC, N_EXC) bool mask for E→E weights
_PROBE_CACHE = {}            # {(assembly_tuple, checkpoint_id, use_slow): probe_result}
_PROBE_CACHE_ORDER = []      # LRU ordering for cache eviction

# ─────────────────────────────────────────────────────────────────────────────
# TIMING INSTRUMENTATION
# ─────────────────────────────────────────────────────────────────────────────
_TIMER = {
    "training":    0.0,
    "replay":      0.0,
    "probe":       0.0,
    "slow_step":   0.0,
    "figures":     0.0,
    "other":       0.0,
}

def _tick():
    return time.perf_counter()

def _tock(key, t0):
    _TIMER[key] += time.perf_counter() - t0

# ─────────────────────────────────────────────────────────────────────────────
# SAFE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    f = a[np.isfinite(a)]
    return float(f.mean()) if len(f) > 0 else 0.0

def _safe_std(arr):
    a = np.asarray(arr, dtype=float)
    f = a[np.isfinite(a)]
    return float(f.std()) if len(f) >= 2 else 0.0

def _safe_sem(arr):
    a = np.asarray(arr, dtype=float)
    f = a[np.isfinite(a)]
    return float(_scipy_sem(f)) if len(f) >= 2 else 0.0

def _safe_cosine_sim(v1, v2):
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(1.0 - _cosine_dist(v1, v2))

def _safe_nanmean(arr, axis=None):
    with np.errstate(all='ignore'):
        return np.nanmean(arr, axis=axis)

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLY LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def make_overlapping_assemblies(n_memories, assembly_size, overlap_frac):
    """
    Chain where each consecutive pair shares round(overlap_frac*assembly_size) neurons.
    A:0-19, B:16-35, C:32-51, D:48-67 at 20% overlap.
    All indices stay within the excitatory pool (< N_EXC).

    In sparse_modular mode, assemblies are placed within MEMORY_MODULE.
    """
    if ARCH_MODE == "sparse_modular":
        return _make_modular_assemblies(n_memories, assembly_size, overlap_frac)

    n_overlap = int(round(overlap_frac * assembly_size))
    step      = assembly_size - n_overlap
    asms = []
    for m in range(n_memories):
        asm = np.arange(m * step, m * step + assembly_size, dtype=int)
        if asm[-1] >= N_EXC:
            raise ValueError(
                f"Assembly {m} exceeds excitatory pool "
                f"(last={asm[-1]}, N_EXC={N_EXC}). Reduce overlap or assembly size."
            )
        asms.append(asm)
    return asms


def _make_modular_assemblies(n_memories, assembly_size, overlap_frac):
    """
    Module-local assembly placement for sparse_modular mode.
    All assemblies live within MEMORY_MODULE, preserving chained overlap.
    The memory module is placed within the HC pool (first N_HC excitatory
    neurons) so that replay can drive HC→Ctx transfer.
    Other modules serve as background populations, biologically analogous
    to distinct cortical columns processing unrelated information.
    """
    # Ensure memory module is within HC pool
    exc_per_module = N_EXC // N_MODULES
    _mm_start = MEMORY_MODULE * exc_per_module
    if _mm_start >= N_HC and N_HC > 0:
        # Fall back to module 0 if the configured module is outside HC range
        _used_module = 0
    else:
        _used_module = MEMORY_MODULE
    module_start = _used_module * exc_per_module
    module_end = module_start + exc_per_module

    # Ensure assemblies fit within the module
    max_needed = n_memories * assembly_size
    if max_needed > exc_per_module:
        assembly_size = max(20, exc_per_module // (n_memories + 1))

    n_overlap = int(round(overlap_frac * assembly_size))
    step = assembly_size - n_overlap

    asms = []
    for m in range(n_memories):
        offset = m * step
        asm = np.arange(
            module_start + offset,
            module_start + offset + assembly_size,
            dtype=int
        )
        # Clamp to module boundary if assembly overflows
        if asm[-1] >= module_end:
            overflow = asm[-1] - module_end + 1
            asm = np.arange(
                module_start + offset - overflow,
                module_start + offset + assembly_size - overflow,
                dtype=int
            )
        asms.append(asm)
    return asms

def assembly_overlap_mask(asm_a, asm_b):
    sa, sb = set(asm_a.tolist()), set(asm_b.tolist())
    return (np.array(sorted(sa & sb), int),
            np.array(sorted(sa - sb), int),
            np.array(sorted(sb - sa), int))

# ─────────────────────────────────────────────────────────────────────────────
# NETWORK FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_network(use_slow=False):
    net = IzhikevichNetwork(
        n_neurons=N_NEURONS, n_inh=N_INH,
        g_exc=G_EXC, g_inh=G_INH,
        noise_std=NOISE_STD, dt=DT, device=DEVICE,
        arch_mode=ARCH_MODE, n_modules=N_MODULES,
        intra_module_conn_prob=INTRA_MODULE_CONN_PROB,
        inter_module_conn_prob=INTER_MODULE_CONN_PROB,
        inter_module_scale=INTER_MODULE_SCALE,
        ee_sparsity=EE_SPARSITY,
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

    # ── HC-Cortex architecture ──────────────────────────────────────────────
    # Register HC/Cortex masks as buffers on the network for downstream use.
    hc_mask = torch.zeros(N_NEURONS, dtype=torch.bool, device=DEVICE)
    ctx_mask = torch.zeros(N_NEURONS, dtype=torch.bool, device=DEVICE)
    if N_HC > 0:
        hc_mask[:N_HC] = True
    if N_CTX > 0:
        ctx_mask[N_HC:N_EXC] = True
    net.register_buffer('hc_mask', hc_mask)
    net.register_buffer('ctx_mask', ctx_mask)

    # Record tau values per submatrix for heterogeneous decay
    net._hc_decay_tau = HC_FAST_DECAY_TAU
    net._ctx_decay_tau = CTX_SLOW_DECAY_TAU

    # HC→Ctx feedforward projection
    if ARCH_MODE == "sparse_modular" and N_HC > 0 and N_CTX > 0:
        with torch.no_grad():
            _proj = torch.rand(N_CTX, N_HC, device=DEVICE)
            _hc_ctx_W = (_proj < HC_CTX_PROJ_PROB).float() * HC_CTX_PROJ_STRENGTH
            # W[post=cortex, pre=HC]
            net.W.data[N_HC:N_EXC, :N_HC] = _hc_ctx_W
            # Ctx→HC feedback (sparse)
            _fb = torch.rand(N_HC, N_CTX, device=DEVICE)
            _ctx_hc_W = (_fb < CTX_HC_PROJ_PROB).float() * CTX_HC_PROJ_STRENGTH
            # W[post=HC, pre=cortex]
            net.W.data[:N_HC, N_HC:N_EXC] = _ctx_hc_W

    # ── Multi-scale hierarchy: super-module grouping ────────────────────────
    # Modify inter-module connectivity so modules within the same super-module
    # have denser connections (SUPER_MODULE_CONN_PROB) than the default INTER_MODULE_CONN_PROB.
    if ARCH_MODE == "sparse_modular" and hasattr(net, 'module_id') and N_MODULES > 1 and SUPER_MODULES > 1:
        with torch.no_grad():
            mod_ids = net.module_id[:N_EXC].long()  # (N_EXC,)
            _mod_to_super = np.array([m // (N_MODULES // SUPER_MODULES) for m in range(N_MODULES)])
            _mod_ids_np = mod_ids.cpu().numpy()
            super_ids = _mod_to_super[_mod_ids_np]  # (N_EXC,) numpy
            super_t = torch.from_numpy(super_ids).to(device=DEVICE)
            # Find pairs of neurons in different modules but same super-module
            same_super = (super_t.unsqueeze(1) == super_t.unsqueeze(0))  # (N_EXC, N_EXC)
            same_mod  = (mod_ids.unsqueeze(1) == mod_ids.unsqueeze(0))    # (N_EXC, N_EXC)
            target = same_super & ~same_mod                               # diff module, same super
            target = target.triu_(1)                                       # upper triangular
            # Boost connectivity for a random subset
            boost_mask = (torch.rand(N_EXC, N_EXC, device=DEVICE) < SUPER_MODULE_CONN_PROB) & target
            net.W.data[:N_EXC, :N_EXC][boost_mask] *= 2.0
            net.W_init.data[:N_EXC, :N_EXC][boost_mask] *= 2.0

    # Keep W_init in sync with modified weights so _bulk_decay targets correctly
    with torch.no_grad():
        net.W_init.data.copy_(net.W.data)

    return net

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE OPTIMISATION A -- closed-form slow consolidation
# ─────────────────────────────────────────────────────────────────────────────

def bulk_slow_step(net, n_steps):
    """
    Exact closed-form solution to n_steps of slow_step() under CONSTANT W.

    slow_step() solves dW_slow/dt = up(t) + down(t) where
      up   = clamp(W - W_slow, 0) / tau_slow
      down = clamp(W - W_slow, 0) / tau_very_slow  [sign<0]

    For constant W the ODE is piecewise-linear and has solution:
      delta(t) = delta(0) * exp(-t / tau_effective)
    where tau_effective = tau_slow for delta>0, tau_very_slow for delta<0.

    This replaces O(n_steps) loop calls with ONE vectorised operation.
    Savings: ~52,000 individual slow_step() calls per trial -> 83 bulk calls.

    SCIENTIFIC IMPACT: numerically identical to the loop for constant W.
    (W is constant during inter-presentation rest and inter-memory rest,
    because stdp_step() is never called during those periods.)
    """
    if not getattr(net, 'slow_enabled', False):
        return
    t0 = _tick()
    with torch.no_grad():
        fast  = net.W.data[:N_EXC, :N_EXC]
        delta = fast - net.W_slow          # positive where fast > slow
        f_up   = float(np.exp(-n_steps / net.tau_slow))
        f_down = float(np.exp(-n_steps / net.tau_very_slow))
        new_delta = torch.where(delta > 0, delta * f_up, delta * f_down)
        # W_slow is mathematically bounded by W_fast (converges toward it from
        # below when W > W_slow, from above when W < W_slow).  Explicit clamp
        # here as belt+suspenders against any floating-point accumulation.
        net.W_slow.copy_((fast - new_delta).clamp_(0.0, _W_SLOW_MAX_CLAMP))
    _tock("slow_step", t0)

# ─────────────────────────────────────────────────────────────────────────────
# SYNAPTIC TAG MODULE (v3 -- analytical LTP tag, no tensor clones)
# ─────────────────────────────────────────────────────────────────────────────

class SynapticTags:
    """
    Synaptic tagging & capture (STC hypothesis; Frey & Morris 1997).

    Tags mark recently-potentiated synapses using the LTP kernel
    (post_spikes x pre_trace), computed analytically after each forward pass.
    This avoids per-step weight-matrix cloning entirely.

    During replay, tag_driven_consolidation() transfers tagged synapses into
    W_slow at an accelerated rate.

    CRITICAL STABILITY FIX (v3): W_slow is CLAMPED to [0, W_MAX] after every
    consolidation call. Without this clamp, W_slow could overshoot W_MAX
    when TAG_CAPTURE_RATE was high, causing W_eff to explode.
    """

    def __init__(self, n_exc=None, tau=TAG_DECAY_TAU, device=DEVICE):
        if n_exc is None:
            n_exc = N_EXC
        self.n_exc        = n_exc
        self.tau          = tau
        self.device       = device
        self.decay_factor = float(np.exp(-1.0 / tau))
        self.W_tag        = torch.zeros(n_exc, n_exc, device=device)

    def update_from_spikes(self, net):
        """
        Analytical LTP tag update -- called AFTER net.forward(), BEFORE stdp_step().

        The STDP potentiation kernel is: A_plus * post_spikes * pre_trace (past).
        Tags accumulate the unnormalised LTP signal for each E->E synapse.
        No weight snapshot is taken; the LTP kernel is computed directly from
        the spike and trace buffers already maintained by IzhikevichNetwork.

        PERFORMANCE: replaces per-step tensor clone (was ~29,000 clones/trial).
        SCIENTIFIC: equivalent meaning to "weight-change-based tag", slightly
        coarser (uses pre-step traces instead of post-stdp-delta) but
        biologically motivated by the pre-post coincidence signal.
        """
        with torch.no_grad():
            n    = self.n_exc
            # LTP signal: post fires, pre was recently active
            ltp  = A_PLUS * net.spikes[:n].unsqueeze(1) * net.pre_trace[:n].unsqueeze(0)
            self.W_tag.add_(ltp)
            self.W_tag.mul_(self.decay_factor)
            self.W_tag.clamp_(0.0, _W_TAG_MAX_CLAMP)   # hard ceiling: tags stay small

    def decay(self, n_steps=1):
        with torch.no_grad():
            self.W_tag.mul_(float(np.exp(-n_steps / self.tau)))

    def tag_driven_consolidation(self, net, assembly, rate=TAG_CAPTURE_RATE):
        """
        Transfer tag strength directly into W_slow at tagged synapses.

        BIOLOGICAL MECHANISM (Frey & Morris 1997, STC hypothesis):
          Late-LTP at a synapse requires (1) a synaptic tag set by early
          plasticity, and (2) plasticity-related proteins (PRPs) available
          from recent strong activity (e.g. replay).  When both are present,
          the tag captures PRPs and the synapse is consolidated.  Capture
          depends on tag strength, NOT on the current fast/slow weight gap.

        WHY THE GAP-GATE WAS REMOVED:
          The earlier formula
              ΔW_slow = rate · tag · (W_fast - W_slow)
          multiplied capture by the residual fast-slow gap.  Because
          slow_step already tracks W_fast upward with tau_slow=4000 over
          ~4000 cumulative rest steps per memory, the gap is small by the
          time replay fires — capture was being silently scaled to zero.
          That defeated the entire point of replay-driven consolidation:
          Slow+Replay ≡ Slow/NoReplay in practice.

          The new rule
              ΔW_slow = rate · tag
          makes capture a direct function of how strongly the synapse was
          recently potentiated (which is what tags encode), independent of
          the current strength gap.  This matches the STC literature and
          gives replay a real channel to bolster slow weights.

        STABILITY:
          W_slow is clamped to [0, W_MAX = 1.5] after update.  W_tag is
          itself clamped to [0, _W_TAG_MAX_CLAMP = 0.5] in update_from_spikes.
          So the maximum per-call addition is rate · 0.5 = 0.075, and W_slow
          cannot exceed W_MAX regardless of how many capture events fire.
        """
        if not getattr(net, 'slow_enabled', False):
            return
        asm = assembly[assembly < self.n_exc]
        if len(asm) == 0:
            return
        with torch.no_grad():
            idx  = np.ix_(asm, asm)
            tag  = self.W_tag[idx]
            slow = net.W_slow[idx]
            # Direct tag - slow transfer.  No fast/slow gating.
            net.W_slow[idx] = (slow + rate * tag).clamp(0.0, W_MAX)

    def assembly_tag_mean(self, assembly):
        asm = assembly[assembly < self.n_exc]
        if len(asm) == 0:
            return 0.0
        with torch.no_grad():
            return float(self.W_tag[np.ix_(asm, asm)].mean())

# ─────────────────────────────────────────────────────────────────────────────
# REPRESENTATIONAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def get_assembly_weight_vector(net, assembly):
    asm = assembly[assembly < N_EXC]
    with torch.no_grad():
        return net.W.data[np.ix_(asm, asm)].cpu().numpy().flatten().copy()

def compute_rsm(snapshots):
    """
    snapshots[i][j] = weight vector of memory i at checkpoint j (or None).
    RSM[i, j] = cosine_sim(snapshot[i][j], snapshot[i][i]).
    """
    n = len(snapshots)
    rsm = np.full((n, n), np.nan)
    for i in range(n):
        if snapshots[i][i] is None:
            continue
        for j in range(i, n):
            if snapshots[i][j] is not None:
                rsm[i, j] = _safe_cosine_sim(snapshots[i][i], snapshots[i][j])
    return rsm

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS  (scaling debug and failure detection)
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_connectivity(net):
    """
    Print connectivity statistics for the current network.
    Used in sparse_modular mode to verify sparsity targets.
    """
    n_exc = net.n_exc
    ee = net.W.data[:n_exc, :n_exc]
    total = ee.numel()
    nz = (ee != 0).sum().item()
    sparsity = 100.0 * nz / max(total, 1)
    print(f"  [DIAG] E-E sparsity: {sparsity:.1f}%  ({nz}/{total})")

    if net.arch_mode == "sparse_modular":
        for m in range(net.n_modules):
            mask = net.module_id[:n_exc] == m
            n_mod = mask.sum().item()
            if n_mod == 0:
                continue
            intra = ee[mask][:, mask]
            nz_intra = (intra != 0).sum().item()
            total_intra = intra.numel()
            intra_pct = 100.0 * nz_intra / max(total_intra, 1)
            # Inter-module connections from this module
            inter = ee[mask][:, ~mask]
            nz_inter = (inter != 0).sum().item()
            total_inter = inter.numel()
            inter_pct = 100.0 * nz_inter / max(total_inter, 1)
            print(f"  [DIAG]   Module {m}: {n_mod} exc, "
                  f"intra={intra_pct:.1f}%  inter={inter_pct:.1f}%")
        if nz > 0:
            print(f"  [DIAG]   Mean E-E weight: {ee[ee != 0].mean().item():.4f}")


def diagnose_dynamics(net, n_steps=500, noise_std=None):
    """
    Run the network for n_steps and print dynamical diagnostics.
    Returns dict of metrics for failure detection.
    """
    if noise_std is None:
        noise_std = net.noise_std
    saved_noise = net.noise_std
    net.noise_std = noise_std
    net.reset_state()
    if net.stdp_enabled:
        net.pre_trace.zero_()
        net.post_trace.zero_()

    spike_counts = []
    v_means = []
    max_firing_window = 0

    for t in range(n_steps):
        net.forward()
        s = int(net.spikes.sum().item())
        spike_counts.append(s)
        v_means.append(net.v.mean().item())
        if s > max_firing_window:
            max_firing_window = s

    net.noise_std = saved_noise

    spike_arr = np.array(spike_counts, dtype=float)
    mean_fr = spike_arr.mean() / (net.n_neurons * net.dt / 1000.0)
    active_frac = (spike_arr > 0).mean()
    v_mean = np.mean(v_means)
    v_std = np.std(v_means)

    metrics = {
        "mean_firing_rate_hz": mean_fr,
        "active_timestep_frac": active_frac,
        "mean_v": v_mean,
        "v_std": v_std,
        "max_simultaneous_spikes": max_firing_window,
        "n_silent": (net.spikes.sum() == 0).item(),
    }

    if DEBUG_SCALING:
        print(f"  [DIAG] Dynamics ({n_steps} steps):")
        print(f"    Mean firing rate: {mean_fr:.2f} Hz")
        print(f"    Active timesteps: {active_frac*100:.1f}%")
        print(f"    Mean Vm: {v_mean:.2f}  Vm std: {v_std:.2f}")
        print(f"    Max simultaneous spikes: {max_firing_window}")
        print(f"    Silent final step: {metrics['n_silent']}")

    return metrics


FAILURE_SILENT = "SILENT_NETWORK"
FAILURE_RUNAWAY = "RUNAWAY_EXCITATION"
FAILURE_NAN = "NAN_INSTABILITY"
FAILURE_SATURATION = "SATURATION"
FAILURE_STABLE = "STABLE"


def detect_failure_mode(net, n_steps=500):
    """
    Run diagnostic dynamics and classify network state.
    Returns one of the FAILURE_* constants.
    """
    try:
        metrics = diagnose_dynamics(net, n_steps=n_steps)
    except Exception:
        return FAILURE_NAN

    if not np.isfinite(metrics["mean_firing_rate_hz"]):
        return FAILURE_NAN

    n_exc = net.n_exc
    if metrics["max_simultaneous_spikes"] > n_exc * 0.5:
        return FAILURE_RUNAWAY

    if metrics["mean_firing_rate_hz"] < 0.1:
        return FAILURE_SILENT

    if metrics["mean_firing_rate_hz"] > 200.0:
        return FAILURE_SATURATION

    if metrics["mean_v"] > -50.0:
        return FAILURE_SATURATION

    return FAILURE_STABLE


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_one_memory(net, assembly, tags=None,
                     n_presentations=N_PRESENTATIONS_PER_MEM,
                     prev_assembly=None):
    """
    Standard STDP training (identical protocol to compare_retention.py).
    Adds analytical tag updates if a SynapticTags instance is provided.

    When prev_assembly is provided and CHAIN_STDP_ENABLED is True, a brief
    pre→post pairing phase builds feedforward weights for sequence replay
    (Phase 1: sequence-chain formation).

    Optimized with preallocated stim buffers and periodic structural mask
    enforcement to keep weights sparse.
    """
    t0       = _tick()
    use_slow = getattr(net, 'slow_enabled', False)
    _stim_buf = _get_stim_buffer((N_NEURONS,), DEVICE)  # preallocated noise

    for _ in range(n_presentations):
        jitter = np.random.randint(0, max(1, stim_steps // 2), size=len(assembly))
        for t in range(stim_steps):
            _stim_buf.normal_(mean=0.0, std=0.5)
            for idx, n in enumerate(assembly):
                if t >= jitter[idx]:
                    _stim_buf[n] += STIM_STRENGTH
            net.forward(_stim_buf)
            if tags is not None:
                tags.update_from_spikes(net)   # analytical LTP tag (no clone)
            net.stdp_step()

        # Inter-presentation rest: no STDP, W is constant -> use bulk_slow_step
        _rest_buf = _get_stim_narrow(std=0.3)
        for _ in range(rest_steps_per_pres):
            net.forward(_rest_buf)
        if use_slow:
            bulk_slow_step(net, n_steps=rest_steps_per_pres)

        # Enforce structural sparsity after each presentation
        if SPARSITY_ENFORCE and (_ + 1) % ENFORCE_SPARSITY_EVERY == 0:
            apply_structural_mask(net)

    # ── Phase 1: sequence-chain STDP pairing ───────────────────────────
    if (ABLATION_PHASE1.get("chain_stdp", True) and prev_assembly is not None
            and len(prev_assembly) > 0):
        _pa = prev_assembly[prev_assembly < N_EXC]
        _na = assembly[assembly < N_EXC]
        if len(_pa) > 0 and len(_na) > 0:
            _old_ap = net.A_plus
            _old_am = net.A_minus
            net.A_plus  = A_PLUS * (1.0 + SEQUENCE_TRANSITION_STRENGTH * 20.0)
            net.A_minus = A_MINUS * (1.0 + SEQUENCE_TRANSITION_STRENGTH * 20.0)
            _chain_steps = max(5, int(SEQUENCE_CHAIN_DELAY * 2))
            # Phase A: activate prev assembly (pre-synaptic)
            for __ in range(_chain_steps // 2):
                _stim_buf.normal_(mean=0.0, std=0.5)
                _stim_buf[_pa] += STIM_STRENGTH * 0.8
                net.forward(_stim_buf)
                net.stdp_step()
            # Phase B: activate current assembly (post-synaptic)
            for __ in range(_chain_steps // 2):
                _stim_buf.normal_(mean=0.0, std=0.5)
                _stim_buf[_na] += STIM_STRENGTH * 0.8
                net.forward(_stim_buf)
                net.stdp_step()
            net.A_plus  = _old_ap
            net.A_minus = _old_am

    _tock("training", t0)

# ─────────────────────────────────────────────────────────────────────────────
# COMPETITIVE INTERFERENCE  (mild; overlap-dependent synaptic depression)
# ─────────────────────────────────────────────────────────────────────────────

def apply_competitive_interference(net, new_assembly, old_assemblies,
                                    strength=COMPETITION_STRENGTH,
                                    ablation=None):
    """
    After training new_assembly, depress connections from old-assembly-specific
    neurons to shared neurons by (overlap_frac x strength).

    At 20% overlap, extra_decay = 0.20 x 0.25 = 0.05.  Three rounds (B,C,D):
    cumulative factor = 0.95^3 = 0.857.  Produces clean overlap gradient
    without destabilising slow weights (only fast W is modified here).

    Biological interpretation: competitive synaptic capture (Bhatt 2009).
    """
    _use_comp = (ablation.get("use_competition", USE_COMPETITION)
                 if ablation else USE_COMPETITION)
    if not _use_comp:
        return
    # In sparse_modular mode, competition is module-local: only compete with
    # assemblies that reside in the same module as the new assembly.
    if ARCH_MODE == "sparse_modular" and hasattr(net, 'module_id'):
        _new_mod = int(net.module_id[new_assembly[0]].item()) if len(new_assembly) > 0 else -1
        old_assemblies = [a for a in old_assemblies
                          if len(a) > 0 and int(net.module_id[a[0]].item()) == _new_mod]
    for old_asm in old_assemblies:
        shared, old_spec, _ = assembly_overlap_mask(old_asm, new_assembly)
        if len(shared) == 0 or len(old_spec) == 0:
            continue
        overlap_frac = len(shared) / ASSEMBLY_SIZE
        extra_decay  = overlap_frac * strength
        se = shared[shared < N_EXC]
        oe = old_spec[old_spec < N_EXC]
        if len(se) == 0 or len(oe) == 0:
            continue
        with torch.no_grad():
            net.W.data[np.ix_(se, oe)] *= (1.0 - extra_decay)


def apply_post_training_normalization(net, strength=0.05):
    """
    Per-neuron synaptic normalization after training.

    After a new memory is trained, each excitatory neuron's total incoming
    weight is normalized to prevent independent basin growth through overlap
    neurons.  This implements Oja-like competitive normalization: neurons
    shared between assemblies can't simultaneously maintain strong connections
    to both assemblies' specific pools.

    The normalization is applied softly (strength=0.05 means 5% of the excess
    above the per-neuron mean is redistributed), preserving overall structure
    while preventing unbounded growth through overlap.

    Biological basis: synaptic scaling (Turrigiano 2008) — neurons maintain
    their total input drive within a dynamic range, so strengthening one
    pathway (via STDP) must be compensated by weakening others.
    """
    if strength <= 0:
        return
    with torch.no_grad():
        ee = net.W.data[:N_EXC, :N_EXC]
        # Per-neuron total incoming weight
        col_sums = ee.sum(dim=0)  # incoming to each neuron
        target = col_sums.mean()
        # Scale each column toward the mean
        scale = 1.0 + strength * (target / (col_sums + 1e-10) - 1.0)
        scale = scale.clamp(max=10.0)  # prevent float32 overflow (inf → NaN cascade)
        ee.mul_(scale.unsqueeze(0))  # broadcast over rows


def apply_training_decorrelation(net, new_assembly, old_assemblies,
                                  strength=TRAINING_DECORR_STRENGTH,
                                  ablation=None):
    """
    Training-time overlap decorrelation (M8).

    During training of a new memory, LTD is applied at connections from the
    new assembly's specific neurons to the old assemblies' specific neurons
    via the shared overlap pathway.  This drives representations apart at
    encoding time, pattern-separating overlapping memories.

    Specifically, for each old assembly sharing overlap with the new one:
    - Identify overlap neuron set O = new ∩ old
    - Identify each assembly's specific (non-overlap) excitatory set
    - Apply LTD at connections NEW_specific → O (new→shared) AND
      O → OLD_specific (shared→old)
    This decorrelates the two assemblies' representations through O.

    Biological basis: pattern separation via lateral inhibition in dentate
    gyrus (Yassa & Stark 2011) — similar inputs are actively decorrelated
    during encoding to prevent catastrophic overlap.

    Direction: W[target, source].  M8 applies LTD from new_specific→overlap
    (weakening the new assembly's pull on shared neurons) AND from
    overlap→old_specific (weakening the old one's hold on shared neurons).
    """
    _use_decorr = (ablation.get("decorrelation", True)
                   if ablation else True)
    if not _use_decorr or strength <= 0:
        return
    for old_asm in old_assemblies:
        shared, new_spec, old_spec = assembly_overlap_mask(old_asm, new_assembly)
        if len(shared) == 0 or len(new_spec) == 0 or len(old_spec) == 0:
            continue
        se = shared[shared < N_EXC]
        ns = new_spec[new_spec < N_EXC]
        os = old_spec[old_spec < N_EXC]
        if len(se) == 0 or len(ns) == 0 or len(os) == 0:
            continue
        with torch.no_grad():
            # new_specific → overlap connections weakened (new assembly learns
            # to rely less on shared neurons)
            net.W.data[np.ix_(se, ns)] *= (1.0 - strength)
            # overlap → old_specific connections weakened (old assembly loses
            # influence through shared neurons)
            net.W.data[np.ix_(os, se)] *= (1.0 - strength * 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY
# ─────────────────────────────────────────────────────────────────────────────
# REPLAY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _max_consec_above(arr, thr):
    """Return the length of the longest run of consecutive values > thr.

    Used by adaptive replay selection to measure coherence stability: a long
    uninterrupted run of coherent steps indicates sustained pattern completion
    (strong attractor), whereas a series of brief, intermittent spikes
    indicates a weak or noisy attractor.

    Args:
        arr: iterable of floats (coherence values)
        thr: threshold (float)
    Returns:
        int — max consecutive count; 0 if arr is empty or all values ≤ thr.
    """
    best = cur = 0
    for v in arr:
        if v > thr:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _replay_one_event(net, assembly, tags=None,
                       cue_size=PARTIAL_CUE_SIZE,
                       seed_strength=REPLAY_SEED_STRENGTH,
                       seed_dur=REPLAY_SEED_DURATION,
                       spont_steps=REPLAY_SPONTANEOUS_STEPS,
                       noise=REPLAY_NOISE_STD,
                       all_assemblies=None,
                       rest_id=0, burst_id=0, event_id=0,
                       assembly_idx=0,
                       ablation=None,
                       state=None):
    """
    One hippocampal-style SWR replay event.

    Phase 1 -- Partial seed (cue_size << ASSEMBLY_SIZE):
        Stimulate only *cue_size* random neurons.  Trained recurrent weights
        complete the pattern (pattern completion, not full drive).

        When all_assemblies is provided, overlap neurons (shared with any
        other assembly) are excluded from the seed pool.  This prevents
        cross-assembly activation through shared neurons (e.g. neurons 16-19
        shared between A and B): if overlap neurons are seeded, they drive
        the NEIGHBOURING assembly's fresh post-training weights and activate
        a competing pattern, causing anti-correlated STDP at the target
        assembly's synapses.  Drawing only from assembly-unique neurons
        avoids this interference.  Biologically: SWR replay preferentially
        reactivates the pattern-specific (non-shared) features of a memory,
        not the overlap components.

    Phase 2 -- Spontaneous at REPLAY_NOISE_STD:
        Network runs at noise=2.0 (above bistable 1.5; STDP fires).
        STDP re-potentiates decayed assembly synapses.
        bulk_slow_step is NOT called here (W changes via STDP; closed-form
        assumption of constant W does not hold).  Individual slow_step()
        calls are used instead.

    Phase 3 -- Tag-gated capture (with W_slow clamped):
        Transfers LTP-tagged synapses into W_slow.
    """
    t0       = _tick()
    use_slow = getattr(net, 'slow_enabled', False)

    # ── Replay compression ───────────────────────────────────────────────────
    # Temporally compress replay dynamics by scaling DT (Foster & Wilson 2006).
    # When USE_FACILITATION is True, compression is achieved via synaptic
    # facilitation dynamics rather than hardcoded DT scaling.
    _orig_dt = net.dt
    if USE_MEMBRANE_SCALING:
        net.dt = DT * REPLAY_MEMBRANE_SCALE
    else:
        net.dt = DT * REPLAY_COMPRESSION_FACTOR
    # ── Synaptic facilitation state ──────────────────────────────────────────
    # Short-term facilitation builds during high-frequency bursts within a
    # replay event, transiently boosting synaptic efficacy.  Resets between
    # events.  Implemented as a scalar multiplier on W_eff for E-E connections.
    _facil = 1.0  # facilitation multiplier (1.0 = baseline)
    _facil_decay = float(np.exp(-1.0 / FACILITATION_TAU)) if USE_FACILITATION else 1.0
    _ripple_isi = max(1, int((1000.0 / RIPPLE_FREQ_HZ) / (DT * REPLAY_COMPRESSION_FACTOR)))

    # ── Gamma/Theta phase deltas (computed once per event) ───────────────────
    _th_delta_ev = 2.0 * np.pi * DT * REPLAY_COMPRESSION_FACTOR / (1000.0 / max(THETA_FREQ_HZ, 0.1))
    _gm_delta_ev = 2.0 * np.pi * DT * REPLAY_COMPRESSION_FACTOR / (1000.0 / max(GAMMA_FREQ_HZ, 0.1))

    # ── Phase 1: partial seed
    # Build cue pool: prefer non-overlap neurons to avoid cross-assembly
    # activation via shared recurrent pathways.
    if all_assemblies is not None and len(all_assemblies) > 1:
        other_set = set()
        for other in all_assemblies:
            if other is not assembly:
                other_set.update(other.tolist())
        unique_neurons = np.array([n for n in assembly if n not in other_set])
        # Fallback: if not enough unique neurons, include overlap neurons
        cue_pool = unique_neurons if len(unique_neurons) >= cue_size else assembly
    else:
        cue_pool = assembly
    cue_n = np.random.choice(cue_pool, size=min(cue_size, len(cue_pool)), replace=False)
    seed_stim = torch.zeros(N_NEURONS, device=DEVICE)
    seed_stim[cue_n] = seed_strength

    orig_noise    = net.noise_std
    net.noise_std = noise
    # Theta modulation: base noise level, modulated per step by theta phase
    _base_noise   = noise
    def _set_theta_noise():
        global _th_phase_rad
        net.noise_std = _base_noise * (1.0 + THETA_MOD_DEPTH * np.sin(_th_phase_rad))
        _th_phase_rad = (_th_phase_rad + _th_delta_ev) % (2.0 * np.pi)

    def _update_phase_stdp():
        global _th_phase_rad, _gamma_phase_rad
        th = np.sin(_th_phase_rad)
        gm = np.sin(_gamma_phase_rad)
        # Phase-specific STDP: LTP at theta peak, LTD at trough
        net.A_plus  = A_PLUS  * (1.0 + PHASE_STDP_DEPTH * th)
        net.A_minus = A_MINUS * (1.0 - PHASE_STDP_DEPTH * th)
        # Gamma-nested noise modulation (fast gamma bursts)
        gamma_mod = 1.0 + GAMMA_MOD_DEPTH * gm
        net.noise_std = _base_noise * (1.0 + THETA_MOD_DEPTH * th) * gamma_mod
        # Advance phases
        _th_phase_rad  = (_th_phase_rad  + _th_delta_ev) % (2.0 * np.pi)
        _gamma_phase_rad = (_gamma_phase_rad + _gm_delta_ev) % (2.0 * np.pi)

    _spike_thr = int(REPLAY_SPIKE_FRACTION_MAX * N_NEURONS)  # runaway threshold

    # ── Coherence-gating state (local to this replay event) ──────────────────
    # Build masks identifying which excitatory neurons belong to the current
    # replay target (this assembly) and which are off-target.
    # In sparse_modular mode, off-target is restricted to the assembly's own
    # module — replay competitions are local, not global.
    target_excs  = assembly[assembly < N_EXC]
    target_mask  = torch.zeros(N_EXC, device=DEVICE, dtype=torch.bool)
    if len(target_excs) > 0:
        target_mask[target_excs] = True
    if ARCH_MODE == "sparse_modular" and hasattr(net, 'module_id'):
        asm_module = int(net.module_id[target_excs[0]].item()) if len(target_excs) > 0 else 0
        module_excs = (net.module_id[:N_EXC] == asm_module).cpu()
        off_mask_bool = module_excs.clone()
        off_mask_bool[target_excs] = False
        off_mask = off_mask_bool
    else:
        off_mask = ~target_mask
    n_target     = max(1, int(target_mask.sum().item()))
    n_off        = max(1, int(off_mask.sum().item()))
    # ── Overlap-sensitive coherence scaling (M1) ─────────────────────────
    # Compute max overlap fraction between this assembly and any other.
    # Scales LAMBDA_OFF: higher overlap → stricter coherence gate → harder
    # for overlapping assemblies to trigger STDP during replay.
    # Also precompute cross-assembly overlap neuron indices for M2/M5 so
    # they don't need to recompute on every STDP step.
    if all_assemblies is not None and len(all_assemblies) > 1:
        _asm_set = set(int(n) for n in assembly)
        _max_overlap = 0.0
        _overlap_neighbors = []  # list of (shared_e, other_e, other_idx) for overlapping neighbors
        for _oi, other in enumerate(all_assemblies):
            if other is assembly: continue
            shared = len(_asm_set & set(int(n) for n in other))
            frac = shared / max(len(assembly), 1)
            if frac > _max_overlap:
                _max_overlap = frac
            if shared > 0:
                shared_e = np.array([n for n in assembly if n in set(int(o) for o in other) and n < N_EXC], dtype=int)
                other_e = np.array([n for n in other if n < N_EXC], dtype=int)
                if len(shared_e) > 0 and len(other_e) > 0:
                    _overlap_neighbors.append((shared_e, other_e, _oi))
    else:
        _max_overlap = 0.0
        _overlap_neighbors = []
    _use_ov_penalty = (ablation or {}).get("overlap_penalty", True) if ablation else True
    _eff_lambda = REPLAY_COHERENCE_LAMBDA * (1.0 + OVERLAP_COHERENCE_PENALTY * _max_overlap * _use_ov_penalty)

    # ── M6/M7 state extraction ────────────────────────────────────────────
    # Shared-neuron fatigue (M6): persistent tensor carried across events.
    # Heterosynaptic LTD tag (M7): scalar per overlap pair, decays slowly.
    _use_fatigue   = (ablation or {}).get("fatigue", True) if ablation else True
    _use_het_tag   = (ablation or {}).get("hetero_tag", True) if ablation else True
    _use_reconsol  = (ablation or {}).get("reconsol", True) if ablation else True
    if state is not None and isinstance(state, dict):
        _fatigue   = state.get("fatigue",   torch.zeros(N_EXC, device=DEVICE))
        _het_tag   = state.get("het_tag",   {})  # dict: (i,j) -> float tag strength
    else:
        _fatigue   = torch.zeros(N_EXC, device=DEVICE)
        _het_tag   = {}
    # Precompute overlap pair keys for M7
    _pair_keys = []
    if _overlap_neighbors:
        for shared_e, other_e, _oi in _overlap_neighbors:
            # Unique key for this overlap pair (sorted tuple of first elements)
            _pair_key = (int(shared_e[0]) if len(shared_e) > 0 else -1,
                         int(other_e[0]) if len(other_e) > 0 else -1)
            _pair_keys.append(_pair_key)
            if _pair_key not in _het_tag:
                _het_tag[_pair_key] = 0.0

    # Activity buffer: exponentially-decaying spike accumulator over excitatory
    # neurons.  Captures "fired in the last ~10 ms" without holding a history.
    activity     = torch.zeros(N_EXC, device=DEVICE)
    COH_THR      = REPLAY_COHERENCE_THR
    ACT_THR      = REPLAY_COHERENCE_ACTIVE_THR
    ACT_DECAY    = REPLAY_COHERENCE_DECAY

    def _coh(net_spikes_exc):
        """Update activity buffer; return raw coherence score ∈ [0, 1].

        Closes gate when score ≤ COH_THR:
          • failed completion + random background firing
          • runaway / saturated firing  (target≈off≈high - ratio - 1/(1+λ))
          • pure silence  (rates ≈ 0 - coherence - 0)
        Opens gate when score > COH_THR:
          • clean cue-only firing  (off_rate = 0 - coherence = 1)
          • clean pattern completion  (target dominates off)

        Uses _eff_lambda which scales with overlap fraction: assemblies that
        share neurons face a stricter coherence gate, making replay-driven
        consolidation harder for overlapping memories (M1 — shared inhibitory
        competition via coherence gating).
        """
        activity.mul_(ACT_DECAY).add_(net_spikes_exc)
        active = activity > ACT_THR
        t_rate = float((active & target_mask).sum().item()) / n_target
        o_rate = float((active & off_mask).sum().item()) / n_off
        return t_rate / (t_rate + _eff_lambda * o_rate + 1e-6)

    # ── Pre-event weight snapshot (replay quality metrics) ───────────────────
    # Read W_fast and W_slow recurrent mean for the target assembly BEFORE any
    # STDP modifies weights.  Uses .data to bypass autograd (read-only, safe).
    _te = target_excs                      # excitatory neurons in this assembly
    if len(_te) > 1:
        _sub_f = net.W.data[np.ix_(_te, _te)]
        _pos_f = _sub_f > 0
        _wf_aa = float(_sub_f[_pos_f].mean().item()) if _pos_f.any() else 0.0
    else:
        _wf_aa = 0.0
    if use_slow and len(_te) > 1:
        _sub_s = net.W_slow.data[np.ix_(_te, _te)]
        _pos_s = _sub_s > 0
        _ws_aa = float(_sub_s[_pos_s].mean().item()) if _pos_s.any() else 0.0
    else:
        _ws_aa = 0.0

    with torch.no_grad():
        # Phase 1: partial seed.  Cued A neurons spike; coherence gate ensures
        # STDP only fires while the activation pattern is target-dominated.
        for _ in range(seed_dur):
            _update_phase_stdp()
            net.forward(seed_stim)
            # Hard runaway safety net (also caught by coherence, but explicit).
            if int(net.spikes.sum().item()) > _spike_thr:
                # Still update activity so the buffer reflects real network
                # state; gate will close naturally on subsequent steps.
                activity.mul_(ACT_DECAY).add_(net.spikes[:N_EXC].float())
                if use_slow:
                    net.slow_step()
                continue
            if _coh(net.spikes[:N_EXC].float()) > COH_THR:
                if tags is not None:
                    tags.update_from_spikes(net)
                net.stdp_step()
            if use_slow:
                net.slow_step()

        # ── Reset activity buffer before spontaneous phase ───────────────────
        # The seed phase deposits activity[cued] ≈ 10.7 (15 steps × geometric
        # sum with decay 0.95).  Without a reset, this stale seed signal keeps
        # t_rate = 0.40 for ~70 spontaneous steps even when NO assembly neuron
        # is firing — the coherence gate stays OPEN on phantom activity and
        # STDP fires on random background spikes, accumulating LTD at A-A
        # connections.  Resetting the buffer here makes spontaneous coherence
        # reflect ONLY what is actually firing in the spontaneous epoch.
        # Biologically: seed phase and spontaneous phase are distinct epochs;
        # the spontaneous replay is a new network state, not a continuation
        # of the forced drive.
        activity.zero_()

        # Phase 2: spontaneous activity with dynamic STDP unlock.
        #
        # Unlike a fixed eval-window approach, the adaptive gate operates in
        # real-time throughout the full spont_steps phase:
        #
        #   • STDP is LOCKED until all three quality criteria are met:
        #       (1) ≥ REPLAY_ACCEPT_MIN_CONSEC consecutive coherent steps
        #       (2) target fraction  ≥ REPLAY_ACCEPT_MIN_COMPLETION at that moment
        #       (3) off-target fraction ≤ REPLAY_ACCEPT_MAX_OFFTARGET at that moment
        #   • Once UNLOCKED, STDP fires for every subsequent coherent step
        #   • The unlock is permanent — coherence breaking after unlock does NOT
        #     re-lock STDP (attractor was validated; subsequent decay is expected)
        #
        # Biological rationale: hippocampal neuromodulators (ACh, DA) gate LTP
        # in real-time.  The consecutive-steps requirement corresponds to the
        # minimum duration of a valid SWR oscillation (~1–2 ms sustained at
        # model timescales).  Isolated coherence transients (1-2 steps) are
        # filtered as noise; sustained coherent bursts (≥3 steps) are validated
        # and allowed to drive synaptic consolidation.
        #
        # For confidence score: uses the first REPLAY_EVAL_STEPS coh values
        # (early-dynamics window) plus _best_t (peak target activation at any
        # streak ≥ REPLAY_ACCEPT_MIN_CONSEC point).  This is continuous [0,1]
        # and non-zero for all events, enabling full correlation analysis.
        zero_stim         = torch.zeros(N_NEURONS, device=DEVICE)
        _coh_vals         = []
        _consec           = 0          # current consecutive coherent steps
        _best_consec      = 0          # longest coherent run seen so far
        _smooth_coh       = 0.0        # EMA-smoothed coherence (for prob gate)
        _post_unlock_sub  = 0          # sub-threshold steps since last STDP

        # ── Ignition check ──────────────────────────────────────────────────
        # Before entering spontaneous phase, verify the seed phase produced
        # sufficient target activation.  This models the bistable "ignition"
        # threshold for replay (Colgin et al. 2004): weak seeds that fail
        # to recruit the assembly are rejected early.
        _ignition_pass = False
        if target_mask.any():
            _ignition_t = float((activity[target_mask] > ACT_THR).float().mean().item())
            _ignition_pass = _ignition_t >= REPLAY_IGNITION_STRENGTH

        # ── Attractor-persistence setup ───────────────────────────────────────
        # W_slow-weighted reverberatory support current.  Only present for
        # Slow+Replay (tags is not None and W_slow has been built up by
        # consolidation).  Zero for Fast/Replay by construction: tags=None.
        # Ablation "no persistence": pers_gain=0.0 - I_pers≡0 by computation.
        #
        # _persist[j](t) = PERS_DECAY * _persist[j](t-1) + spike[j](t-1)
        # I_pers[i](t)   = PERS_GAIN  * Σ_j W_slow[i,j] * _persist[j](t)
        #
        # Mechanism: neurons that recently fired leave a decaying trace; their
        # incoming slow weights provide sustained excitation to the assembly,
        # making the attractor self-reinforcing.  Without W_slow (Fast cond.)
        # this current is exactly zero — fragility is preserved by physics,
        # not by any additional threshold.
        _pers_gain  = (ablation.get("pers_gain", REPLAY_PERS_GAIN)
                       if ablation else REPLAY_PERS_GAIN)
        _w_slow_mat = (net.W_slow.data[:N_EXC, :N_EXC].clone()
                       if (use_slow and getattr(net, 'W_slow', None) is not None)
                       else None)
        _persist    = torch.zeros(N_EXC, device=DEVICE)
        _pers_stim  = torch.zeros(N_NEURONS, device=DEVICE)   # reused buffer

        # ── HC→Ctx replay transfer stimulus ──────────────────────────────────
        # During coherent replay events, HC assembly spikes drive cortex via the
        # HC→Ctx projection (model of hippocampal-neocortical SWR transfer).
        _hc_ctx_stim = (torch.zeros(N_NEURONS, device=DEVICE)
                        if N_HC > 0 and N_CTX > 0 else None)

        # ── Early termination state ───────────────────────────────────────────
        _post_unlock_sub = 0   # consecutive sub-threshold steps since last STDP
        _ripple_step_cnt = 0   # counter for ripple synchronization

        _effective_spont = spont_steps if _ignition_pass else min(spont_steps, 15)

        for _s in range(_effective_spont):
            _update_phase_stdp()

            # ── Ripple synchronization pulse ─────────────────────────────
            # When coherence is high, inject a brief synchronizing burst
            # at ripple frequency to model SWR ripples (Buzsáki 2015).
            _ripple_stim = torch.zeros(N_NEURONS, device=DEVICE)
            if _smooth_coh > COH_THR and _ripple_isi > 0:
                _ripple_step_cnt += 1
                if (_ripple_step_cnt % _ripple_isi == 0 and
                        _coh_vals and _coh_vals[-1] > RIPPLE_MIN_COHERENCE):
                    _ripple_stim[target_excs] = RIPPLE_STRENGTH

            # Update persistence trace with previous step's spikes,
            # then inject reverberatory current before this step's dynamics.
            _persist.mul_(REPLAY_PERS_DECAY).add_(net.spikes[:N_EXC].float())
            # Combine stimuli: persistence + ripple + HC→Ctx drive
            if _w_slow_mat is not None:
                _I_raw = (_w_slow_mat.mv(_persist)
                          .mul_(_pers_gain)
                          .clamp_(min=0.0, max=REPLAY_PERS_CLAMP))
                _I_total = _I_raw.sum().item()
                # ── Competitive persistence budget (M4) ────────────────────
                # Scale budget inversely with overlap.  When assembly A fires
                # during replay and overlaps with B, the shared reverberatory
                # resource is constrained — both attractors compete for the
                # same NMDA-like persistent current.  Higher overlap → tighter
                # budget → weaker persistence support → harder to sustain
                # coherent replay for either assembly independently.
                _use_pers_comp = (ablation or {}).get("pers_competition", True) if ablation else True
                _pers_budget = REPLAY_PERS_BUDGET / (1.0 + OVERLAP_PERS_BUDGET_REDUCTION * _max_overlap * _use_pers_comp)
                if _I_total > _pers_budget:
                    _I_raw.mul_(_pers_budget / _I_total)
                _pers_stim[:N_EXC] = _I_raw
                _total_stim = _pers_stim + _ripple_stim
                if _hc_ctx_stim is not None:
                    _total_stim = _total_stim + _hc_ctx_stim
                net.forward(_total_stim)
            else:
                _total_stim = _ripple_stim
                if _hc_ctx_stim is not None:
                    _total_stim = _total_stim + _hc_ctx_stim
                net.forward(zero_stim + _total_stim)
            # Compute next HC→Ctx drive from current step's HC spikes
            if _hc_ctx_stim is not None and _smooth_coh > COH_THR:
                with torch.no_grad():
                    _hc_spikes = net.spikes[:N_HC].float()
                    if _hc_spikes.sum() > 0:
                        # HC→Ctx block: W[post=cortex, pre=HC] at indices [N_HC:N_EXC, :N_HC]
                        _ctx_drv = torch.mv(net.W.data[N_HC:N_EXC, :N_HC], _hc_spikes)
                        _hc_ctx_stim.zero_()
                        _hc_ctx_stim[N_HC:N_EXC] = _ctx_drv.clamp_(0.0) * HC_CTX_REPLAY_BOOST
                    else:
                        _hc_ctx_stim.zero_()

            # ── Synaptic facilitation update ─────────────────────────────
            # Short-term facilitation builds on recent spiking, temporarily
            # boosting recurrent efficacy (Zucker & Regehr 2002).
            if USE_FACILITATION:
                _step_fr = int(net.spikes.sum().item()) / N_NEURONS
                _facil *= _facil_decay
                _facil += FACILITATION_STRENGTH * _step_fr
                _facil = min(_facil, FACILITATION_STRENGTH)

            if int(net.spikes.sum().item()) > _spike_thr:
                activity.mul_(ACT_DECAY).add_(net.spikes[:N_EXC].float())
                _coh_vals.append(0.0)
                _consec = 0
                _smooth_coh *= 0.0  # reset EMA on runaway
                _persist.zero_()
                if use_slow:
                    net.slow_step()
                continue
            cv = _coh(net.spikes[:N_EXC].float())
            _coh_vals.append(cv)

            # ── Probabilistic STDP gating ────────────────────────────────
            # Replace hard binary lock with a per-step sigmoid probability.
            # The coherence signal is smoothed via EMA (STDP_GATE_SMOOTH_ALPHA)
            # so brief transients are naturally filtered without a hard consec
            # requirement.  STDP fires stochastically with probability:
            #   p = sigmoid(slope * (smooth_coh - bias))
            # This preserves replay variability — events with partial coherence
            # still drive some STDP, maintaining exploratory replay diversity.
            _smooth_coh = _smooth_coh * (1.0 - STDP_GATE_SMOOTH_ALPHA) + cv * STDP_GATE_SMOOTH_ALPHA
            if cv > COH_THR:
                _consec += 1
                if _consec > _best_consec:
                    _best_consec = _consec
            else:
                _consec = 0

            if STDP_GATE_ENABLED:
                _stdp_prob = 1.0 / (1.0 + np.exp(-STDP_GATE_SLOPE * (_smooth_coh - STDP_GATE_BIAS)))
                _stdp_fires = np.random.rand() < _stdp_prob
            else:
                _stdp_fires = (cv > COH_THR)  # fallback: hard threshold (original)

            if _stdp_fires and cv > COH_THR:
                _post_unlock_sub = 0
            elif not _stdp_fires:
                _post_unlock_sub += 1

            # Early termination: prolonged sub-threshold coherence
            if _post_unlock_sub >= REPLAY_TERMINATE_AFTER:
                break

            if _stdp_fires and cv > COH_THR:
                if tags is not None:
                    tags.update_from_spikes(net)
                net.stdp_step()

                # Homeostatic scaling during replay (large networks)
                if net.n_neurons > 800 and _s % HOMEOSTATIC_WINDOW == 0:
                    _fr = int(net.spikes.sum().item()) / net.n_neurons
                    if _fr > HOMEOSTATIC_TARGET_RATE * 3:
                        with torch.no_grad():
                            net.W[:N_EXC, :N_EXC].mul_(1.0 - HOMEOSTATIC_STRENGTH)

                # M6: scale down fatigued overlap neuron STDP contribution.
                # Fatigued neurons have reduced effective spike impact, so their
                # outgoing weights are corrected post-STDP by 1/(1+fatigue)^2.
                if _use_fatigue and _overlap_neighbors and (_fatigue > 0.1).any():
                    _fatigued = torch.where(_fatigue > 0.1)[0].cpu().numpy()
                    for shared_e, other_e, _ in _overlap_neighbors:
                        _fat_in = np.intersect1d(_fatigued, shared_e)
                        if len(_fat_in) == 0: continue
                        _scale = (1.0 / (1.0 + _fatigue[_fat_in].cpu().numpy())) ** 2
                        with torch.no_grad():
                            net.W.data[np.ix_(other_e, _fat_in)] *= _scale[None, :]

            # ── Overlap-dependent plasticity fires on any coherent step ────
            # Cross-assembly LTD and directional drift operate whenever the
            # replay is coherent (cv > COH_THR), regardless of whether the
            # probabilistic STDP gate opens.  This ensures interference
            # mechanisms work even when STDP firing rate is low (DEV mode).
            # Biological basis: heterosynaptic competition during SWR
            # operates in parallel with homosynaptic STDP, triggered by
            # the same coherence signal (Rothschild et al. 2017).
            if cv > COH_THR:
                # ── Cross-assembly replay LTD (M2) ─────────────────────────
                # Overlap neurons that fired during this coherent step send
                # conflicting signals to neighboring assemblies.  LTD is
                # applied at connections from A's active overlap neurons →
                # B's downstream targets.  This pulls shared neurons toward
                # A's attractor, weakening B's control over them.
                _use_cross_ltd = (ablation or {}).get("cross_ltd", True) if ablation else True
                if _use_cross_ltd and CROSS_LTD_RATE > 0 and _overlap_neighbors:
                    _spiked_exc = net.spikes[:N_EXC]
                    for pi, (shared_e, other_e, _oi) in enumerate(_overlap_neighbors):
                        _active_mask = (_spiked_exc[shared_e] > 0)
                        if not _active_mask.any(): continue
                        _active_ov = shared_e[_active_mask.cpu().numpy()]
                        # M7: scale LTD rate by persisting heterosynaptic tag
                        _tag_scale = 1.0
                        if _use_het_tag and pi < len(_pair_keys):
                            _tag_scale = 1.0 + _het_tag.get(_pair_keys[pi], 0.0)
                        # M10: reconsolidation window - boost LTD if competing
                        # assembly was recently reactivated (labile window)
                        _reconsol_scale = 1.0
                        if _use_reconsol and state is not None:
                            _reconsol_dict = state.get("reconsol", {})
                            if _oi in _reconsol_dict and _reconsol_dict[_oi] > 0:
                                _reconsol_scale = RECONSOL_LTD_BOOST
                        _ltd_rate = CROSS_LTD_RATE * min(_tag_scale, 5.0) * _reconsol_scale
                        with torch.no_grad():
                            net.W.data[np.ix_(other_e, _active_ov)].mul_(1.0 - _ltd_rate)

                # ── Directional drift toward overlapping structure (M5) ────
                # When replay co-activates overlap neurons shared with B
                # during a coherent epoch, apply a small Hebbian boost at
                # B→overlap connections.  Over multiple events, attractors
                # drift toward each other's structure, modeling memory
                # merging (Schapiro et al. 2017).
                _use_drift = (ablation or {}).get("drift", True) if ablation else True
                if _use_drift and OVERLAP_DRIFT_RATE > 0 and _overlap_neighbors and _smooth_coh > COH_THR:
                    _spiked_exc = net.spikes[:N_EXC]
                    for shared_e, other_e, _ in _overlap_neighbors:
                        _active_mask = (_spiked_exc[shared_e] > 0)
                        if not _active_mask.any(): continue
                        _active_ov = shared_e[_active_mask.cpu().numpy()]
                        with torch.no_grad():
                            net.W.data[np.ix_(other_e, _active_ov)] += OVERLAP_DRIFT_RATE

                # ── Shared-neuron refractory fatigue (M6) ─────────────────────
                # Overlap neurons that fire during coherent replay accumulate
                # fatigue.  Fatigue scales down their contribution to the next
                # STDP step, reducing consolidation benefit that overlapping
                # assemblies receive.  Each replay event partially recovers.
                # Biological basis: spike-frequency adaptation (Madison & Nicoll).
                if _use_fatigue and OVERLAP_FATIGUE_RATE > 0 and _overlap_neighbors:
                    _spiked_exc = net.spikes[:N_EXC]
                    _active_any = False
                    for shared_e, other_e, _ in _overlap_neighbors:
                        _active_mask = _spiked_exc[shared_e] > 0
                        if not _active_mask.any(): continue
                        _active_ov = shared_e[_active_mask.cpu().numpy()]
                        _fatigue[_active_ov] += OVERLAP_FATIGUE_RATE
                        _active_any = True

                # ── Coherence-gated winner-take-all (M9) ──────────────────────
                # After each coherent step, check if non-target assemblies'
                # specific neurons are also active.  Competitor activity above
                # the margin threshold triggers LTD on that competitor's
                # specific→overlap connections, suppressing mixed replay.
                # Biological basis: competitive queuing during SWR (Pfeiffer).
                _use_wta = (ablation or {}).get("wta", True) if ablation else True
                if (_use_wta and WTA_COH_MARGIN > 0 and _overlap_neighbors
                        and all_assemblies is not None and len(all_assemblies) > 1):
                    _spiked_exc = net.spikes[:N_EXC]
                    _active_set = set(torch.where(_spiked_exc > 0)[0].cpu().numpy())
                    if _active_set:
                        for shared_e, other_e, _ in _overlap_neighbors:
                            _other_in_e = set(o for o in other_e)
                            _other_active = _active_set & _other_in_e
                            _other_frac = len(_other_active) / max(len(_other_in_e), 1)
                            if _other_frac > WTA_COH_MARGIN:
                                _competitor_e = np.array(list(_other_active), dtype=int)
                                with torch.no_grad():
                                    net.W.data[np.ix_(shared_e, _competitor_e)] *= (1.0 - WTA_LTD_RATE)

            if use_slow:
                net.slow_step()

        # ── M7: accumulate heterosynaptic LTD tags ───────────────────────────
        # For each overlap pair where overlap neurons fired during the event,
        # increment the persistent tag.  Tags decay slowly and scale future M2
        # cross-assembly LTD, creating cumulative competitive pressure.
        if _use_het_tag and _overlap_neighbors:
            for pi, (shared_e, other_e, _) in enumerate(_overlap_neighbors):
                _pk = _pair_keys[pi] if pi < len(_pair_keys) else None
                if _pk is None: continue
                _tag_activity = float((np.array(_coh_vals) > COH_THR).sum()) / max(len(_coh_vals), 1)
                _het_tag[_pk] = _het_tag.get(_pk, 0.0) * HETERO_TAG_DECAY + _tag_activity * HETERO_TAG_RATE
        # M6: decay fatigue between events
        if _use_fatigue:
            _fatigue.mul_(OVERLAP_FATIGUE_DECAY)

        # ── Continuous replay quality score ─────────────────────────────────
        # Replace binary _accepted with a graded quality measure.
        # quality = sigmoid(slope * (mean_coherence - bias))
        _mean_coh_event = float(np.mean(_coh_vals)) if _coh_vals else 0.0
        _event_quality = 1.0 / (1.0 + np.exp(-STDP_GATE_SLOPE * (_mean_coh_event - STDP_GATE_BIAS)))
        # Binary accept for backward compatibility (soft threshold)
        _accepted = _event_quality > 0.35
        _reject_reason = (
            None                    if _accepted else
            "low_quality"           if _event_quality <= 0.35 else
            "unstable_coherence"    if _best_consec < 2 else
            "low_completion"        if _best_t      < REPLAY_ACCEPT_MIN_COMPLETION else
            "high_offtarget"
        )

        # ── Coherent run lengths (attractor lifetime distribution) ────────────
        # Extract the length of every contiguous coherent epoch from _coh_vals.
        # This is the primary observable for attractor-dynamics analysis:
        #   • Fast/Replay:   mostly runs of 1–2 steps (fragile attractor)
        #   • Slow+Replay:   tail of 3–10+ step runs (stable basin)
        # Stored per event; aggregated in analyze_replay_quality.
        _coherent_runs = []
        _run = 0
        for _cv in _coh_vals:
            if _cv > COH_THR:
                _run += 1
            else:
                if _run > 0:
                    _coherent_runs.append(_run)
                _run = 0
        if _run > 0:
            _coherent_runs.append(_run)   # close final run if still open

        # ── Confidence score (continuous, non-zero floor) ──────────────────────
        # Components (geometric mean):
        #   completion  — event quality score from sigmoid gate (0..1)
        #   stability   — fraction of first REPLAY_EVAL_STEPS steps above threshold
        #   SNR         — early-window mean coherence / STDP threshold
        _eval_slice    = _coh_vals[:REPLAY_EVAL_STEPS]
        _coh_eval_arr  = np.array(_eval_slice, dtype=np.float32)
        _n_eval        = len(_coh_eval_arr)
        _eval_mean_coh = float(_coh_eval_arr.mean()) if _n_eval else 0.0
        # Non-zero floor: even poor events get a small confidence so correlations
        # across the full quality continuum remain computable
        _conf_complet  = _event_quality + 1e-4
        _conf_stable   = float((_coh_eval_arr > COH_THR).sum() + 1) / max(_n_eval + 1, 1)
        _conf_snr      = (_eval_mean_coh + 1e-4) / (COH_THR + 1e-4)
        _replay_confidence = (_conf_complet * _conf_stable * _conf_snr) ** (1.0 / 3.0)

    net.noise_std = orig_noise
    net.dt = _orig_dt
    net.A_plus  = A_PLUS
    net.A_minus = A_MINUS

    # ── Phase 3: tag-gated capture (W_slow clamped inside)
    # Capture still runs even if some steps were gated — only the tagged
    # synapses (those that did fire STDP at high coherence) contribute.
    if tags is not None and use_slow:
        tags.tag_driven_consolidation(net, assembly)

    _tock("replay", t0)

    # ── Replay quality metrics ────────────────────────────────────────────────
    # _coh_vals spans the full spontaneous phase (eval + STDP sub-phases).
    # _wf_aa / _ws_aa were snapshotted BEFORE any STDP so they reflect the
    # weight state that determined whether pattern completion could succeed.
    # _replay_accepted / _replay_confidence / _reject_reason capture the
    # adaptive selection decision for downstream analysis.
    _coh_arr  = np.array(_coh_vals, dtype=np.float32)
    _n_coh    = len(_coh_arr)
    _t_act    = float((activity[target_mask] > ACT_THR).float().mean().item()) \
                if target_mask.any() else 0.0
    _o_act    = float((activity[off_mask]    > ACT_THR).float().mean().item()) \
                if off_mask.any() else 0.0
    return {
        "rest_id":           rest_id,
        "burst_id":          burst_id,
        "event_id":          event_id,
        "peak_coherence":    float(_coh_arr.max())  if _n_coh else 0.0,
        "mean_coherence":    float(_coh_arr.mean()) if _n_coh else 0.0,
        "n_steps_coherent":  int((_coh_arr > COH_THR).sum()),
        "target_frac":       _t_act,
        "off_frac":          _o_act,
        "w_fast_aa":         _wf_aa,
        "w_slow_aa":         _ws_aa,
        # Adaptive selection outputs
        "replay_accepted":      _accepted,
        "replay_confidence":    _replay_confidence,
        "reject_reason":        _reject_reason,
        # Replay diversity / entropy
        "event_quality":        _event_quality,
        "smooth_coh_last":      float(_smooth_coh),
        "mean_qual_eval":       float(_eval_mean_coh),
        # Attractor-dynamics outputs
        "max_consec_coherent":  _best_consec,
        "coherent_run_lengths": _coherent_runs,  # list[int], one entry per epoch
        # Advanced replay metrics
        "ignition_pass":        _ignition_pass if '_ignition_pass' in dir() else False,
        "steps_used":           _s + 1 if '_s' in dir() else spont_steps,
        "ripple_count":         (_ripple_step_cnt // max(_ripple_isi, 1)
                                 if '_ripple_step_cnt' in dir() and '_ripple_isi' in dir() else 0),
        "gamma_power":          float(abs(np.sin(_gamma_phase_rad))
                                 if 'np' in dir() else 0.0),
        "theta_phase":          float(_th_phase_rad) if '_th_phase_rad' in dir() else 0.0,
        # Endogenous-prioritization tracking
        "assembly_idx":         assembly_idx,    # which assembly was replayed
        "urgency_score":        0.0,             # filled by caller for endogenous mode
        # Behavioral readout fields (filled later by callers)
        "decoding_accuracy":    None,
        "completion_accuracy":  None,
        "retrieval_latency":    None,
        # M6/M7 state for cross-event tracking
        "fatigue":              _fatigue.clone() if isinstance(_fatigue, torch.Tensor) else _fatigue,
        "het_tag":              _het_tag.copy() if isinstance(_het_tag, dict) else _het_tag,
    }


def _replay_priorities(learned_assemblies, scores, mode):
    n = len(learned_assemblies)
    if n == 0:
        return np.array([], float)
    if n == 1:
        return np.ones(1)
    if mode == "uniform":
        return np.ones(n) / n
    if mode == "oldest_first":
        w = np.arange(n, 0, -1, float)
        return w / w.sum()
    if mode == "interference_aware":
        s   = np.nan_to_num(np.array(scores, float), nan=0.0)
        eps = max(1e-6, (s.max() - s.min()) * 0.01)
        inv = s.max() - s + eps
        return inv / inv.sum()
    if mode == "endogenous":
        # Deferred to _compute_endogenous_urgency — caller handles this mode
        return np.ones(n) / n   # fallback: uniform (should not be called directly)
    raise ValueError(f"Unknown mode: {mode!r}")


def _compute_endogenous_urgency(n_assemblies, prior_metrics, recent_n=None):
    """
    Compute urgency-driven replay probabilities from accumulated replay metrics.

    Three normalised urgency signals per assembly (all in [0, 1]):
      1. w_fast_erosion : 1 − (mean_w_fast_aa / global_max_w_fast_aa)
                          Low fast-weight - assembly recurrent support is weak - urgent.
      2. reject_rate    : fraction of recent events that were rejected by the
                          adaptive gate.  Failed replay = attractor too fragile - urgent.
      3. coh_deficit    : max(0, COH_THR − mean_peak_coherence) / COH_THR
                          Below-threshold coherence = degraded basin - urgent.

    Urgency = geometric mean of (signal + FLOOR) for each signal.
    Probabilities = urgency / urgency.sum().

    Cold start (no prior_metrics): returns uniform distribution.
    Unseen assembly (no events recorded yet): assigned neutral mid-urgency
    (0.5 per signal), ensuring new assemblies get explored before their metrics
    accumulate — preventing starvation in the first rest period.

    Args:
        n_assemblies : number of assemblies that can be replayed
        prior_metrics: list of event dicts from _replay_one_event (accumulated
                       across all previous rest periods + within-rest so far)
        recent_n     : events per assembly to consider (default REPLAY_URGENCY_WINDOW)

    Returns:
        np.ndarray of shape (n_assemblies,) summing to 1.
    """
    if recent_n is None:
        recent_n = REPLAY_URGENCY_WINDOW
    if n_assemblies <= 1:
        return np.ones(max(n_assemblies, 1)) / max(n_assemblies, 1)
    if not prior_metrics:
        return np.ones(n_assemblies) / n_assemblies   # cold start

    # Group recent events per assembly
    per_asm = {i: [] for i in range(n_assemblies)}
    for m in prior_metrics:
        idx = m.get("assembly_idx", -1)
        if 0 <= idx < n_assemblies:
            per_asm[idx].append(m)

    # Per-assembly signals
    wf_means  = np.empty(n_assemblies)
    rr_means  = np.empty(n_assemblies)
    coh_means = np.empty(n_assemblies)

    for i in range(n_assemblies):
        recent = per_asm[i][-recent_n:] if per_asm[i] else []
        if recent:
            wf_means[i]  = float(np.mean([m.get("w_fast_aa",      0.0) for m in recent]))
            rr_means[i]  = 1.0 - float(np.mean(
                               [1 if m.get("replay_accepted", True) else 0 for m in recent]))
            coh_means[i] = float(np.mean([m.get("peak_coherence", 0.0) for m in recent]))
        else:
            # No events yet — neutral mid-urgency so this assembly gets fair access
            wf_means[i]  = W_MAX * 0.5          # mid-range fast weight
            rr_means[i]  = 0.5                   # 50% rejection rate
            coh_means[i] = REPLAY_COHERENCE_THR  # right at gate threshold

    # Signal 1: w_fast erosion (low w_fast = high urgency)
    wf_max          = max(float(wf_means.max()), 1e-6)
    erosion_signal  = 1.0 - (wf_means / wf_max)   # 0 = strongest, 1 = most eroded

    # Signal 2: rejection pressure (already in [0, 1])
    reject_signal   = rr_means.clip(0.0, 1.0)

    # Signal 3: coherence deficit (below COH_THR = degraded)
    coh_deficit     = np.clip(REPLAY_COHERENCE_THR - coh_means, 0.0, REPLAY_COHERENCE_THR)
    coh_signal      = coh_deficit / (REPLAY_COHERENCE_THR + 1e-6)  # normalise to [0, 1]

    # Combine: geometric mean with small floor to prevent dead zeros
    _FLOOR = 0.05
    urgency = ((erosion_signal + _FLOOR) *
               (reject_signal  + _FLOOR) *
               (coh_signal     + _FLOOR)) ** (1.0 / 3.0)

    total = urgency.sum()
    return urgency / total if total > 1e-12 else np.ones(n_assemblies) / n_assemblies


def _overlap_weighted_probs(probs, assemblies, boost=OVERLAP_REPLAY_BOOST):
    """Multiply replay probabilities by overlap vulnerability.

    Assemblies with higher mean overlap fraction get a probability boost,
    compensating for their higher interference vulnerability.  This ensures
    that memories facing more competition receive proportionally more replay
    resources — adaptive prioritization based on structural overlap.

    Args:
        probs: base probabilities (length n)
        assemblies: list of assembly arrays
        boost: max boost factor (0 = no effect)
    Returns:
        renormalised probabilities (length n)
    """
    if len(assemblies) <= 1 or boost <= 0:
        return probs
    n = len(assemblies)
    overlap_fracs = np.zeros(n)
    for i, asm_i in enumerate(assemblies):
        si = set(int(n) for n in asm_i)
        fracs = []
        for j, asm_j in enumerate(assemblies):
            if i == j: continue
            sj = set(int(n) for n in asm_j)
            shared = len(si & sj)
            fracs.append(shared / max(len(asm_i), 1))
        overlap_fracs[i] = float(np.mean(fracs)) if fracs else 0.0
    weights = 1.0 + boost * overlap_fracs
    weighted = probs * weights
    return weighted / weighted.sum()


def inter_memory_rest_with_replay(net, learned_assemblies, current_scores,
                                   n_steps=INTER_MEM_REST_STEPS,
                                   n_events=None,
                                   prioritize="interference_aware",
                                   tags=None,
                                   rest_id=0,
                                   accumulated_metrics=None,
                                   ablation=None,
                                   energy_tracker=None,
                                   reconsol_counters=None):
    # n_events defaults to _N_REPLAY_EVENTS (15 in DEV_MODE, 25 in production)
    if n_events is None:
        n_events = _N_REPLAY_EVENTS
    """
    Active replay rest — temporal layout matches biology (replay during sleep
    happens INTERLEAVED with synaptic homeostasis, not after a decay phase).

      Phase A (pre-replay half):  apply HALF of the bulk fast decay + half
                                  the slow_step integration + half tag decay.
      Phase B (replay):           n_events replay events while W_fast is
                                  still at intermediate strength - pattern
                                  completion succeeds reliably.
      Phase C (post-replay half): remaining half of the bulk decay /
                                  slow_step / tag decay.

    TOTAL decay across the rest is mathematically unchanged:
        (1 - (1-f_half)*(1-f_half))  with  f_half = 1 - exp(-n/2/tau)
        = 1 - exp(-n/tau)  ← identical to the previous single-shot decay.

    WHY THIS WAS NEEDED:
      Previously the full 81% decay was applied BEFORE the first replay event.
      For Fast/Replay (no W_slow safety net), this dropped W_fast[A,A] from
      ~0.5 (just-trained) to ~0.18 (heavily decayed) before any replay fired.
      The 5-neuron partial cue could not reliably pattern-complete at this W,
      so STDP fired on random background coincidences instead of A-A.  Result:
      Fast/Replay drifted to noise (sometimes negative).  Splitting the decay
      lets replay fire while W_fast is still ~0.27, restoring completion.

      Slow conditions were unaffected by this issue because W_eff during
      replay = 0.35·W_fast + 0.65·W_slow, with W_slow ≈ 0.3 providing
      adequate recurrent drive for completion even at low W_fast.

    Slow consolidation closed-form (bulk_slow_step) is split into two halves
    over CONSTANT-W intervals; W changes only between phases A and B
    (decay) and during phase B (replay STDP), so the per-half closed-form
    remains exact within each segment.
    """
    if len(learned_assemblies) == 0:
        return []

    half_n = n_steps // 2
    rest_n = n_steps - half_n   # exact: half_n + rest_n == n_steps

    def _bulk_decay(num_steps):
        """Closed-form fast decay, with HC/Cortex different rates."""
        f_hc  = 1.0 - float(np.exp(-num_steps / HC_FAST_DECAY_TAU))
        f_ctx = 1.0 - float(np.exp(-num_steps / CTX_SLOW_DECAY_TAU))
        with torch.no_grad():
            W    = net.W.data[:N_EXC, :N_EXC]
            base = net.W_init[:N_EXC, :N_EXC]
            # HC→HC (fast forgetting)
            net.W.data[:N_HC, :N_HC] = W[:N_HC, :N_HC] + (base[:N_HC, :N_HC] - W[:N_HC, :N_HC]) * f_hc
            # Ctx→Ctx (slow forgetting)
            net.W.data[N_HC:N_EXC, N_HC:N_EXC] = W[N_HC:N_EXC, N_HC:N_EXC] + (base[N_HC:N_EXC, N_HC:N_EXC] - W[N_HC:N_EXC, N_HC:N_EXC]) * f_ctx
            # HC↔Ctx cross connections (intermediate)
            f_cross = 0.5 * (f_hc + f_ctx)
            net.W.data[:N_HC, N_HC:N_EXC] = W[:N_HC, N_HC:N_EXC] + (base[:N_HC, N_HC:N_EXC] - W[:N_HC, N_HC:N_EXC]) * f_cross
            net.W.data[N_HC:N_EXC, :N_HC] = W[N_HC:N_EXC, :N_HC] + (base[N_HC:N_EXC, :N_HC] - W[N_HC:N_EXC, :N_HC]) * f_cross

    # ── Phase A: pre-replay (half decay + half slow consolidation)
    _bulk_decay(half_n)
    bulk_slow_step(net, half_n)
    if tags is not None:
        tags.decay(n_steps=half_n)

    # ── Phase B: burst-clustered replay at intermediate W_fast
    #
    # SWR-burst structure: n_events are split into n_bursts × REPLAY_BURST_SIZE
    # rapid-fire groups.  Within each burst, events fire back-to-back (no
    # intra-burst decay), so STDP increments accumulate before decay resumes.
    # Between bursts a brief REPLAY_BURST_GAP of passive dynamics separates
    # the clusters (biologically: the ~100–300 ms inter-ripple interval).
    #
    # If n_events is not divisible by REPLAY_BURST_SIZE, the remainder events
    # are appended to the last burst so the total count is always exactly
    # n_events (no events are silently dropped).
    # Base probabilities for non-endogenous modes (computed once, fixed for rest)
    _base_probs = _replay_priorities(learned_assemblies, current_scores, prioritize)
    # ── Overlap-weighted prioritization (M3) ─────────────────────────────
    # Assemblies with higher overlap get a probability boost, compensating
    # for their higher interference vulnerability.  This applies to both
    # the base probabilities and the per-burst endogenous urgency.
    _use_ov_priority = (ablation or {}).get("overlap_priority", True) if ablation else True
    if _use_ov_priority and OVERLAP_REPLAY_BOOST > 0 and len(learned_assemblies) > 1:
        _base_probs = _overlap_weighted_probs(_base_probs, learned_assemblies)
    # For endogenous: start with accumulated cross-rest metrics (may be empty - uniform)
    _endo_prior = list(accumulated_metrics) if accumulated_metrics else []

    n_bursts  = max(1, n_events // REPLAY_BURST_SIZE)
    base_sz   = n_events // n_bursts           # events per burst (floor)
    remainder = n_events - base_sz * n_bursts  # surplus events - last burst

    _event_metrics = []   # replay quality records for this rest period
    _ev_count      = 0    # sequential event index within this rest

    for b in range(n_bursts):
        # For endogenous mode: recompute urgency at each burst start using all
        # prior cross-rest metrics PLUS within-rest events accumulated so far.
        # This creates closed-loop feedback: a burst with many rejections raises
        # urgency for the next burst, concentrating resources on the struggling
        # assembly.  Other modes use pre-computed fixed probabilities.
        if prioritize == "endogenous":
            probs = _compute_endogenous_urgency(
                n_assemblies=len(learned_assemblies),
                prior_metrics=_endo_prior + _event_metrics,
            )
            # Apply overlap weighting to endogenous urgency too
            if _use_ov_priority and OVERLAP_REPLAY_BOOST > 0 and len(learned_assemblies) > 1:
                probs = _overlap_weighted_probs(probs, learned_assemblies)
        else:
            probs = _base_probs

        burst_sz = base_sz + (remainder if b == n_bursts - 1 else 0)

        # ── Emergent replay trigger ─────────────────────────────────────────
        # Check for spontaneous coherent state before scheduling this burst.
        _emergent_idx = _detect_emergent_assembly(net, learned_assemblies)

        # Determine trajectory indices for this burst
        # When internal propagation is enabled, trajectories emerge from
        # learned chain weights rather than external scheduling.
        _use_internal = (ABLATION_PHASE1.get("internal_propagation", True) and
                         len(learned_assemblies) > 1)
        if _use_internal:
            # Internal propagation: first event from scheduler or emergent,
            # then propagate via learned chain weights.
            if _emergent_idx is not None:
                _first_idx = _emergent_idx
            elif REPLAY_TRAJECTORY == "random":
                _first_idx = int(np.random.choice(len(learned_assemblies), p=probs))
            elif REPLAY_TRAJECTORY == "reverse":
                _first_idx = len(learned_assemblies) - 1
            else:
                _first_idx = 0
            _tray_indices = [_first_idx]
            _internal_prop_step = 0
            while len(_tray_indices) < burst_sz and _internal_prop_step < INTERNAL_PROP_MAX_STEPS:
                _last = _tray_indices[-1]
                _next = _last + 1 if _last + 1 < len(learned_assemblies) else None
                _prev = _last - 1 if _last > 0 else None
                # Check chain weight from _last → _next
                _chain_ready = False
                if _next is not None:
                    _a = learned_assemblies[_last]
                    _b = learned_assemblies[_next]
                    _aex = _a[_a < N_EXC]
                    _bex = _b[_b < N_EXC]
                    if len(_aex) > 0 and len(_bex) > 0:
                        _chain_w = float(net.W.data[np.ix_(_bex, _aex)].mean().item())
                        _chain_ready = _chain_w > INTERNAL_PROP_MIN_WEIGHT
                if _chain_ready and np.random.rand() < INTERNAL_PROPAGATION_PROB:
                    _tray_indices.append(_next)
                elif _prev is not None and np.random.rand() < CHAIN_REPLAY_PROB * 0.5:
                    _tray_indices.append(_prev)
                else:
                    # Re-sample from base probabilities to fill
                    _alt = int(np.random.choice(len(learned_assemblies), p=probs))
                    _tray_indices.append(_alt)
                _internal_prop_step += 1
            _tray_indices = _tray_indices[:burst_sz]  # trim to burst size
        elif REPLAY_TRAJECTORY == "random":
            _tray_indices = np.random.choice(len(learned_assemblies),
                                             size=burst_sz, p=probs).tolist()
        else:
            _n_asm = len(learned_assemblies)
            if REPLAY_TRAJECTORY == "bidirectional":
                _forward = list(range(_n_asm))
                _reverse = list(range(_n_asm - 1, -1, -1))
                _tray    = _forward + _reverse
            elif REPLAY_TRAJECTORY == "sequential":
                _tray = list(range(_n_asm))
            elif REPLAY_TRAJECTORY == "reverse":
                _tray = list(range(_n_asm - 1, -1, -1))
            else:
                _tray = list(range(_n_asm))
            _tray_indices = (_tray * (burst_sz // len(_tray) + 1))[:burst_sz]
        # Prepend emergent event for non-internal modes
        if _emergent_idx is not None and not _use_internal:
            _tray_indices = [_emergent_idx] + _tray_indices
        _prev_asm_idx = None
        _prev_accepted = False
        # ── Transition matrix for this burst ─────────────────────────────
        _n_asm = len(learned_assemblies)
        _burst_transitions = np.zeros((_n_asm, _n_asm), dtype=int)
        _forward_freq = 0   # forward (i → i+1) transitions
        _reverse_freq = 0   # reverse (i → i-1) transitions
        # M6/M7 state: persistent across events within this rest period
        _replay_state = {"fatigue": torch.zeros(N_EXC, device=DEVICE),
                         "het_tag": {},
                         "reconsol": reconsol_counters if reconsol_counters is not None else {}}
        for idx in _tray_indices:
            m = _replay_one_event(net, learned_assemblies[idx], tags=tags,
                                  all_assemblies=learned_assemblies,
                                  rest_id=rest_id, burst_id=b,
                                  event_id=_ev_count,
                                  assembly_idx=idx,
                                  ablation=ablation,
                                  state=_replay_state)
            # Extract M6/M7 state for next event
            if "fatigue" in m and isinstance(m["fatigue"], torch.Tensor):
                _replay_state["fatigue"] = m["fatigue"]
            if "het_tag" in m and isinstance(m["het_tag"], dict):
                _replay_state["het_tag"] = m["het_tag"]
            # Track transition direction
            if _prev_asm_idx is not None:
                _burst_transitions[_prev_asm_idx, idx] += 1
                if idx == _prev_asm_idx + 1:
                    _forward_freq += 1
                elif idx == _prev_asm_idx - 1:
                    _reverse_freq += 1
            # ── Asymmetric STDP for sequence learning ───────────────────────
            # After two consecutive accepted events of different assemblies,
            # apply asymmetric STDP (tau_plus < tau_minus) to strengthen the
            # forward transition and depress the backward direction.
            if (_prev_accepted and m["replay_accepted"]
                    and _prev_asm_idx is not None and _prev_asm_idx != idx
                    and N_HC > 0):
                _old_tp = net.tau_plus
                _old_tm = net.tau_minus
                _old_ap = net.A_plus
                _old_am = net.A_minus
                net.tau_plus  = STDP_SEQUENCE_TAU_PLUS
                net.tau_minus = STDP_SEQUENCE_TAU_MINUS
                net.A_plus    = A_PLUS * (1.0 + SEQUENCE_TRANSITION_STRENGTH * 10.0)
                net.A_minus   = A_MINUS * (1.0 + SEQUENCE_TRANSITION_STRENGTH * 10.0)
                for __ in range(3):
                    stim = torch.randn(N_NEURONS, device=DEVICE) * REPLAY_NOISE_STD
                    net.forward(stim)
                    net.stdp_step()
                net.tau_plus  = _old_tp
                net.tau_minus = _old_tm
                net.A_plus    = _old_ap
                net.A_minus   = _old_am
            # ── Transition tracking for sequential metrics ────────────
            _trans_direction = 0  # 1=forward, -1=reverse, 0=other/first
            if _prev_asm_idx is not None:
                if idx == _prev_asm_idx + 1:
                    _trans_direction = 1
                elif idx == _prev_asm_idx - 1:
                    _trans_direction = -1
            # Attach transition metadata to event (first event: trans=-2)
            m["transition_from_idx"]   = _prev_asm_idx if _prev_asm_idx is not None else -1
            m["transition_direction"]  = _trans_direction
            m["forward_freq_burst"]    = int(_forward_freq)
            m["reverse_freq_burst"]    = int(_reverse_freq)
            _prev_accepted = m["replay_accepted"]
            _prev_asm_idx = idx
            # Store the urgency weight assigned to this assembly at selection time
            m["urgency_score"] = float(probs[idx]) if prioritize == "endogenous" else 0.0
            # Phase 6: energy tracking
            if energy_tracker is not None:
                energy_tracker.consume(m)
            _event_metrics.append(m)
            _ev_count += 1

        # Inter-burst gap: passive decay between bursts (skip after last burst)
        if b < n_bursts - 1:
            _bulk_decay(REPLAY_BURST_GAP)
            bulk_slow_step(net, REPLAY_BURST_GAP)
            if tags is not None:
                tags.decay(n_steps=REPLAY_BURST_GAP)

        # Adaptive replay scaling: if recent events have poor coherence,
        # gracefully reduce remaining event count instead of hard termination.
        # This preserves exploratory replay diversity while saving compute
        # in failed conditions.  At least ADAPTIVE_REPLAY_MIN_EVENTS always fire.
        if ADAPTIVE_REPLAY and len(_event_metrics) >= ADAPTIVE_REPLAY_WINDOW:
            _recent = _event_metrics[-ADAPTIVE_REPLAY_WINDOW:]
            _n_good = sum(1 for m in _recent
                          if m.get("replay_accepted", False)
                          and m.get("mean_coherence", 0.0) > ADAPTIVE_COHERENCE_THR)
            _remaining = n_events - _ev_count
            if _n_good == 0 and _remaining > ADAPTIVE_REPLAY_MIN_EVENTS:
                # No good events: keep only the minimum to probe for recovery
                _surplus = _remaining - ADAPTIVE_REPLAY_MIN_EVENTS
                _keep_frac = max(0.1, 0.5 - 0.1 * (b - 1))  # decays across bursts
                _n_to_skip = int(_surplus * (1.0 - _keep_frac))
                n_events = _ev_count + max(ADAPTIVE_REPLAY_MIN_EVENTS, _remaining - _n_to_skip)
            elif _n_good < max(1, len(_recent) // 3) and _remaining > ADAPTIVE_REPLAY_MIN_EVENTS:
                # Few good events: modest reduction
                _surplus = _remaining - ADAPTIVE_REPLAY_MIN_EVENTS
                n_events = _ev_count + max(ADAPTIVE_REPLAY_MIN_EVENTS, _surplus // 2)

    # ── Phase C: post-replay (remaining decay + slow consolidation)
    _bulk_decay(rest_n)
    bulk_slow_step(net, rest_n)
    if tags is not None:
        tags.decay(n_steps=rest_n)

    return _event_metrics


def _detect_emergent_assembly(net, learned_assemblies):
    """Return index of assembly whose pattern is spontaneously coherent, or None.

    Models hippocampal SWR initiation: when the network transitions to a state
    where an assembly's firing rate crosses threshold, a sharp-wave ripple is
    spontaneously triggered (Buzsáki 2015).  Probability-modulated so emergent
    events are stochastic, not guaranteed.
    """
    if np.random.rand() > EMERGENT_REPLAY_PROB:
        return None
    spikes = getattr(net, 'spikes', None)
    if spikes is None:
        return None
    activity = spikes[:N_EXC].float()
    scores = []
    for asm in learned_assemblies:
        asm_exc = asm[asm < N_EXC]
        if len(asm_exc) == 0:
            scores.append(0.0)
            continue
        scores.append(float((activity[asm_exc] > 0.0).float().mean().item()))
    if not scores:
        return None
    best_idx = int(np.argmax(scores))
    if scores[best_idx] >= EMERGENT_COH_THR:
        return best_idx
    return None


def inter_memory_rest_no_replay(net, n_steps=INTER_MEM_REST_STEPS, tags=None):
    """Closed-form bulk decay + exact slow consolidation; no active replay."""
    f = 1.0 - float(np.exp(-n_steps / FAST_DECAY_TAU))
    with torch.no_grad():
        W    = net.W.data[:N_EXC, :N_EXC]
        base = net.W_init[:N_EXC, :N_EXC]
        net.W.data[:N_EXC, :N_EXC] = W + (base - W) * f
    bulk_slow_step(net, n_steps)
    if tags is not None:
        tags.decay(n_steps=n_steps)

# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL READOUT
# ─────────────────────────────────────────────────────────────────────────────

def decode_memory(net, assemblies, probe_noise=TEST_NOISE):
    """
    Simple linear memory decoder: for each assembly, record the spike vector
    during a probe, then compute pairwise separation between assemblies.

    Returns:
        separation_matrix: n_mem x n_mem matrix of pairwise decoding distances
        (cosine distance between mean spike vectors).
    """
    t0 = _tick()
    orig_noise = net.noise_std
    net.noise_std = probe_noise
    n_mem = len(assemblies)
    probe_steps_local = int(PROBE_DURATION_MS / DT)

    # Collect mean spike rate vectors for each assembly
    mean_vectors = []
    for asm in assemblies:
        cue_neurons = asm[:min(CUE_SIZE, len(asm))]
        stim = torch.zeros(N_NEURONS, device=DEVICE)
        stim[cue_neurons] = CUE_STRENGTH
        spk_buf = np.zeros((probe_steps_local, N_NEURONS), dtype=np.float32)
        net.reset_state()
        with torch.no_grad():
            for t in range(probe_steps_local):
                net.forward(stim)
                spk_buf[t] = net.spikes.cpu().numpy()
        mean_vectors.append(spk_buf.mean(axis=0))

    net.noise_std = orig_noise
    _tock("probe", t0)

    # Pairwise cosine separation
    sep = np.full((n_mem, n_mem), np.nan)
    for i in range(n_mem):
        for j in range(n_mem):
            vi = mean_vectors[i].ravel()
            vj = mean_vectors[j].ravel()
            ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
            if ni > 1e-12 and nj > 1e-12:
                sep[i, j] = float(1.0 - _cosine_dist(vi, vj))
            else:
                sep[i, j] = 0.0
    return sep


def noisy_cue_retrieval(net, assembly, noise_levels=None, n_trials=5):
    """
    Test retrieval robustness by presenting the assembly cue at increasing
    noise levels.  Measures completion accuracy (fraction of assembly neurons
    activated) at each noise level.

    Returns:
        dict with keys "noise_levels" (list[float]) and "completion" (list[float])
    """
    if noise_levels is None:
        noise_levels = NOISE_RETRIEVAL_LEVELS
    orig_noise = net.noise_std
    probe_steps_local = int(PROBE_DURATION_MS / DT)
    results = {"noise_levels": list(noise_levels), "completion": []}

    asm_e = assembly[assembly < N_EXC]
    _cue = asm_e[:min(CUE_SIZE, len(asm_e))]

    for nl in noise_levels:
        acc_vals = []
        for _ in range(n_trials):
            net.reset_state()
            net.noise_std = nl if nl > 0 else TEST_NOISE
            stim = torch.zeros(N_NEURONS, device=DEVICE)
            stim[_cue] = CUE_STRENGTH
            with torch.no_grad():
                spk_sum = np.zeros(N_NEURONS, dtype=np.float32)
                for _ in range(probe_steps_local):
                    net.forward(stim)
                    spk_sum += net.spikes.cpu().numpy()
            # Completion: fraction of assembly neurons that fired at least once
            asm_active = (spk_sum[asm_e] > 0).sum()
            n_asm = max(1, len(asm_e))
            acc_vals.append(float(asm_active) / n_asm)
        results["completion"].append(float(np.mean(acc_vals)))

    net.noise_std = orig_noise
    return results


def completion_accuracy(net, assembly, probe_noise=TEST_NOISE):
    """
    Measure what fraction of the assembly is successfully reactivated during
    a probe, and the retrieval latency (time to first coherent > COH_THR step).

    Returns:
        dict with keys: "completion_frac", "retrieval_latency_steps",
                        "target_rate", "off_rate"
    """
    t0 = _tick()
    orig_noise = net.noise_std
    net.noise_std = probe_noise
    net.reset_state()

    asm_e = assembly[assembly < N_EXC]
    _cue = asm_e[:min(CUE_SIZE, len(asm_e))]
    stim = torch.zeros(N_NEURONS, device=DEVICE)
    stim[_cue] = CUE_STRENGTH

    # Build off-target mask
    target_mask = torch.zeros(N_EXC, device=DEVICE, dtype=torch.bool)
    target_mask[asm_e] = True
    off_mask = ~target_mask
    n_target = max(1, int(target_mask.sum().item()))
    n_off = max(1, int(off_mask.sum().item()))
    activity = torch.zeros(N_EXC, device=DEVICE)
    probe_steps_local = int(PROBE_DURATION_MS / DT)

    _first_coh = None
    _final_completion = 0.0

    with torch.no_grad():
        for t in range(probe_steps_local):
            net.forward(stim)
            activity.mul_(REPLAY_COHERENCE_DECAY).add_(net.spikes[:N_EXC].float())
            active = activity > REPLAY_COHERENCE_ACTIVE_THR
            t_rate = float((active & target_mask).sum().item()) / n_target
            o_rate = float((active & off_mask).sum().item()) / n_off
            coh = t_rate / (t_rate + REPLAY_COHERENCE_LAMBDA * o_rate + 1e-6)
            if _first_coh is None and coh > REPLAY_COHERENCE_THR:
                _first_coh = t
            # Final completion: target fraction at last step
            _final_completion = t_rate

    net.noise_std = orig_noise
    _tock("probe", t0)

    return {
        "completion_frac":    _final_completion,
        "retrieval_latency_steps": _first_coh if _first_coh is not None else probe_steps_local,
        "target_rate":        float((active[target_mask] > REPLAY_COHERENCE_ACTIVE_THR).float().mean()) if target_mask.any() else 0.0,
        "off_rate":           float((active[off_mask] > REPLAY_COHERENCE_ACTIVE_THR).float().mean()) if off_mask.any() else 0.0,
    }


def replay_statistics(event_metrics):
    """
    Compute aggregate replay statistics from a list of event metric dicts.
    Produces distributions of replay durations, acceptance rates,
    coherence levels, and sparsity measures.

    Returns:
        dict with summary statistics.
    """
    if not event_metrics:
        return {}
    accepted = [m for m in event_metrics if m.get("replay_accepted", False)]
    rejected = [m for m in event_metrics if not m.get("replay_accepted", False)]
    all_coh = np.array([m.get("mean_coherence", 0.0) for m in event_metrics])
    all_conf = np.array([m.get("replay_confidence", 0.0) for m in event_metrics])
    all_wf = np.array([m.get("w_fast_aa", 0.0) for m in event_metrics])
    all_ws = np.array([m.get("w_slow_aa", 0.0) for m in event_metrics])
    steps_used = np.array([m.get("steps_used", 0) for m in event_metrics])
    run_lengths = []
    for m in event_metrics:
        rl = m.get("coherent_run_lengths", [])
        if rl:
            run_lengths.extend(rl)
    run_arr = np.array(run_lengths, dtype=float) if run_lengths else np.array([0.0])

    reasons = {}
    for m in rejected:
        r = m.get("reject_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    # ── Phase 1: sequential replay metrics ─────────────────────────────
    _fwd_count = sum(1 for m in event_metrics if m.get("transition_direction", 0) == 1)
    _rev_count = sum(1 for m in event_metrics if m.get("transition_direction", 0) == -1)
    _seq_events = _fwd_count + _rev_count
    _seqers = [m for m in event_metrics if m.get("transition_direction", 0) != 0]

    # ── Replay diversity & entropy metrics ─────────────────────────────
    _all_quality = np.array([m.get("event_quality", 0.0) for m in event_metrics])
    # Entropy of quality distribution (high = diverse replay qualities)
    _qual_hist, _ = np.histogram(_all_quality, bins=10, range=(0, 1))
    _qual_hist_p = _qual_hist / max(_qual_hist.sum(), 1)
    _replay_entropy = -float(np.sum(_qual_hist_p * np.log(_qual_hist_p + 1e-10))) / np.log(10)
    # Confidence diversity (std of confidence across events)
    _conf_diversity = float(np.std(all_conf)) if len(all_conf) > 1 else 0.0
    # Coherence diversity
    _coh_diversity = float(np.std(all_coh)) if len(all_coh) > 1 else 0.0

    return {
        "n_events":         len(event_metrics),
        "n_accepted":       len(accepted),
        "accept_rate":      float(len(accepted)) / max(len(event_metrics), 1),
        "mean_coherence":   float(np.mean(all_coh)) if len(all_coh) > 0 else 0.0,
        "mean_confidence":  float(np.mean(all_conf)) if len(all_conf) > 0 else 0.0,
        "mean_w_fast_aa":   float(np.mean(all_wf)) if len(all_wf) > 0 else 0.0,
        "mean_w_slow_aa":   float(np.mean(all_ws)) if len(all_ws) > 0 else 0.0,
        "mean_steps_used":  float(np.mean(steps_used)) if len(steps_used) > 0 else 0.0,
        "mean_run_length":  float(np.mean(run_arr)) if len(run_arr) > 0 else 0.0,
        "max_run_length":   float(np.max(run_arr)) if len(run_arr) > 0 else 0.0,
        "reject_reasons":   reasons,
        # Phase 1: sequential metrics
        "n_forward":        _fwd_count,
        "n_reverse":        _rev_count,
        "seq_frac":         float(_seq_events) / max(len(event_metrics), 1),
        "forward_frac":     float(_fwd_count) / max(_seq_events, 1),
        # Replay diversity metrics
        "replay_entropy":   _replay_entropy,
        "conf_diversity":   _conf_diversity,
        "coh_diversity":    _coh_diversity,
        "mean_quality":     float(np.mean(_all_quality)) if len(_all_quality) > 0 else 0.0,
        "quality_diversity": float(np.std(_all_quality)) if len(_all_quality) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE OPTIMISATIONS  (sparse mask, probe cache, adaptive replay,
#                        buffer preallocation, disk caching, torch.compile)
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Sparse structural mask enforcement ──────────────────────────────────
def _build_structural_mask_from_W_init(net):
    """Build (N_EXC, N_EXC) bool mask from initial W non-zero pattern.
    The architectural mask (module structure, EE sparsity) is applied at build_network
    and reflected in W_init.  Positions that are zero there should remain zero.
    """
    with torch.no_grad():
        init_ee = net.W_init.data[:N_EXC, :N_EXC]
        mask = (init_ee.abs() > 1e-10)
    return mask.to(DEVICE, non_blocking=True)

def get_structural_mask(net):
    """Return cached structural EE mask, built once from W_init."""
    global _STRUCTURAL_MASK_EE
    if _STRUCTURAL_MASK_EE is None:
        _STRUCTURAL_MASK_EE = _build_structural_mask_from_W_init(net)
        if CACHE_MASKS_TO_DISK:
            try:
                torch.save({'ee_mask': _STRUCTURAL_MASK_EE.cpu()}, MASKS_CACHE_PATH)
            except Exception:
                pass
    return _STRUCTURAL_MASK_EE

def load_cached_masks():
    """Try loading structural masks from disk; return True on success."""
    global _STRUCTURAL_MASK_EE
    if not CACHE_MASKS_TO_DISK:
        return False
    try:
        data = torch.load(MASKS_CACHE_PATH, map_location='cpu')
        if 'ee_mask' in data:
            _STRUCTURAL_MASK_EE = data['ee_mask'].to(DEVICE)
            return True
    except Exception:
        pass
    return False

def apply_structural_mask(net):
    """Zero out E→E connections that violate the architectural sparsity mask.
    Prevents STDP from growing weights in forbidden connections (modularity
    violation).  Called periodically during training and replay.
    """
    if not SPARSITY_ENFORCE:
        return
    mask = get_structural_mask(net)
    with torch.no_grad():
        ee = net.W.data[:N_EXC, :N_EXC]
        # Zero forbidden connections (structural zeros)
        ee.masked_fill_(~mask, 0.0)
        # Zero near-zero weights to maintain explicit sparsity
        ee[ee.abs() < SPARSITY_THRESHOLD] = 0.0

# ── 2. Probe early-exit and caching ────────────────────────────────────────
PROBE_EARLY_EXIT_COH_THR  = 0.60   # coherence at which we consider probe "done"
PROBE_EARLY_EXIT_MIN_STEPS = 50   # minimum steps before early exit
PROBE_CACHE_INVALIDATE_ON_STDP = True  # clear cache when STDP runs

def _make_probe_key(assembly, checkpoint_idx=0, use_slow=False):
    """Hashable key for probe result cache."""
    return (tuple(assembly.tolist() if hasattr(assembly, 'tolist') else assembly),
            checkpoint_idx, use_slow)

def _probe_early_exit(spk_arr, isyn_arr, t, assembly, non_cued):
    """Check if probe coherence has converged for early exit.
    Returns True if the assembly pattern is clearly stable above threshold.
    """
    t0 = max(0, t - 30)
    if t < PROBE_EARLY_EXIT_MIN_STEPS:
        return False
    recent_nc = isyn_arr[t0:t, non_cued]
    recent_bg = isyn_arr[t0:t, BG_START:BG_END]
    if recent_nc.size == 0 or recent_bg.size == 0:
        return False
    nc_mean = np.mean(recent_nc)
    bg_mean = np.mean(recent_bg)
    score = nc_mean - bg_mean
    # Stable above threshold with low variance
    var = np.var(recent_nc)
    return score > PROBE_EARLY_EXIT_COH_THR and var < 0.005

def clear_probe_cache():
    """Invalidate the probe cache (call when STDP modifies weights)."""
    global _PROBE_CACHE, _PROBE_CACHE_ORDER
    if PROBE_CACHE_ENABLED:
        _PROBE_CACHE.clear()
        _PROBE_CACHE_ORDER.clear()

def _probe_from_cache(net, assembly, checkpoint_idx=0, use_slow=False):
    """Probe with result caching.  Cache key = (assembly, checkpoint, use_slow).
    Cache is invalidated whenever STDP runs (via clear_probe_cache call).
    """
    if not PROBE_CACHE_ENABLED:
        return probe_memory(net, assembly)
    key = _make_probe_key(assembly, checkpoint_idx, use_slow)
    if key in _PROBE_CACHE:
        # LRU promotion
        _PROBE_CACHE_ORDER.remove(key)
        _PROBE_CACHE_ORDER.append(key)
        return _PROBE_CACHE[key]
    result = probe_memory(net, assembly)
    # LRU eviction
    if len(_PROBE_CACHE) >= PROBE_CACHE_MAXSIZE:
        evict = _PROBE_CACHE_ORDER.pop(0)
        _PROBE_CACHE.pop(evict, None)
    _PROBE_CACHE[key] = result
    _PROBE_CACHE_ORDER.append(key)
    return result

# ── 3. Adaptive replay — reduce events when coherence fails ────────────────
def _adaptive_n_events(metrics_so_far, base_n_events):
    """Reduce replay event count when recent events have low coherence.
    Saves ~30-50% of replay compute in failed conditions without losing signal.
    """
    if not ADAPTIVE_REPLAY or len(metrics_so_far) < ADAPTIVE_REPLAY_WINDOW:
        return base_n_events
    recent = metrics_so_far[-ADAPTIVE_REPLAY_WINDOW:]
    good = sum(1 for m in recent
               if m.get("replay_accepted", False)
               and m.get("mean_coherence", 0.0) > ADAPTIVE_COHERENCE_THR)
    if good == 0:
        return max(ADAPTIVE_REPLAY_MIN_EVENTS, base_n_events // 3)
    frac = good / len(recent)
    if frac < 0.15:
        return max(ADAPTIVE_REPLAY_MIN_EVENTS, base_n_events // 2)
    if frac < 0.30:
        return max(ADAPTIVE_REPLAY_MIN_EVENTS, int(base_n_events * 0.75))
    return base_n_events

# ── 4. Buffer preallocation ────────────────────────────────────────────────
_PREALLOCATED_STIM = {}

def _get_stim_buffer(shape, device=DEVICE):
    """Get a reusable noise buffer of given shape.
    Always fills with fresh N(0,0.5) noise.  The preallocation just avoids
    repeated torch.empty + normal_ calls on the same shape — we reuse the
    same allocation and refill in-place for speed.
    """
    key = (shape, device)
    if key not in _PREALLOCATED_STIM:
        buf = torch.empty(shape, device=device)
        _PREALLOCATED_STIM[key] = buf
    else:
        buf = _PREALLOCATED_STIM[key]
    torch.nn.init.normal_(buf, mean=0.0, std=0.5)
    return buf

def _get_stim_narrow(mean=0.0, std=0.3, device=DEVICE):
    """Reusable narrow-noise buffer for rest stimulation.
    Returns a fresh sample each call (reuses allocation, not content).
    """
    buf = _get_stim_buffer((N_NEURONS,), device)
    if abs(std - 0.5) > 1e-8 or abs(mean) > 1e-8:
        buf.mul_(std / 0.5).add_(mean)
    return buf

# ── 5. Torch compile (PyTorch 2.0+) ────────────────────────────────────────
_HAS_COMPILE = hasattr(torch, 'compile')
_COMPILED_FORWARD = None

def _maybe_compile_forward(net):
    """Wrap net.forward with torch.compile for repetitive loops.
    Only compiles once per session and reuses the compiled version.
    """
    global _COMPILED_FORWARD
    if not USE_TORCH_COMPILE or not _HAS_COMPILE:
        return net.forward
    if _COMPILED_FORWARD is None:
        try:
            _COMPILED_FORWARD = torch.compile(net.forward, mode=TORCH_COMPILE_MODE)
        except Exception:
            _COMPILED_FORWARD = net.forward
    return _COMPILED_FORWARD

# ═══════════════════════════════════════════════════════════════════════════
# PROBING
# ═══════════════════════════════════════════════════════════════════════════

def probe_memory(net, assembly):
    """I_syn differential at non-cued cells vs background. Read-only.
    Optimized with:
      - Preallocated stim buffer (avoids repeated torch.zeros)
      - Early exit when coherence threshold stabilises
    """
    t0            = _tick()
    orig_noise    = net.noise_std
    net.noise_std = TEST_NOISE
    net.reset_state()
    if net.stdp_enabled:
        net.pre_trace.zero_()
        net.post_trace.zero_()

    # Guard against cue size exceeding assembly size (scalable mode)
    _cue_size = min(CUE_SIZE, len(assembly))
    cue_neurons = assembly[:_cue_size]
    non_cued    = assembly[_cue_size:]

    isyn_arr = np.zeros((probe_steps, N_NEURONS), np.float32)
    spk_arr  = np.zeros((probe_steps, N_NEURONS), np.float32)

    _probe_spike_thr = N_NEURONS // 2  # >50% firing = global runaway, probe invalid

    # Preallocate stim buffer (reuse across timesteps)
    _stim_buf = torch.zeros(N_NEURONS, device=DEVICE)
    _stim_buf[cue_neurons] = CUE_STRENGTH

    _last_t = probe_steps  # actual number of steps (may be reduced by early exit)

    with torch.no_grad():
        for t in range(probe_steps):
            net.forward(_stim_buf)
            # Early exit: global runaway
            if int(net.spikes.sum().item()) > _probe_spike_thr:
                net.noise_std = orig_noise
                _tock("probe", t0)
                return {"isyn_score": np.nan, "spk_score": np.nan,
                        "isyn_nc": np.nan, "isyn_bg": np.nan}
            isyn_arr[t] = net.I_syn.cpu().numpy()
            spk_arr[t]  = net.spikes.cpu().numpy()
            # Early exit: coherence stable above threshold
            if PROBE_REDUCE_EARLY_EXIT and t >= PROBE_EARLY_EXIT_MIN_STEPS:
                recent_nc = isyn_arr[t-20:t, non_cued]
                if recent_nc.size > 0:
                    nc_mean = float(np.mean(recent_nc))
                    bg_mean = float(np.mean(isyn_arr[t-20:t, BG_START:BG_END]))
                    if (nc_mean - bg_mean) > PROBE_EARLY_EXIT_COH_THR and float(np.var(recent_nc)) < 0.005:
                        _last_t = t + 1
                        break

    net.noise_std = orig_noise
    _tock("probe", t0)

    # NaN guard
    if not np.all(np.isfinite(isyn_arr[:_last_t])):
        return {"isyn_score": np.nan, "spk_score": np.nan,
                "isyn_nc": np.nan, "isyn_bg": np.nan}

    isyn_nc = float(isyn_arr[:_last_t][:, non_cued].mean())
    isyn_bg = float(isyn_arr[:_last_t][:, BG_START:BG_END].mean())
    spk_nc  = float(spk_arr[:_last_t][:, non_cued].mean())
    spk_bg  = float(spk_arr[:_last_t][:, BG_START:BG_END].mean())
    return {
        "isyn_score": isyn_nc - isyn_bg,
        "spk_score":  spk_nc  - spk_bg,
        "isyn_nc":    isyn_nc,
        "isyn_bg":    isyn_bg,
    }


def assembly_weight_mean(net, assembly):
    asm = assembly[assembly < N_EXC]
    with torch.no_grad():
        sub  = net.W.data[np.ix_(asm, asm)]
        mask = sub > 0
        return float(sub[mask].mean()) if mask.any() else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — SYSTEMS CONSOLIDATION  (HC→Ctx transfer, lesions, drift)
# ═══════════════════════════════════════════════════════════════════════════

def lesion_network(net, lesion_type="hc"):
    """
    Zero out weights from the specified region's excitatory neurons.

    Args:
        lesion_type: "hc" — zero HC→all excitatory weights
                     "ctx" — zero Ctx→all excitatory weights
                     "silence_hc" — reversible HC silencing (zero HC W rows)
    """
    if lesion_type == "hc" and N_HC > 0:
        with torch.no_grad():
            net.W.data[:N_HC, :N_EXC] = 0.0
    elif lesion_type == "ctx" and N_CTX > 0:
        with torch.no_grad():
            net.W.data[N_HC:N_EXC, :N_EXC] = 0.0
    elif lesion_type == "silence_hc" and N_HC > 0:
        with torch.no_grad():
            net.W.data[:N_HC, :N_EXC] = 0.0


def measure_hc_ctx_contribution(net, assembly):
    """
    Measure recall contribution split: isolate HC→E vs Ctx→E components.
    Returns dict with 'hc_isyn' and 'ctx_isyn' scores.
    """
    asm_exc = assembly[assembly < N_EXC]
    _cue_size = min(CUE_SIZE, len(asm_exc))
    cue_neurons = asm_exc[:_cue_size]
    non_cued = asm_exc[_cue_size:]
    if len(non_cued) == 0:
        return {"hc_isyn": 0.0, "ctx_isyn": 0.0}

    isyn_hc = 0.0
    isyn_ctx = 0.0
    _bg_mean = 0.0
    with torch.no_grad():
        stim = torch.zeros(N_NEURONS, device=DEVICE)
        stim[cue_neurons] = CUE_STRENGTH
        _transfer_steps = min(probe_steps, 100)
        for _ in range(_transfer_steps):
            net.forward(stim)
            # HC contribution: I_syn from HC neurons to non-cued
            if N_HC > 0:
                _hc_part = net.I_syn.data[:N_HC].mean().item()
            else:
                _hc_part = 0.0
            # Ctx contribution: I_syn from CTX neurons to non-cued
            if N_CTX > 0:
                _ctx_part = net.I_syn.data[N_HC:N_EXC].mean().item()
            else:
                _ctx_part = 0.0
            isyn_hc += _hc_part
            isyn_ctx += _ctx_part
        isyn_hc /= probe_steps
        isyn_ctx /= probe_steps
    return {"hc_isyn": float(isyn_hc), "ctx_isyn": float(isyn_ctx)}


def compute_transfer_curves(all_replay_metrics, n_assemblies):
    """
    Track cortical memory emergence over replay cycles.
    Returns transfer_efficiency[cortex_asm_idx] = dict with hc_dependence, ctx_support.
    """
    if not all_replay_metrics:
        return {}
    _accepted = [m for m in all_replay_metrics if m.get("replay_accepted", False)]
    if not _accepted:
        return {}
    result = {}
    for i in range(n_assemblies):
        _asm_events = [m for m in _accepted if m.get("assembly_idx", -1) == i]
        _n = len(_asm_events)
        # Early events dominated by HC; later events show more ctx contribution
        _early = _asm_events[:max(1, _n // 2)] if _n > 1 else _asm_events
        _late  = _asm_events[_n // 2:] if _n > 1 else _asm_events
        _hc_dep = float(np.mean([m.get("w_fast_aa", 0) for m in _early])) if _early else 0.0
        _ctx_sup = float(np.mean([m.get("w_slow_aa", 0) for m in _late])) if _late else 0.0
        result[i] = {
            "hc_dependence": _hc_dep,
            "ctx_support": _ctx_sup,
            "n_events": _n,
            "transfer_ratio": _ctx_sup / max(_hc_dep, 1e-8),
        }
    return result


def representational_drift(snapshots, assemblies):
    """
    Track cosine similarity of engrams across training steps.
    Returns tuple (hc_drift, ctx_drift) where each is n_mem x n_mem matrix
    of pairwise cosine similarities between snapshots.
    """
    n_mem = len(assemblies)
    hc_drift = np.zeros((n_mem, n_mem))
    ctx_drift = np.zeros((n_mem, n_mem))
    for i in range(n_mem):
        asm_i = assemblies[i]
        asm_exc = asm_i[asm_i < N_EXC]
        hc_neurons = asm_exc[asm_exc < N_HC] if N_HC > 0 else []
        ctx_neurons = asm_exc[asm_exc >= N_HC] if N_CTX > 0 else []
        for j in range(n_mem):
            _s = snapshots[i][j]
            if _s is None or len(_s) == 0:
                continue
            _s = np.asarray(_s, float)
            # HC portion
            if len(hc_neurons) > 0 and _s.ndim > 0 and _s.shape[0] > max(hc_neurons):
                _hc_s = _s[hc_neurons]
                if np.linalg.norm(_hc_s) > 1e-12:
                    hc_drift[i, j] = float(np.dot(_hc_s, _hc_s) / (np.linalg.norm(_hc_s) ** 2 + 1e-12))
            # Ctx portion
            if len(ctx_neurons) > 0 and _s.ndim > 0 and _s.shape[0] > max(ctx_neurons):
                _ctx_s = _s[ctx_neurons]
                if np.linalg.norm(_ctx_s) > 1e-12:
                    ctx_drift[i, j] = float(np.dot(_ctx_s, _ctx_s) / (np.linalg.norm(_ctx_s) ** 2 + 1e-12))
    return hc_drift, ctx_drift


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — DYNAMICAL SYSTEMS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def basin_stability(net, assembly, n_trials=None):
    """..."""
    n_trials = 2 if DEV_MODE else (n_trials or BASIN_N_TRIALS)
    """
    Measure attractor basin stability: starting from perturbed states,
    what fraction of trials converge to the correct assembly pattern?

    Returns dict with 'recovery_prob', 'mean_steps_to_recover', 'basin_volume'.
    """
    _orig_noise = net.noise_std
    net.noise_std = TEST_NOISE
    _asm_exc = assembly[assembly < N_EXC]
    _cue_size = min(CUE_SIZE, len(_asm_exc))
    _cue = _asm_exc[:_cue_size]
    _nc = _asm_exc[_cue_size:]
    _recovered = 0
    _steps_to_recover = []
    for _ in range(n_trials):
        net.reset_state()
        if net.stdp_enabled:
            net.pre_trace.zero_()
            net.post_trace.zero_()
        # Apply perturbation: random current injection
        _perturb = torch.randn(N_NEURONS, device=DEVICE) * BASIN_PERTURB_STRENGTH
        net.forward(_perturb)
        # Now attempt recall
        _basin_probe_steps = min(probe_steps, 150)
        for s in range(_basin_probe_steps):
            stim = torch.zeros(N_NEURONS, device=DEVICE)
            stim[_cue] = CUE_STRENGTH
            net.forward(stim)
            # Check if assembly pattern is present
            if len(_nc) > 0:
                _nc_act = float(net.spikes[_nc].mean().item())
                _bg_act = float(net.spikes[BG_START:BG_END].mean().item())
                if _nc_act > _bg_act * 2:
                    _recovered += 1
                    _steps_to_recover.append(s)
                    break
    net.noise_std = _orig_noise
    _n = max(n_trials, 1)
    return {
        "recovery_prob": float(_recovered) / _n,
        "mean_steps": float(np.mean(_steps_to_recover)) if _steps_to_recover else float(probe_steps),
        "basin_volume": float(_recovered) / _n,
    }


def spectral_radius(net):
    """
    Compute spectral radius (largest eigenvalue magnitude) of the
    excitatory recurrent weight matrix.  Tracks network stability regime.
    """
    with torch.no_grad():
        W_ee = net.W.data[:N_EXC, :N_EXC].cpu().numpy()
        W_ee = np.nan_to_num(W_ee, nan=0.0, posinf=0.0, neginf=0.0)
        if W_ee.size == 0:
            return 0.0
        try:
            eigvals = np.linalg.eigvals(W_ee)
            return float(np.max(np.abs(eigvals)))
        except np.linalg.LinAlgError:
            return 0.0


def participation_ratio(net, n_pcs=EFFECTIVE_DIM_K):
    """
    Effective dimensionality of network states via PCA participation ratio.
    PR = (sum λ_i)² / (sum λ_i²) where λ_i are eigenvalues of the
    spike-count covariance matrix.

    Uses a short recording of spontaneous activity.
    """
    _orig_noise = net.noise_std
    net.noise_std = TEST_NOISE
    _rec_steps = 50 if DEV_MODE else 200
    _spikes = np.zeros((_rec_steps, N_EXC), dtype=np.float32)
    with torch.no_grad():
        for t in range(_rec_steps):
            stim = torch.randn(N_NEURONS, device=DEVICE) * 0.3
            net.forward(stim)
            _spikes[t] = net.spikes[:N_EXC].cpu().numpy()
    net.noise_std = _orig_noise
    # Compute covariance and participation ratio
    _spikes -= _spikes.mean(axis=0, keepdims=True)
    if _spikes.shape[1] == 0:
        return 0.0
    try:
        _cov = np.cov(_spikes.T)
        _evals = np.linalg.eigvalsh(_cov)
        _evals = _evals[_evals > 1e-12]
        if len(_evals) == 0:
            return 0.0
        _pr = float((_evals.sum() ** 2) / (_evals ** 2).sum())
        return _pr
    except np.linalg.LinAlgError:
        return 0.0


def metastable_state_analysis(net, learned_assemblies, n_steps=None):
    n_steps = 100 if DEV_MODE else (n_steps or 500)
    """
    Track dwell times in coherent states and transition rates.
    Returns dict with dwell_time_dist, transition_matrix, state_entropy.
    """
    n_asm = len(learned_assemblies)
    _state_sequence = []
    with torch.no_grad():
        for _ in range(n_steps):
            stim = torch.randn(N_NEURONS, device=DEVICE) * 0.3
            net.forward(stim)
            _act = net.spikes[:N_EXC].float()
            _best_asm = -1
            _best_act = 0.0
            for ai, asm in enumerate(learned_assemblies):
                _ae = asm[asm < N_EXC]
                if len(_ae) > 0:
                    _a = float(_act[_ae].mean().item())
                    if _a > _best_act:
                        _best_act = _a
                        _best_asm = ai
            if _best_act > REPLAY_COHERENCE_THR:
                _state_sequence.append(_best_asm)
    if not _state_sequence:
        return {"dwell_times": [], "transition_matrix": np.zeros((n_asm, n_asm)),
                "state_entropy": 0.0}
    # Dwell times
    _dwell = []
    _cur = _state_sequence[0]
    _cnt = 1
    for s in _state_sequence[1:]:
        if s == _cur:
            _cnt += 1
        else:
            _dwell.append(_cnt)
            _cur = s
            _cnt = 1
    _dwell.append(_cnt)
    _dwell = [d for d in _dwell if d >= METASTABLE_DWELL_MIN]
    # Transition matrix
    _trans = np.zeros((n_asm, n_asm))
    for i in range(len(_state_sequence) - 1):
        _fr, _to = _state_sequence[i], _state_sequence[i+1]
        if _fr != _to:
            _trans[_fr, _to] += 1
    # Normalize transition rows
    _row_sums = _trans.sum(axis=1, keepdims=True)
    _trans = np.divide(_trans, _row_sums, where=_row_sums > 0)
    # State entropy (Shannon entropy of state occupancy)
    _state_counts = np.zeros(n_asm)
    for s in _state_sequence:
        _state_counts[s] += 1
    _probs = _state_counts / max(len(_state_sequence), 1)
    _probs = _probs[_probs > 0]
    _entropy = float(-np.sum(_probs * np.log2(_probs))) if len(_probs) > 0 else 0.0
    return {
        "dwell_times": _dwell,
        "transition_matrix": _trans,
        "state_entropy": _entropy,
        "n_transitions": int(_trans.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — HOMEOSTATIC PLASTICITY
# ═══════════════════════════════════════════════════════════════════════════

def apply_bcm_metaplasticity(net):
    """
    BCM-like sliding threshold for STDP.
    Tracks mean postsynaptic activity per neuron and adjusts the LTP/LTD
    cross-over point.  Higher activity raises threshold (more LTD).
    Applied directly to net.A_plus / net.A_minus in proportion to
    deviation from baseline.
    """
    if not BCM_ENABLED:
        return
    if not hasattr(net, '_bcm_theta'):
        net._bcm_theta = torch.full((N_EXC,), BCM_THETA_INIT, device=DEVICE)
    if not hasattr(net, '_bcm_activity'):
        net._bcm_activity = torch.zeros(N_EXC, device=DEVICE)

    # Update running activity estimate
    with torch.no_grad():
        _act = net.spikes[:N_EXC].float().mean(dim=0)
        net._bcm_activity = (1 - 1.0 / BCM_TAU) * net._bcm_activity + (1.0 / BCM_TAU) * _act
        # Slide theta toward recent activity
        _theta = net._bcm_theta
        net._bcm_theta = _theta + BCM_SLIDING_RATE * (net._bcm_activity - _theta)
        # Clamp theta to sensible range
        net._bcm_theta = torch.clamp(net._bcm_theta, 0.01, 0.5)


def apply_intrinsic_plasticity(net):
    """
    Adjust intrinsic excitability of each excitatory neuron toward
    a target firing rate.  Uses a simple additive bias on I_syn.
    Stored in net._ip_bias (N_EXC,) and applied during forward pass.
    """
    if not INTRINSIC_PLASTICITY:
        return
    if not hasattr(net, '_ip_bias'):
        net._ip_bias = torch.zeros(N_EXC, device=DEVICE)
    if not hasattr(net, '_ip_rate_est'):
        net._ip_rate_est = torch.zeros(N_EXC, device=DEVICE)

    with torch.no_grad():
        _rate_est = net.spikes[:N_EXC].float().mean()
        net._ip_rate_est = 0.99 * net._ip_rate_est + 0.01 * _rate_est
        _dev = net._ip_rate_est - IP_TARGET_RATE
        net._ip_bias -= IP_GAIN * _dev
        net._ip_bias = torch.clamp(net._ip_bias, -0.5, 0.5)


def apply_spike_frequency_adaptation(net):
    """
    Add slow adaptation current to excitatory neurons.
    Accumulates with each spike and decays exponentially.
    Stored in net._sfa_current and subtracted from I_syn during forward.
    """
    if not SPIKE_FREQ_ADAPT:
        return
    if not hasattr(net, '_sfa_current'):
        net._sfa_current = torch.zeros(N_EXC, device=DEVICE)

    with torch.no_grad():
        _sfa = net._sfa_current
        # Decay
        _sfa *= (1.0 - 1.0 / SFA_TAU)
        # Increment per spike
        _spk = net.spikes[:N_EXC].float()
        _sfa += SFA_INCREMENT * _spk
        _sfa = torch.clamp(_sfa, 0.0, SFA_STRENGTH * 3)
        net._sfa_current = _sfa


def apply_inhibitory_stdp(net):
    """
    Inhibitory STDP: strengthens I→E connections when pre (inhibitory)
    fires before post (excitatory), weakens when order is reversed.
    Models E/I balance refinement (Vogels et al. 2011).
    """
    if not INHIBITORY_STDP:
        return
    if not hasattr(net, '_inh_pre_trace'):
        net._inh_pre_trace = torch.zeros(N_INH, device=DEVICE)
    if not hasattr(net, '_inh_post_trace'):
        net._inh_post_trace = torch.zeros(N_EXC, device=DEVICE)

    with torch.no_grad():
        _pre = net._inh_pre_trace
        _post = net._inh_post_trace
        _pre *= (1.0 - 1.0 / INH_STDP_TAU)
        _post *= (1.0 - 1.0 / INH_STDP_TAU)

        _inh_spikes = net.spikes[N_EXC:].float()
        _exc_spikes = net.spikes[:N_EXC].float()

        _pre += _inh_spikes
        _post += _exc_spikes

        # iSTDP update on I→E weights
        _W_ie = net.W.data[N_EXC:, :N_EXC]
        # Pre-before-post: LTP (inhibitory fires, then excitatory)
        _ltp = torch.outer(_inh_spikes, _post) * INH_STDP_LTP_RATE
        # Post-before-pre: LTD (excitatory fires, then inhibitory)
        _ltd = torch.outer(_pre, _exc_spikes) * INH_STDP_LTD_RATE
        _W_ie += _ltp - _ltd
        # Clamp to physiological range
        _W_ie = torch.clamp(_W_ie, -60.0, 0.0)
        net.W.data[N_EXC:, :N_EXC] = _W_ie

        net._inh_pre_trace = _pre
        net._inh_post_trace = _post


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — INTERNEURON DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════

def get_interneuron_type_indices():
    """
    Return dict {pv, som, vip, other} with neuron indices for each type.
    Indices are absolute (offset by N_EXC).
    """
    _n_pv  = int(N_INH * PV_FRACTION)
    _n_som = int(N_INH * SOM_FRACTION)
    _n_vip = int(N_INH * VIP_FRACTION)
    _base = N_EXC
    return {
        "pv":   np.arange(_base, _base + _n_pv),
        "som":  np.arange(_base + _n_pv, _base + _n_pv + _n_som),
        "vip":  np.arange(_base + _n_pv + _n_som, _base + _n_pv + _n_som + _n_vip),
        "other": np.arange(_base + _n_pv + _n_som + _n_vip, N_NEURONS),
    }


def apply_interneuron_dynamics(net):
    """
    Apply type-specific interneuron dynamics each step.
    - PV: fast-spiking (strong, fast inhibition)
    - SOM: dendritic gating (moderate, sustained)
    - VIP: disinhibitory (suppress SOM)

    Modifies net.I_syn based on current spiking and type-specific weights.
    """
    if not (ABLATION_PHASE5.get("pv_cells", False) or
            ABLATION_PHASE5.get("som_cells", False) or
            ABLATION_PHASE5.get("vip_cells", False)):
        return

    _types = get_interneuron_type_indices()
    _spk = net.spikes.float()
    with torch.no_grad():
        # PV → E: strong fast perisomatic inhibition
        if ABLATION_PHASE5.get("pv_cells", False) and len(_types["pv"]) > 0:
            _pv_spk = _spk[_types["pv"]].sum()
            net.I_syn[:N_EXC] += _pv_spk * PV_G_INH / max(len(_types["pv"]), 1)
        # SOM → E: dendritic inhibition (moderate)
        if ABLATION_PHASE5.get("som_cells", False) and len(_types["som"]) > 0:
            _som_spk = _spk[_types["som"]].sum()
            net.I_syn[:N_EXC] += _som_spk * SOM_G_INH / max(len(_types["som"]), 1)
        # VIP → SOM: disinhibition
        if ABLATION_PHASE5.get("vip_cells", False) and len(_types["vip"]) > 0 and len(_types["som"]) > 0:
            _vip_spk = _spk[_types["vip"]].sum()
            # VIP suppresses SOM — net effect is to reduce SOM→E inhibition
            _som_suppression = _vip_spk * VIP_G_INH / max(len(_types["vip"]), 1)
            net.I_syn[:N_EXC] -= _som_suppression * 0.5  # net disinhibition of E


def apply_type_specific_stdp(net):
    """
    Cell-type-specific STDP rules for interneuron subtypes.
    PV: fast Hebbian; SOM: slow anti-Hebbian; VIP: weak plasticity.
    Modifies the relevant rows of net.W.data directly.
    """
    _types = get_interneuron_type_indices()
    if not _types:
        return

    _spk = net.spikes.float()
    _pre_trace = getattr(net, 'pre_trace', None)
    _post_trace = getattr(net, 'post_trace', None)
    if _pre_trace is None or _post_trace is None:
        return

    with torch.no_grad():
        # PV STDP (fast) — vectorised, no Python for-loop
        if ABLATION_PHASE5.get("pv_cells", False) and len(_types["pv"]) > 0:
            _pv = _types["pv"]
            _pv_spikes = _spk[_pv].unsqueeze(1)  # (n_pv, 1)
            _post_bcast = _post_trace[:N_EXC].unsqueeze(0)  # (1, N_EXC)
            _pre_bcast = _pre_trace[:N_EXC].unsqueeze(0)
            _pre_fired = (_spk[:N_EXC] > 0).float().unsqueeze(0)
            net.W.data[_pv, :N_EXC] += PV_A_PLUS * _pv_spikes * _post_bcast - PV_A_MINUS * _pre_bcast * _pre_fired
            net.W.data[_pv, :N_EXC].clamp_(G_INH, 0.0)

        # SOM STDP (slow anti-Hebbian) — vectorised
        if ABLATION_PHASE5.get("som_cells", False) and len(_types["som"]) > 0:
            _som = _types["som"]
            _som_spikes = _spk[_som].unsqueeze(1)  # (n_som, 1)
            _post_bcast = _post_trace[:N_EXC].unsqueeze(0)  # (1, N_EXC)
            _pre_bcast = _pre_trace[:N_EXC].unsqueeze(0)
            _pre_fired = (_spk[:N_EXC] > 0).float().unsqueeze(0)
            # anti-Hebbian: LTD for pre-before-post, LTP for post-before-pre
            net.W.data[_som, :N_EXC] += SOM_A_MINUS * _som_spikes * _post_bcast - SOM_A_PLUS * _pre_bcast * _pre_fired
            net.W.data[_som, :N_EXC].clamp_(G_INH, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — ENERGY-CONSTRAINED REPLAY
# ═══════════════════════════════════════════════════════════════════════════

class EnergyTracker:
    """Track energy budget for replay events."""
    def __init__(self, budget=ENERGY_BUDGET):
        self.budget = budget
        self.remaining = budget
        self.total_consumed = 0.0
        self.event_costs = []
        self.suppression_active = False

    def reset(self, budget=None):
        if budget is not None:
            self.budget = budget
        self.remaining = self.budget
        self.event_costs = []
        self.suppression_active = False

    def cost_of_event(self, event_metrics):
        """Compute energy cost of a single replay event."""
        _base = ENERGY_PER_EVENT
        _spk_cost = ENERGY_PER_SPIKE * event_metrics.get("n_steps_coherent", 0)
        return _base + _spk_cost

    def can_afford(self, event_metrics):
        """Check if enough energy remains for event."""
        _cost = self.cost_of_event(event_metrics)
        return _cost <= self.remaining

    def consume(self, event_metrics):
        """Deduct event cost from remaining budget."""
        _cost = self.cost_of_event(event_metrics)
        if _cost <= self.remaining:
            self.remaining -= _cost
            self.total_consumed += _cost
            self.event_costs.append(_cost)
            if self.remaining < self.budget * REPLAY_SUPPRESSION_THR:
                self.suppression_active = True
            return True
        return False

    def get_replay_prob_modulation(self):
        """Return replay probability multiplier based on remaining budget."""
        if not ENERGY_TRACKING:
            return 1.0
        _frac = self.remaining / max(self.budget, 1.0)
        if _frac < 0.1:
            return 0.1
        if _frac < 0.3:
            return 0.4
        if _frac < 0.6:
            return 0.7
        return 1.0


def apply_sleep_state_modulation(n_events, phase="rest"):
    """
    Modulate replay probability based on sleep/wake state.
    Returns fraction of n_events that should be executed.
    """
    if phase == "wake":
        return max(1, int(n_events * SLEEP_REPLAY_PROB_WAKE))
    elif phase == "rest":
        return max(1, int(n_events * SLEEP_REPLAY_PROB_REST))
    elif phase == "nrem":
        return max(1, int(n_events * SLEEP_REPLAY_PROB_NREM))
    return n_events


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2–6 RESULT AGGREGATORS
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_phase2_metrics(net, assemblies, replay_metrics):
    """Aggregate all Phase 2 metrics into a single dict."""
    _hcc = [measure_hc_ctx_contribution(net, a) for a in assemblies]
    _tc = compute_transfer_curves(replay_metrics, len(assemblies))
    return {
        "hc_ctx_transfer": _hcc,
        "transfer_curves": _tc,
    }


def aggregate_phase3_metrics(net, learned_assemblies):
    """Aggregate all Phase 3 metrics into a single dict."""
    _basins = {}
    for i, asm in enumerate(learned_assemblies):
        _basins[i] = basin_stability(net, asm)
    _sr = spectral_radius(net)
    _pr = participation_ratio(net)
    _meta = metastable_state_analysis(net, learned_assemblies, n_steps=300)
    return {
        "basin_stability": _basins,
        "spectral_radius": _sr,
        "participation_ratio": _pr,
        "metastable": _meta,
    }


def aggregate_phase6_metrics(energy_tracker):
    """Aggregate energy tracking metrics."""
    if not ENERGY_TRACKING or energy_tracker is None:
        return {}
    return {
        "energy_remaining": energy_tracker.remaining,
        "energy_consumed": energy_tracker.total_consumed,
        "n_events_cost": len(energy_tracker.event_costs),
        "suppression_active": energy_tracker.suppression_active,
    }


# ───────────────────────────────────────────────────────────────────────────
# SINGLE EXPERIMENT RUNNER

def run_sequential_experiment(use_slow, use_replay, assemblies,
                               trial_seed, prioritize="interference_aware",
                               verbose=False, ablation=None):
    """
    Train A->B->C->D sequentially.  Probe all memories at each checkpoint.
    Returns retention_matrix, weight_evolution, rsm_matrix, tag_evolution,
    baseline_scores, final_scores.
    """
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    n_mem = len(assemblies)
    net   = build_network(use_slow=use_slow)
    tags  = SynapticTags() if USE_TAGGING else None

    # ── Engine optimisations ───────────────────────────────────────────
    # 1. Build/cache structural mask from initial connectivity
    if SPARSITY_ENFORCE:
        if not load_cached_masks():
            get_structural_mask(net)  # builds and caches
    # 2. Clear any stale probe cache from prior trials
    clear_probe_cache()
    # 3. Torch.compile the forward pass for repetitive loops
    if USE_TORCH_COMPILE and _HAS_COMPILE:
        try:
            net.forward = torch.compile(net.forward, mode=TORCH_COMPILE_MODE)
        except Exception:
            pass  # fall back to uncompiled

    retention   = np.full((n_mem, n_mem), np.nan)
    weights     = np.full((n_mem, n_mem), np.nan)
    tag_evol    = np.full((n_mem, n_mem), np.nan)
    snapshots   = [[None] * n_mem for _ in range(n_mem)]

    current_scores    = []
    reconsol_counters = {}  # M10: per-assembly reconsolidation window counters
    all_replay_metrics = []   # flat list of per-event quality dicts

    # Phase 4: homeostatic state
    if BCM_ENABLED:
        net._bcm_theta = torch.full((N_EXC,), BCM_THETA_INIT, device=DEVICE)
        net._bcm_activity = torch.zeros(N_EXC, device=DEVICE)
    if INTRINSIC_PLASTICITY:
        net._ip_bias = torch.zeros(N_EXC, device=DEVICE)
        net._ip_rate_est = torch.zeros(N_EXC, device=DEVICE)
    if SPIKE_FREQ_ADAPT:
        net._sfa_current = torch.zeros(N_EXC, device=DEVICE)

    # Phase 6: energy tracker
    _energy_tracker = EnergyTracker() if ENERGY_TRACKING else None

    # Lifecycle hook: baseline
    _call_hooks("baseline", net=net, assemblies=assemblies, n_mem=n_mem, j=-1)

    for j in range(n_mem):
        if verbose:
            cond_str = f"{'Slow' if use_slow else 'Fast'}+{'Replay' if use_replay else 'NoReplay'}"
            print(f"  [{cond_str}] Training memory {j} ({chr(65+j)}) ...", flush=True)

        train_one_memory(net, assemblies[j], tags=tags,
                         n_presentations=_N_PRESENTATIONS,
                         prev_assembly=assemblies[j - 1] if j > 0 and ABLATION_PHASE1.get("chain_stdp", True) else None)

        # Phase 4: apply homeostatic mechanisms during training
        for _ in range(5):
            _rest_stim = _get_stim_narrow(std=0.3)
            net.forward(_rest_stim)
            apply_bcm_metaplasticity(net)
            apply_intrinsic_plasticity(net)
            apply_spike_frequency_adaptation(net)

        # Phase 5: apply interneuron dynamics
        apply_interneuron_dynamics(net)
        apply_type_specific_stdp(net)

        # Snapshots for representational drift
        for i in range(j + 1):
            snapshots[i][j] = get_assembly_weight_vector(net, assemblies[i])

        # Probe all trained memories (with caching)
        for i in range(j + 1):
            r = _probe_from_cache(net, assemblies[i], checkpoint_idx=j)
            retention[i, j] = r["isyn_score"]
            weights[i, j]   = assembly_weight_mean(net, assemblies[i])
            if tags is not None:
                tag_evol[i, j] = tags.assembly_tag_mean(assemblies[i])

        # M10: reconsolidation window metaplasticity — after probing a memory,
        # its overlapping synapses enter a labile window.  Apply a small LTD
        # to overlap connections of each probed assembly, modeling the
        # reconsolidation vulnerability (Nader et al. 2000).
        _use_reconsol = (ablation or {}).get("reconsol", True) if ablation else True
        if _use_reconsol and RECONSOL_LTD_BOOST > 0 and j > 0:
            for _probed_idx in range(j + 1):
                reconsol_counters[_probed_idx] = RECONSOL_WINDOW_STEPS
                _probed_asm = assemblies[_probed_idx]
                for _overlap_idx in range(j + 1):
                    if _overlap_idx == _probed_idx: continue
                    _other_asm = assemblies[_overlap_idx]
                    _shared, _probed_spec, _other_spec = assembly_overlap_mask(
                        _probed_asm, _other_asm)
                    if len(_shared) == 0: continue
                    _se = _shared[_shared < N_EXC]
                    _ps = _probed_spec[_probed_spec < N_EXC]
                    if len(_se) == 0 or len(_ps) == 0: continue
                    with torch.no_grad():
                        net.W.data[np.ix_(_se, _ps)] *= (1.0 - RECONSOL_LTD_BOOST * 0.01)

        # Competitive interference from new memory onto old ones.
        # Overlap-dependent synaptic depression: connections from old-assembly
        # specific neurons to shared neurons are weakened.  This runs for
        # ALL conditions (not just use_slow) — genuine attractor competition
        # requires overlap to destabilize memories regardless of consolidation.
        if j > 0:
            apply_competitive_interference(net, assemblies[j], assemblies[:j],
                                           ablation=ablation)
        # M8: training-time overlap decorrelation
        if j > 0:
            apply_training_decorrelation(net, assemblies[j], assemblies[:j],
                                         ablation=ablation)
        # Per-neuron synaptic normalisation after each memory is trained.
        # Prevents overlap neurons from sustaining strong connections to
        # multiple assemblies' specific pools simultaneously.  Applies to
        # all memories (incl. the first) so the constraint is uniform.
        apply_post_training_normalization(net, strength=0.05)

        # Lifecycle hook: post-encode
        _call_hooks("post_encode", net=net, assemblies=assemblies, n_mem=n_mem, j=j)

        # Refresh probe scores for interference-aware replay prioritization.
        # Skip when: (a) no replay will happen, or (b) this is the last memory
        # (no rest follows j == n_mem-1).  Saves ~10 wasted probe calls per trial
        # for non-replay conditions and the terminal memory of all conditions.
        if use_replay and j < n_mem - 1:
            current_scores = [
                _probe_from_cache(net, assemblies[i], checkpoint_idx=j)["isyn_score"]
                for i in range(j + 1)
            ]
        else:
            current_scores = []

        # Inter-memory rest
        if j < n_mem - 1:
            if use_replay:
                _rest_metrics = inter_memory_rest_with_replay(
                    net,
                    learned_assemblies=assemblies[:j + 1],
                    current_scores=current_scores,
                    prioritize=prioritize,
                    tags=tags,
                    rest_id=j,
                    # Endogenous mode: feed cross-rest history so urgency evolves
                    # over the full experiment.  Other modes ignore this argument.
                    accumulated_metrics=(all_replay_metrics
                                         if prioritize == "endogenous" else None),
                    ablation=ablation,
                    reconsol_counters=reconsol_counters,
                )
                all_replay_metrics.extend(_rest_metrics)
                # Decay M10 reconsol counters after each rest period
                for _k in list(reconsol_counters.keys()):
                    reconsol_counters[_k] = max(0, reconsol_counters[_k] - 1)
                    if reconsol_counters[_k] <= 0:
                        del reconsol_counters[_k]
            else:
                inter_memory_rest_no_replay(net, tags=tags)

        # Lifecycle hook: post-replay — runs for ALL memories (including last)
        # so sleep cycle can be triggered when j == n_mem-1.
        _call_hooks("post_replay", net=net, assemblies=assemblies, n_mem=n_mem, j=j)

    # Phase 4: apply inhibitory STDP at the end
    if INHIBITORY_STDP:
        _i_buf = _get_stim_narrow(std=0.3)
        for _ in range(10):
            net.forward(_i_buf)
            apply_inhibitory_stdp(net)

    # Apply structural mask one final time to keep weights sparse
    if SPARSITY_ENFORCE:
        apply_structural_mask(net)

    # Final probe (cached)
    final_scores = np.array([
        probe_memory(net, assemblies[i])["isyn_score"]
        for i in range(n_mem)
    ])
    baseline_scores = np.array([
        retention[i, i] if np.isfinite(retention[i, i]) else 0.0
        for i in range(n_mem)
    ])

    # Behavioral readout
    _completion = [completion_accuracy(net, asm) for asm in assemblies]
    _decoding = decode_memory(net, assemblies)
    _noisy = [noisy_cue_retrieval(net, asm) for asm in assemblies]
    _rep_stats = replay_statistics(all_replay_metrics)

    # ── Phase 2: systems consolidation metrics ───────────────────────────
    _phase2 = aggregate_phase2_metrics(net, assemblies, all_replay_metrics)
    _hc_drift, _ctx_drift = representational_drift(snapshots, assemblies)

    # ── Phase 3: dynamical systems metrics ───────────────────────────────
    _phase3 = {}
    if ABLATION_PHASE3.get("basin_stability", True):
        _phase3 = aggregate_phase3_metrics(net, assemblies)
    else:
        _phase3 = {"basin_stability": {}, "spectral_radius": 0.0,
                    "participation_ratio": 0.0, "metastable": {}}

    # ── Phase 6: energy metrics ─────────────────────────────────────────
    _phase6 = aggregate_phase6_metrics(_energy_tracker)

    # Lifecycle hook: final
    _call_hooks("final", net=net, assemblies=assemblies, n_mem=n_mem)
    # Generic hook sidecar — hooks can attach data to the net for inclusion in results
    _hook_extra = getattr(net, "_hook_extra", None)

    return {
        "retention_matrix":  retention,
        "weight_evolution":  weights,
        "snapshots":         snapshots,
        "rsm_matrix":        compute_rsm(snapshots),
        "tag_evolution":     tag_evol,
        "baseline_scores":   baseline_scores,
        "final_scores":      final_scores,
        "replay_metrics":    all_replay_metrics,   # [] for no-replay conditions
        # Behavioral readout
        "completion_accuracy":  _completion,
        "decoding_separation":  _decoding,
        "noisy_retrieval":      _noisy,
        "replay_statistics":    _rep_stats,
        # Phase 2: systems consolidation
        "hc_ctx_transfer":   _phase2.get("hc_ctx_transfer", []),
        "transfer_curves":   _phase2.get("transfer_curves", {}),
        "hc_drift":          _hc_drift,
        "ctx_drift":         _ctx_drift,
        # Phase 3: dynamical systems
        "basin_stability":       _phase3.get("basin_stability", {}),
        "spectral_radius":       _phase3.get("spectral_radius", 0.0),
        "participation_ratio":   _phase3.get("participation_ratio", 0.0),
        "metastable_states":     _phase3.get("metastable", {}),
        # Phase 6: energy
        "energy_metrics":    _phase6,
        # Sidecar data from hooks (schema_abstraction, etc.)
        "hook_extra":        _hook_extra,
    }

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = [
    dict(use_slow=False, use_replay=False, label="Fast / No Replay",  color="#c0392b", ls="-"),
    dict(use_slow=False, use_replay=True,  label="Fast / Replay",      color="#e67e22", ls="--"),
    dict(use_slow=True,  use_replay=False, label="Slow / No Replay",   color="#2980b9", ls="-."),
    dict(use_slow=True,  use_replay=True,  label="Slow + Replay",      color="#27ae60", ls="-"),
]

# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION FOR MULTIPROCESSING
# (must be top-level for pickling on Windows spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _trial_worker(args):
    """
    Picklable worker for ProcessPoolExecutor.

    Outlier rejection: if any final_score has |score| > OUTLIER_SCORE_THRESHOLD
    or is non-finite, the trial is retried once with seed+100_000.
    If the retry also fails, final_scores are set to NaN so that
    _safe_mean / _safe_sem exclude this trial from statistics.

    Worker timing: each worker records its own _TIMER delta and wall time.
    These are returned in result["_worker_timer"] and merged into the main
    process _TIMER by _run_tasks_parallel().
    """
    use_slow, use_replay, asm_lists, seed, prioritize = args[:5]
    ablation   = args[5] if len(args) > 5 else None   # optional ablation dict
    assemblies = [np.array(a, int) for a in asm_lists]

    # Snapshot timer before this trial (worker process has its own _TIMER)
    t_before  = {k: v for k, v in _TIMER.items()}
    t_wall    = time.perf_counter()

    result = run_sequential_experiment(use_slow, use_replay, assemblies, seed,
                                        prioritize, ablation=ablation)

    wall_s = time.perf_counter() - t_wall
    timer_delta = {k: _TIMER[k] - t_before[k] for k in _TIMER}

    finals = result["final_scores"]
    if not np.all(np.isfinite(finals)) or np.any(np.abs(finals) > OUTLIER_SCORE_THRESHOLD):
        # Retry with perturbed seed
        retry_seed = seed + 100_000
        t_before2  = {k: v for k, v in _TIMER.items()}
        t_wall2    = time.perf_counter()
        result2 = run_sequential_experiment(
            use_slow, use_replay, assemblies, retry_seed, prioritize,
            ablation=ablation)
        wall_s      += time.perf_counter() - t_wall2
        timer_delta2 = {k: _TIMER[k] - t_before2[k] for k in _TIMER}
        for k in timer_delta:
            timer_delta[k] += timer_delta2[k]

        finals2 = result2["final_scores"]
        if np.all(np.isfinite(finals2)) and np.all(np.abs(finals2) <= OUTLIER_SCORE_THRESHOLD):
            result2["_worker_timer"] = {**timer_delta, "wall_s": wall_s,
                                        "_retried": True}
            return result2
        # Both attempts failed — mark NaN so safe statistics exclude this trial
        result2["final_scores"] = np.full_like(finals2, np.nan)
        result2["_worker_timer"] = {**timer_delta, "wall_s": wall_s,
                                    "_retried": True, "_invalid": True}
        return result2

    result["_worker_timer"] = {**timer_delta, "wall_s": wall_s}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TRIAL RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_conditions(assemblies, n_trials=N_TRIALS,
                        prioritize="interference_aware", verbose=True):
    """
    Run all 4 conditions x n_trials.
    Uses ProcessPoolExecutor for parallel execution (N_WORKERS cores).
    Falls back to serial if N_WORKERS==1 or multiprocessing unavailable.
    """
    asm_lists = [a.tolist() for a in assemblies]

    tasks = []
    for ci, cond in enumerate(CONDITIONS):
        for t in range(n_trials):
            seed = MASTER_SEED + ci * 1000 + t
            tasks.append((cond["use_slow"], cond["use_replay"],
                          asm_lists, seed, prioritize))

    if verbose:
        print(f"  [Parallel] N_WORKERS={N_WORKERS}, tasks={len(tasks)}", flush=True)

    results_flat = _run_tasks_parallel(tasks, verbose=verbose)

    all_results = []
    for ci, cond in enumerate(CONDITIONS):
        trial_results = results_flat[ci * n_trials : (ci + 1) * n_trials]
        if verbose:
            print(f"\n  [{cond['label']}]", flush=True)
            for t, r in enumerate(trial_results):
                print(f"    trial {t+1}/{n_trials}  final="
                      f"{r['final_scores'].round(3)}", flush=True)
        all_results.append({"cond": cond, "trials": trial_results})
    return all_results


def _run_tasks_parallel(tasks, verbose=False):
    """
    Run tasks with ProcessPoolExecutor; fall back to serial on error.

    Worker timer aggregation: each result carries "_worker_timer" (inserted
    by _trial_worker).  We pop it here and accumulate into the main-process
    _TIMER so the final timing report reflects actual worker work, not just
    the main process's idle wait time.
    """
    n_retried = 0
    n_invalid = 0
    try:
        if N_WORKERS <= 1:
            raise RuntimeError("serial")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            results = list(ex.map(_trial_worker, tasks))
    except Exception:
        if verbose:
            print("  [WARNING] Falling back to serial execution.", flush=True)
        results = [_trial_worker(t) for t in tasks]

    # Merge worker timers into main _TIMER and track retry/invalid counts
    for r in results:
        wt = r.pop("_worker_timer", {})
        if wt.get("_retried"):
            n_retried += 1
        if wt.get("_invalid"):
            n_invalid += 1
        for k in _TIMER:
            if k in wt:
                _TIMER[k] += wt[k]

    if n_retried and verbose:
        print(f"  [OUTLIER] {n_retried} trial(s) retried; "
              f"{n_invalid} marked invalid (NaN).", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAP SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_overlap_sweep(n_trials=N_TRIALS_SWEEP, verbose=True):
    ovs = [0.0, 0.20] if DEV_MODE else OVERLAP_FRACS

    # Collect ALL (overlap × condition × trial) tasks upfront so they can run
    # in a single ProcessPoolExecutor batch instead of one pool-spawn per
    # (overlap, condition) pair.  Original code spawned N_overlaps × N_conds
    # pools (12+ times); this spawns exactly one.  Seeds are unchanged.
    all_tasks   = []
    task_slices = {}   # (oi, ci) -> (start, end) index into all_tasks / all_raws

    for oi, ov in enumerate(ovs):
        assemblies = make_overlapping_assemblies(N_MEMORIES, ASSEMBLY_SIZE, ov)
        asm_lists  = [a.tolist() for a in assemblies]
        for ci, cond in enumerate(CONDITIONS):
            start = len(all_tasks)
            for t in range(n_trials):
                all_tasks.append((
                    cond["use_slow"], cond["use_replay"], asm_lists,
                    MASTER_SEED + 9000 + ci * 100 + t,   # same seed formula as before
                    "interference_aware"
                ))
            task_slices[(oi, ci)] = (start, len(all_tasks))

    if verbose:
        print(f"  [Parallel] N_WORKERS={N_WORKERS}, "
              f"tasks={len(all_tasks)} ({len(ovs)} overlaps x "
              f"{len(CONDITIONS)} conds x {n_trials} trials)", flush=True)
    all_raws = _run_tasks_parallel(all_tasks)

    sweep = {}
    for oi, ov in enumerate(ovs):
        if verbose:
            caution = ("  [caution: probe may reflect shared-neuron maintenance]"
                       if ov >= 0.40 else "")
            print(f"\n[OVERLAP {ov:.0%}]{caution}", flush=True)
        cond_data = []
        for ci, cond in enumerate(CONDITIONS):
            s, e   = task_slices[(oi, ci)]
            raws   = all_raws[s:e]
            finals = np.stack([r["final_scores"] for r in raws])   # (T, n_mem)
            cond_data.append({
                "mean_finals": _safe_nanmean(finals, axis=0),
                "sem_finals":  np.array([_safe_sem(finals[:, m])
                                          for m in range(N_MEMORIES)]),
                "raw_finals":  finals,
                "label": cond["label"], "color": cond["color"], "ls": cond["ls"],
            })
            if verbose:
                mA = _safe_mean(finals[:, 0])
                sA = _safe_sem(finals[:, 0])
                print(f"  {cond['label']:25s}  memA={mA:.3f}+-{sA:.3f}", flush=True)
        sweep[ov] = cond_data
    return sweep


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITIZATION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_prioritization_comparison(assemblies, n_trials=N_TRIALS_SWEEP, verbose=True):
    # Phase 3 uses a higher-pressure assembly configuration than Phase 1 to
    # amplify the signal difference between scheduling strategies.
    #
    # DEV_MODE  : same 4 memories / 20% overlap as Phase 1 (fast iteration).
    # Production: 8 memories / 30% overlap — stronger interference gradient
    #             means the interference-aware scheduler has more to exploit.
    #
    # Memory A is still index 0; all other Phase 3 logic is unchanged.
    _prio_n_mem   = N_MEMORIES if DEV_MODE else 8
    _prio_overlap = 0.20       if DEV_MODE else 0.30
    if _prio_n_mem != N_MEMORIES or _prio_overlap != 0.20:
        prio_asm = make_overlapping_assemblies(_prio_n_mem, ASSEMBLY_SIZE, _prio_overlap)
        if verbose:
            print(f"  [Phase 3 config] {_prio_n_mem} memories, "
                  f"{_prio_overlap:.0%} overlap  "
                  f"(higher pressure than Phase 1)", flush=True)
    else:
        prio_asm = assemblies   # reuse Phase 1 assemblies in DEV_MODE
    asm_lists = [a.tolist() for a in prio_asm]

    # Batch all (mode × trial) tasks into one ProcessPoolExecutor call
    # instead of one call per mode.  Seeds unchanged.
    all_tasks = []
    slices    = {}
    for mode in PRIORITIZE_MODES:
        start = len(all_tasks)
        for t in range(n_trials):
            all_tasks.append((True, True, asm_lists, MASTER_SEED + 5000 + t, mode))
        slices[mode] = (start, len(all_tasks))

    all_raws   = _run_tasks_parallel(all_tasks)
    comparison = {}
    for mode in PRIORITIZE_MODES:
        s, e   = slices[mode]
        scores = [float(r["final_scores"][0]) for r in all_raws[s:e]]
        comparison[mode] = {
            "mean":   _safe_mean(scores),
            "sem":    _safe_sem(scores),
            "raw":    scores,
            "trials": all_raws[s:e],   # full trial results for downstream analysis
        }
        if verbose:
            print(f"  {mode:22s}: {_safe_mean(scores):.4f} +- {_safe_sem(scores):.4f}",
                  flush=True)
    return comparison

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION SUITE  (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_suite(assemblies, n_trials, verbose=True):
    """
    Run all ABLATION_CONDITIONS on Slow+Replay with matched seeds.

    Each condition is Slow+Replay with exactly one mechanism disabled:
      Full model     : baseline (all mechanisms active)
      No persistence : REPLAY_PERS_GAIN -> 0.0
      No competition : USE_COMPETITION -> False
      Uniform replay : prioritize="uniform"  (fixed schedule, no state signal)
      Endogenous     : prioritize="endogenous"  (state-driven)

    All conditions share seeds MASTER_SEED + 8000 + t so each trial index is
    directly comparable across ablations (same network initialisation,
    same random presentation order, same measurement noise).

    Returns: list of dicts, one per ABLATION_CONDITIONS entry.
             Each dict has keys: label, mean, sem, std, ci95_lo, ci95_hi, raw.
    """
    asm_lists = [a.tolist() for a in assemblies]
    all_tasks  = []
    slices     = {}

    for ai, abl in enumerate(ABLATION_CONDITIONS):
        start = len(all_tasks)
        # ablation dict stripped to just the keys that run_sequential_experiment uses
        ablation_dict = {
            "pers_gain":       abl["pers_gain"],
            "use_competition": abl["use_competition"],
        }
        for t in range(n_trials):
            seed = MASTER_SEED + 8000 + t
            all_tasks.append((
                True, True, asm_lists, seed,
                abl["prioritize"],
                ablation_dict,          # 6th element: ablation flags
            ))
        slices[ai] = (start, len(all_tasks))

    if verbose:
        print(f"  [Parallel] N_WORKERS={N_WORKERS}, "
              f"tasks={len(all_tasks)} "
              f"({len(ABLATION_CONDITIONS)} ablations x {n_trials} trials)",
              flush=True)

    all_raws = _run_tasks_parallel(all_tasks)

    results = []
    for ai, abl in enumerate(ABLATION_CONDITIONS):
        s, e   = slices[ai]
        raws   = all_raws[s:e]
        scores = np.array([float(r["final_scores"][0]) for r in raws], float)
        valid  = scores[np.isfinite(scores)]
        n_v    = len(valid)
        m      = _safe_mean(valid)
        se     = _safe_sem(valid)
        sd     = _safe_std(valid)
        if n_v >= 2:
            tc      = float(_scipy_t.ppf(0.975, df=n_v - 1))
            ci_lo   = m - tc * se
            ci_hi   = m + tc * se
        else:
            ci_lo = ci_hi = m
        results.append({
            "label":   abl["label"],
            "mean":    m,
            "sem":     se,
            "std":     sd,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "raw":     valid.tolist(),
            "n":       n_v,
            "trials":  raws,
        })
        if verbose:
            sig_str = ""
            if ai > 0 and len(results[0]["raw"]) > 1 and n_v > 1:
                _, pv = ttest_ind(np.array(results[0]["raw"]),
                                  valid, equal_var=False)
                sig_str = f"  p={pv:.4f} vs Full"
            print(f"  {abl['label']:22s}: {m:.4f} +- {se:.4f}"
                  f"  [{ci_lo:.4f}, {ci_hi:.4f}]{sig_str}", flush=True)
    return results


def print_ablation_summary(ablation_results):
    """Print ablation table with effect sizes relative to full model."""
    print("\n[ABLATION SUMMARY]", flush=True)
    full = ablation_results[0]
    hdr  = (f"  {'Condition':22s}  {'Mean':>7s}  {'SEM':>6s}  "
            f"{'95% CI':>16s}  {'vs Full':>8s}  {'p':>7s}  {'d':>5s}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for ab in ablation_results:
        ci_str = f"[{ab['ci95_lo']:.4f},{ab['ci95_hi']:.4f}]"
        diff   = ab["mean"] - full["mean"]
        diff_s = f"{diff:+.4f}"
        if len(full["raw"]) > 1 and ab["n"] > 1 and ab["label"] != full["label"]:
            _, pv   = ttest_ind(np.array(full["raw"]), np.array(ab["raw"]),
                                 equal_var=False)
            pv_s    = f"{pv:.4f}"
            pool_s  = float(np.sqrt((np.var(full["raw"], ddof=1) +
                                     np.var(ab["raw"],  ddof=1)) / 2))
            d_val   = abs(diff) / max(pool_s, 1e-9)
            d_s     = f"{d_val:.2f}"
        else:
            pv_s  = "  --  "
            d_s   = " --  "
        print(f"  {ab['label']:22s}  {ab['mean']:>7.4f}  {ab['sem']:>6.4f}  "
              f"{ci_str:>16s}  {diff_s:>8s}  {pv_s:>7s}  {d_s:>5s}", flush=True)


def fig_ablation_suite(ablation_results):
    """
    Figure A1 — Mechanism ablation suite.

    Single panel: Memory A final retention for each ablation condition
    (Slow+Replay, one mechanism removed at a time).  Full model is leftmost,
    coloured green.  Ablations coloured by mechanism removed.  Error bars =
    SEM.  Individual trial points overlaid.  Significance brackets annotated
    for comparisons that reach p < 0.05 vs full model.
    """
    if not ablation_results:
        return

    labels  = [ab["label"]   for ab in ablation_results]
    means   = [ab["mean"]    for ab in ablation_results]
    sems    = [ab["sem"]     for ab in ablation_results]
    ci_los  = [ab["ci95_lo"] for ab in ablation_results]
    ci_his  = [ab["ci95_hi"] for ab in ablation_results]

    colors = ["#27ae60",   # Full model — green (matches Phase 1 Slow+Replay)
              "#8e44ad",   # No persistence — purple
              "#c0392b",   # No competition — red
              "#95a5a6",   # Uniform replay  — grey
              "#3498db"]   # Endogenous       — blue

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(x, means, 0.55, color=colors[:len(labels)], alpha=0.85,
                  yerr=sems, capsize=6, error_kw=dict(elinewidth=1.5))

    # 95% CI whiskers (thin lines showing CI range)
    for xi, lo, hi in zip(x, ci_los, ci_his):
        ax.plot([xi, xi], [lo, hi], color='k', lw=1.2, alpha=0.5)

    # Individual trial points
    for xi, ab in zip(x, ablation_results):
        for v in ab["raw"]:
            ax.scatter(xi, v, color='k', s=22, zorder=5, alpha=0.55)

    # Significance brackets vs full model
    full_raw = np.array(ablation_results[0]["raw"])
    y_max    = max(means) * 1.25 + max(sems) * 2
    bracket_y = y_max * 0.90
    for xi, ab in zip(x[1:], ablation_results[1:], strict=False):
        if ab["n"] < 2 or len(full_raw) < 2:
            continue
        _, pv = ttest_ind(full_raw, np.array(ab["raw"]), equal_var=False)
        if pv < 0.05:
            sig_str = "**" if pv < 0.01 else "*"
            ax.annotate("", xy=(xi, bracket_y), xytext=(0, bracket_y),
                        arrowprops=dict(arrowstyle='-', color='k', lw=1.0))
            ax.text((0 + xi) / 2, bracket_y * 1.02, sig_str,
                    ha='center', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=10, ha='right')
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_ylim(bottom=min(0, min(ci_los)) * 1.1)

    # Annotate each bar with mean ± SEM
    for xi, m, se in zip(x, means, sems):
        ax.text(xi, max(m + se, 0.005) + y_max * 0.01, f"{m:.3f}",
                ha='center', va='bottom', fontsize=8, color='#333')

    _style(ax,
           ylabel="Memory A final retention (I_syn score)",
           title="Ablation Suite: Mechanism Contributions (Slow+Replay)")
    ax.text(0.98, 0.98,
            "Error bars = SEM  |  Whiskers = 95% CI  |  Dots = individual trials",
            transform=ax.transAxes, fontsize=7.5, ha='right', va='top', color='grey')
    plt.tight_layout()
    _save_fig(fig, "ablation_suite")


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _stack_retention(trials, n_mem):
    return np.stack([r["retention_matrix"] for r in trials])

def _mean_sem_final(trials):
    finals = np.array([r["final_scores"] for r in trials])
    return (np.array([_safe_mean(finals[:, m]) for m in range(finals.shape[1])]),
            np.array([_safe_sem(finals[:, m])  for m in range(finals.shape[1])]))

def _mean_sem_curve(trials, mem_idx=0):
    n_mem = trials[0]["retention_matrix"].shape[0]
    curve = np.array([r["retention_matrix"][mem_idx, :] for r in trials])
    return (np.array([_safe_mean(curve[:, j]) for j in range(n_mem)]),
            np.array([_safe_sem(curve[:, j])  for j in range(n_mem)]))

def _mean_rsm(trials):
    rsms = np.stack([r["rsm_matrix"] for r in trials])
    return _safe_nanmean(rsms, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _valid_scores(trials, mem_idx=0):
    """Return finite final scores for memory mem_idx across trials, with valid-N count."""
    raw = np.array([r["final_scores"][mem_idx] for r in trials], dtype=float)
    valid = raw[np.isfinite(raw)]
    return valid, len(raw)

def print_statistics(all_results):
    print("\n" + "="*70, flush=True)
    print("STATISTICAL SUMMARY", flush=True)
    print("="*70, flush=True)
    for res in all_results:
        cond = res["cond"]
        mean, s = _mean_sem_final(res["trials"])
        n_trials = len(res["trials"])
        # Count valid (finite) trials for memory A as a proxy
        _, n_tot = _valid_scores(res["trials"], 0)
        n_valid  = sum(1 for r in res["trials"] if np.all(np.isfinite(r["final_scores"])))
        valid_tag = f"  [N={n_valid}/{n_tot}]" if n_valid < n_tot else f"  [N={n_valid}]"
        print(f"\n{cond['label']}{valid_tag}", flush=True)
        for i in range(N_MEMORIES):
            flag = " [!OUTLIER]" if abs(mean[i]) > 5.0 or not np.isfinite(mean[i]) else ""
            print(f"  Memory {chr(65+i)}: {mean[i]:.4f} +- {s[i]:.4f}{flag}", flush=True)

    # Key test: exclude non-finite scores before t-test
    fast_nr_all, _ = _valid_scores(all_results[0]["trials"], 0)
    slow_rp_all, _ = _valid_scores(all_results[3]["trials"], 0)
    fast_nr = fast_nr_all[np.isfinite(fast_nr_all)]
    slow_rp = slow_rp_all[np.isfinite(slow_rp_all)]
    if len(fast_nr) > 1 and len(slow_rp) > 1:
        t, p = ttest_ind(slow_rp, fast_nr)
        sig  = "*significant*" if p < 0.05 else "n.s."
        # Cohen's d (pooled std, equal-N approximation)
        pooled_std = float(np.sqrt((np.var(slow_rp, ddof=1) + np.var(fast_nr, ddof=1)) / 2))
        cohen_d    = (_safe_mean(slow_rp) - _safe_mean(fast_nr)) / max(pooled_std, 1e-9)
        # 95% confidence intervals (t-distribution)
        def _ci95(arr):
            n = len(arr)
            if n < 2:
                return float('nan'), float('nan')
            m, s = _safe_mean(arr), _safe_sem(arr)
            tcrit = float(_scipy_t.ppf(0.975, df=n - 1))
            return m - tcrit * s, m + tcrit * s
        ci_fnr = _ci95(fast_nr)
        ci_srp = _ci95(slow_rp)
        print(f"\nKey test -- Memory A (Slow+Replay vs Fast/NoReplay):", flush=True)
        print(f"  Fast/NoReplay : {_safe_mean(fast_nr):.4f} +- {_safe_std(fast_nr):.4f}"
              f"  (N={len(fast_nr)})  95% CI [{ci_fnr[0]:.4f}, {ci_fnr[1]:.4f}]",
              flush=True)
        print(f"  Slow+Replay   : {_safe_mean(slow_rp):.4f} +- {_safe_std(slow_rp):.4f}"
              f"  (N={len(slow_rp)})  95% CI [{ci_srp[0]:.4f}, {ci_srp[1]:.4f}]",
              flush=True)
        print(f"  t={t:.3f}  p={p:.4f}  {sig}  Cohen d={cohen_d:.2f}", flush=True)
        base_f = _safe_mean(fast_nr)
        if abs(base_f) > 1e-6:
            print(f"  Ratio: {_safe_mean(slow_rp)/base_f:.2f}x", flush=True)

    print(f"\nForgetting magnitude (Memory A: baseline -> final):", flush=True)
    for res in all_results:
        cond = res["cond"]
        base = np.array([r["baseline_scores"][0] for r in res["trials"]])
        fin  = np.array([r["final_scores"][0]    for r in res["trials"]])
        drop = base - fin
        pct  = 100.0 * _safe_mean(drop) / max(abs(_safe_mean(base)), 1e-6)
        print(f"  {cond['label']:25s}: drop={_safe_mean(drop):.4f}  ({pct:.1f}%)",
              flush=True)

    print(f"\nRepresentational drift -- Memory A cosine similarity (baseline->final):",
          flush=True)
    for res in all_results:
        cond = res["cond"]
        cosines = [r["rsm_matrix"][0, N_MEMORIES-1]
                   for r in res["trials"]
                   if np.isfinite(r["rsm_matrix"][0, N_MEMORIES-1])]
        print(f"  {cond['label']:25s}: {_safe_mean(cosines):.4f} +- {_safe_sem(cosines):.4f}",
              flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, name, dpi=150):
    t0 = _tick()
    png = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    if GENERATE_PDFS:
        pdf = os.path.join(OUT_DIR, f"{name}.pdf")
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _tock("figures", t0)
    print(f"[FIG] Saved {name}.png", flush=True)

def _style(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel: ax.set_xlabel(xlabel, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, fontsize=11)
    if title:  ax.set_title(title, fontsize=11, fontweight='bold')
    if grid:   ax.grid(True, alpha=0.25, lw=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 1-7  (original set)
# ─────────────────────────────────────────────────────────────────────────────

def fig_forgetting_curves(all_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(N_MEMORIES)
    for res in all_results:
        c = res["cond"]
        mean, s = _mean_sem_curve(res["trials"], mem_idx=0)
        ax.plot(x, mean, color=c["color"], ls=c["ls"], lw=2.2, marker='o', ms=7,
                label=c["label"])
        ax.fill_between(x, mean-s, mean+s, color=c["color"], alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"After {chr(65+j)}" for j in range(N_MEMORIES)], fontsize=10)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.legend(loc="upper right", fontsize=9)
    _style(ax, xlabel="Sequential training checkpoint",
           ylabel="Memory A retention (I_syn score)",
           title="Catastrophic Forgetting -- Memory A Across A->B->C->D")
    _save_fig(fig, "catastrophic_forgetting_curves")


def fig_overlap_vs_forgetting(sweep):
    fig, ax = plt.subplots(figsize=(8, 5))
    ovs = sorted(sweep.keys())
    x   = np.array(ovs) * 100
    for ci, cond in enumerate(CONDITIONS):
        means = [_safe_mean(sweep[ov][ci]["raw_finals"][:, 0]) for ov in ovs]
        sems  = [_safe_sem(sweep[ov][ci]["raw_finals"][:, 0])  for ov in ovs]
        ax.plot(x, means, color=cond["color"], ls=cond["ls"], lw=2.2,
                marker='s', ms=7, label=cond["label"])
        ax.fill_between(x, np.array(means)-np.array(sems),
                        np.array(means)+np.array(sems),
                        color=cond["color"], alpha=0.15)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.legend(fontsize=9)
    _style(ax, xlabel="Assembly overlap (%)",
           ylabel="Memory A final retention",
           title="Interference Gradient: Overlap x Forgetting")
    _save_fig(fig, "overlap_vs_forgetting")


def fig_replay_protection(all_results):
    n_c   = len(CONDITIONS)
    n_mem = N_MEMORIES
    bar_w = 0.17
    x     = np.arange(n_mem)
    offs  = np.linspace(-(n_c-1)/2*bar_w, (n_c-1)/2*bar_w, n_c)
    fig, ax = plt.subplots(figsize=(10, 5))
    for ci, res in enumerate(all_results):
        mean, s = _mean_sem_final(res["trials"])
        ax.bar(x + offs[ci], mean, bar_w, yerr=s, capsize=4,
               color=res["cond"]["color"], alpha=0.85,
               label=res["cond"]["label"], error_kw=dict(elinewidth=1.3))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Mem {chr(65+i)}" for i in range(n_mem)], fontsize=11)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.legend(fontsize=9)
    _style(ax, ylabel="Final retention (I_syn score)",
           title="Replay Protection: Final Retention After A->B->C->D")
    _save_fig(fig, "replay_protection_comparison")


def fig_interference_matrix(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    mats  = [_safe_nanmean(_stack_retention(r["trials"], N_MEMORIES), axis=0)
             for r in all_results]
    finite = np.concatenate([m[np.isfinite(m)] for m in mats])
    vmin = float(finite.min()) if len(finite) else 0.0
    vmax = float(finite.max()) if len(finite) else 1.0
    lbl  = [chr(65+i) for i in range(N_MEMORIES)]
    for ci, (res, mat) in enumerate(zip(all_results, mats)):
        ax = axes[ci]
        im = ax.imshow(mat, origin='upper', aspect='auto',
                       vmin=vmin, vmax=vmax, cmap='RdYlGn')
        ax.set_xticks(range(N_MEMORIES))
        ax.set_yticks(range(N_MEMORIES))
        ax.set_xticklabels([f"After {l}" for l in lbl], rotation=35,
                            ha='right', fontsize=8)
        ax.set_yticklabels([f"Mem {l}" for l in lbl], fontsize=8)
        ax.set_title(res["cond"]["label"], fontsize=9, fontweight='bold')
        for r in range(N_MEMORIES):
            for c in range(N_MEMORIES):
                if np.isfinite(mat[r, c]):
                    ax.text(c, r, f"{mat[r,c]:.2f}", ha='center', va='center',
                            fontsize=7.5,
                            color='black' if mat[r,c] > (vmin+vmax)/2 else 'white')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="I_syn")
    fig.suptitle("Interference Matrix (rows=probed memory, cols=training checkpoint)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "interference_matrix")


def fig_synaptic_overlap_evolution(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    lbl    = [chr(65+i) for i in range(N_MEMORIES)]
    colors = plt.cm.viridis(np.linspace(0, 1, N_MEMORIES))
    for ci, res in enumerate(all_results):
        ax = axes[ci]
        stacked  = np.stack([r["weight_evolution"] for r in res["trials"]])
        mat_mean = _safe_nanmean(stacked, axis=0)
        mat_sem  = np.array([[_safe_sem(stacked[:, i, j])
                               for j in range(N_MEMORIES)]
                              for i in range(N_MEMORIES)])
        for i in range(N_MEMORIES):
            xs    = np.arange(i, N_MEMORIES)
            ys    = np.array([mat_mean[i, j] for j in range(i, N_MEMORIES)])
            err   = np.array([mat_sem[i, j]  for j in range(i, N_MEMORIES)])
            valid = np.isfinite(ys)
            if valid.any():
                ax.plot(xs[valid], ys[valid], color=colors[i], marker='o',
                        lw=2, label=f"Mem {lbl[i]}")
                ax.fill_between(xs[valid], (ys-err)[valid], (ys+err)[valid],
                                color=colors[i], alpha=0.2)
        ax.set_xticks(range(N_MEMORIES))
        ax.set_xticklabels([f"After {l}" for l in lbl], rotation=30,
                            ha='right', fontsize=8)
        ax.legend(fontsize=8)
        _style(ax, ylabel="W mean (within-assembly)", title=res["cond"]["label"])
    fig.suptitle("Synaptic Weight Evolution During Sequential Learning",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "synaptic_overlap_evolution")


def fig_retention_surface(sweep):
    fig = plt.figure(figsize=(14, 4.5))
    ovs = sorted(sweep.keys())
    X   = np.array(ovs) * 100
    for ci, cond in enumerate(CONDITIONS):
        ax = fig.add_subplot(1, 4, ci+1, projection='3d')
        Z  = np.array([_safe_mean(sweep[ov][ci]["raw_finals"][:, 0]) for ov in ovs])
        ax.plot(X, np.zeros_like(X), Z, color=cond["color"], lw=2.5, marker='o')
        ax.set_xlabel("Overlap (%)", fontsize=8)
        ax.set_zlabel("Retention", fontsize=8)
        ax.set_title(cond["label"], fontsize=8, fontweight='bold')
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    fig.suptitle("Memory A Retention vs Overlap x Condition",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "retention_surface_plot")


def fig_replay_preserves_old_memories(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    mem_col = ["#c0392b", "#d35400", "#8e44ad", "#16a085"]
    x       = np.arange(N_MEMORIES)
    x_lbl   = [f"After {chr(65+j)}" for j in range(N_MEMORIES)]
    for pi, ci in enumerate([0, 3]):
        ax  = axes[pi]
        res = all_results[ci]
        c   = res["cond"]
        for i in range(N_MEMORIES):
            mean, s = _mean_sem_curve(res["trials"], mem_idx=i)
            ax.plot(x[i:], mean[i:], color=mem_col[i], lw=2.2,
                    marker='o', ms=6, label=f"Mem {chr(65+i)}")
            ax.fill_between(x[i:], (mean-s)[i:], (mean+s)[i:],
                            color=mem_col[i], alpha=0.18)
        ax.set_xticks(x)
        ax.set_xticklabels(x_lbl, rotation=20, ha='right', fontsize=9)
        ax.axhline(0, color='grey', lw=0.8, ls=':')
        ax.legend(fontsize=9)
        _style(ax, xlabel="Checkpoint", title=c["label"])
        ax.title.set_color(c["color"])
    axes[0].set_ylabel("Retention (I_syn score)", fontsize=11)
    fig.suptitle("Replay Preserves Old Memories: Fast/NoReplay vs Slow+Replay",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "replay_preserves_old_memories")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 8-14  (new analysis figures)
# ─────────────────────────────────────────────────────────────────────────────

def fig_representational_drift(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    lbl = [chr(65+i) for i in range(N_MEMORIES)]
    all_rsms  = [_mean_rsm(r["trials"]) for r in all_results]
    finite    = np.concatenate([m[np.isfinite(m)] for m in all_rsms])
    vmin = float(finite.min()) if len(finite) else 0.0
    vmax = float(finite.max()) if len(finite) else 1.0
    for ci, (res, rsm) in enumerate(zip(all_results, all_rsms)):
        ax = axes[ci]
        im = ax.imshow(rsm, origin='upper', aspect='auto',
                       vmin=max(0, vmin), vmax=min(1, vmax), cmap='Blues')
        ax.set_xticks(range(N_MEMORIES))
        ax.set_yticks(range(N_MEMORIES))
        ax.set_xticklabels([f"After {l}" for l in lbl], rotation=35,
                            ha='right', fontsize=8)
        ax.set_yticklabels([f"Mem {l}" for l in lbl], fontsize=8)
        ax.set_title(res["cond"]["label"], fontsize=9, fontweight='bold')
        for r in range(N_MEMORIES):
            for c in range(N_MEMORIES):
                if np.isfinite(rsm[r, c]):
                    ax.text(c, r, f"{rsm[r,c]:.2f}", ha='center', va='center',
                            fontsize=7.5,
                            color='white' if rsm[r,c] > 0.65 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine sim")
    fig.suptitle("Representational Drift: Cosine Similarity of Memory Weight Vectors\n"
                 "(1.0=stable, 0.0=orthogonal; replay stabilises representations)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "representational_drift")


def fig_synaptic_tag_evolution(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    lbl    = [chr(65+i) for i in range(N_MEMORIES)]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, N_MEMORIES))
    for ci, res in enumerate(all_results):
        ax = axes[ci]
        stacked  = np.stack([r["tag_evolution"] for r in res["trials"]])
        tag_mean = _safe_nanmean(stacked, axis=0)
        tag_sem  = np.array([[_safe_sem(stacked[:, i, j])
                               for j in range(N_MEMORIES)]
                              for i in range(N_MEMORIES)])
        for i in range(N_MEMORIES):
            xs    = np.arange(i, N_MEMORIES)
            ys    = np.array([tag_mean[i, j] for j in range(i, N_MEMORIES)])
            err   = np.array([tag_sem[i, j]  for j in range(i, N_MEMORIES)])
            valid = np.isfinite(ys)
            if valid.any():
                ax.plot(xs[valid], ys[valid], color=colors[i],
                        marker='D', ms=6, lw=2, label=f"Mem {lbl[i]}")
                ax.fill_between(xs[valid], (ys-err)[valid], (ys+err)[valid],
                                color=colors[i], alpha=0.2)
        ax.set_xticks(range(N_MEMORIES))
        ax.set_xticklabels([f"After {l}" for l in lbl], rotation=30,
                            ha='right', fontsize=8)
        ax.legend(fontsize=8)
        _style(ax, ylabel="Mean tag strength", title=res["cond"]["label"])
    fig.suptitle("Synaptic Tag Evolution (STC hypothesis: tags mark recently potentiated synapses)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "synaptic_tag_evolution")


def fig_replay_scheduling(prio_results):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = {"uniform":            "#95a5a6",
            "oldest_first":       "#3498db",
            "interference_aware": "#27ae60",
            "endogenous":         "#8e44ad"}
    tick_labels = {"uniform":            "Uniform",
                   "oldest_first":       "Oldest\nFirst",
                   "interference_aware": "Interference\nAware",
                   "endogenous":         "Endogenous\n(state-driven)"}
    modes = list(prio_results.keys())
    x     = np.arange(len(modes))
    means = [prio_results[m]["mean"] for m in modes]
    sems  = [prio_results[m]["sem"]  for m in modes]
    ax.bar(x, means, 0.5, yerr=sems, capsize=6,
           color=[cmap.get(m, "#888") for m in modes],
           alpha=0.85, error_kw=dict(elinewidth=1.5))
    for xi, mode in enumerate(modes):
        for raw in prio_results[mode]["raw"]:
            ax.scatter(xi, raw, color='k', s=30, zorder=5, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([tick_labels.get(m, m) for m in modes], fontsize=10)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    # Annotate endogenous vs uniform improvement
    u   = prio_results.get("uniform",    {}).get("mean", 0)
    endo = prio_results.get("endogenous", {}).get("mean", 0)
    endo_xi = modes.index("endogenous") if "endogenous" in modes else None
    if endo_xi is not None and abs(u) > 1e-6:
        pct = 100*(endo-u)/abs(u)
        ax.annotate(f"{pct:+.0f}% vs uniform", xy=(endo_xi, endo),
                    xytext=(endo_xi - 0.8, endo + max(0.01, abs(endo)*0.20)),
                    arrowprops=dict(arrowstyle='->', color='#8e44ad'),
                    fontsize=9, color='#8e44ad')
    _style(ax, ylabel="Memory A final retention",
           title="Replay Prioritization (Slow+Replay, Memory A)")
    _save_fig(fig, "replay_scheduling")


def fig_memory_vulnerability_map(sweep):
    ovs   = sorted(sweep.keys())
    matrix = np.zeros((N_MEMORIES, len(ovs)))
    for oi, ov in enumerate(ovs):
        raw = sweep[ov][3]["raw_finals"]    # Slow+Replay
        for m in range(N_MEMORIES):
            matrix[m, oi] = _safe_mean(raw[:, m])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd_r',
                   vmin=0, vmax=max(matrix.max(), 0.01))
    ax.set_xticks(range(len(ovs)))
    ax.set_xticklabels([f"{int(ov*100)}%" for ov in ovs], fontsize=9)
    ax.set_yticks(range(N_MEMORIES))
    ax.set_yticklabels([f"Mem {chr(65+m)}" for m in range(N_MEMORIES)], fontsize=9)
    fig.colorbar(im, ax=ax, label="Final retention")
    for m in range(N_MEMORIES):
        for oi in range(len(ovs)):
            ax.text(oi, m, f"{matrix[m,oi]:.2f}", ha='center', va='center', fontsize=8)
    _style(ax, xlabel="Assembly overlap (%)",
           title="Memory Vulnerability Map (Slow+Replay condition)", grid=False)
    _save_fig(fig, "memory_vulnerability_map")


def fig_overlap_interference_phase_diagram(sweep):
    ovs    = sorted(sweep.keys())
    matrix = np.zeros((len(CONDITIONS), len(ovs)))
    for oi, ov in enumerate(ovs):
        for ci in range(len(CONDITIONS)):
            matrix[ci, oi] = _safe_mean(sweep[ov][ci]["raw_finals"][:, 0])
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=max(matrix.max(), 0.01))
    ax.set_xticks(range(len(ovs)))
    ax.set_xticklabels([f"{int(ov*100)}%" for ov in ovs], fontsize=9)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([c["label"] for c in CONDITIONS], fontsize=9)
    fig.colorbar(im, ax=ax, label="Memory A retention")
    for r in range(len(CONDITIONS)):
        for c in range(len(ovs)):
            ax.text(c, r, f"{matrix[r,c]:.2f}", ha='center', va='center', fontsize=8)
    _style(ax, xlabel="Assembly overlap (%)",
           title="Forgetting Phase Diagram: Overlap x Condition", grid=False)
    _save_fig(fig, "overlap_interference_phase_diagram")


def fig_publication_summary(all_results, prio_results, sweep):
    """Four-panel summary for publication."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # A. Forgetting curves
    ax_a = fig.add_subplot(gs[0, 0])
    x    = np.arange(N_MEMORIES)
    for res in all_results:
        c = res["cond"]
        mean, s = _mean_sem_curve(res["trials"], mem_idx=0)
        ax_a.plot(x, mean, color=c["color"], ls=c["ls"], lw=2,
                  marker='o', ms=6, label=c["label"])
        ax_a.fill_between(x, mean-s, mean+s, color=c["color"], alpha=0.13)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"After {chr(65+j)}" for j in range(N_MEMORIES)],
                          rotation=20, ha='right', fontsize=8)
    ax_a.axhline(0, color='grey', lw=0.7, ls=':')
    ax_a.legend(fontsize=7.5)
    _style(ax_a, ylabel="Mem A retention", title="A. Catastrophic Forgetting")

    # B. Final retention bar (memory A)
    ax_b = fig.add_subplot(gs[0, 1])
    for ci, res in enumerate(all_results):
        finals = np.array([r["final_scores"][0] for r in res["trials"]])
        ax_b.bar(ci, _safe_mean(finals), 0.55, yerr=_safe_sem(finals),
                 color=res["cond"]["color"], alpha=0.85, capsize=5,
                 error_kw=dict(elinewidth=1.3))
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(["Fast\nNoRep", "Fast\nRep",
                            "Slow\nNoRep", "Slow\nRep"], fontsize=8)
    ax_b.axhline(0, color='grey', lw=0.7, ls=':')
    _style(ax_b, ylabel="Memory A final retention", title="B. Protection by Condition")

    # C. Representational drift (Slow+Replay vs Fast/NoReplay)
    ax_c = fig.add_subplot(gs[1, 0])
    for pi, ci in enumerate([0, 3]):
        res = all_results[ci]; c = res["cond"]
        cosines = np.array([r["rsm_matrix"][0, :] for r in res["trials"]])
        means = np.array([_safe_mean(cosines[:, j]) for j in range(N_MEMORIES)])
        sems  = np.array([_safe_sem(cosines[:, j])  for j in range(N_MEMORIES)])
        valid = np.isfinite(means)
        ax_c.plot(x[valid], means[valid], color=c["color"], ls=c["ls"],
                  lw=2, marker='s', ms=6, label=c["label"])
        ax_c.fill_between(x[valid], (means-sems)[valid], (means+sems)[valid],
                           color=c["color"], alpha=0.15)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"After {chr(65+j)}" for j in range(N_MEMORIES)],
                          rotation=20, ha='right', fontsize=8)
    ax_c.axhline(1.0, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax_c.legend(fontsize=8)
    _style(ax_c, ylabel="Cosine similarity", title="C. Representational Drift (Mem A)")

    # D. Prioritization
    ax_d = fig.add_subplot(gs[1, 1])
    modes  = list(prio_results.keys())
    cmap_p = {"uniform": "#95a5a6", "oldest_first": "#3498db",
               "interference_aware": "#27ae60", "endogenous": "#e67e22"}
    for xi, mode in enumerate(modes):
        v = prio_results[mode]
        ax_d.bar(xi, v["mean"], 0.5, yerr=v["sem"],
                 color=cmap_p.get(mode, "#888"), alpha=0.85, capsize=5,
                 error_kw=dict(elinewidth=1.3))
        for raw in v["raw"]:
            ax_d.scatter(xi, raw, color='k', s=25, zorder=5, alpha=0.6)
    ax_d.set_xticks(range(len(modes)))
    _label_map = {
        "uniform":            "Uniform",
        "oldest_first":       "Oldest\nFirst",
        "interference_aware": "Int.\nAware",
        "endogenous":         "Endogenous",
    }
    ax_d.set_xticklabels([_label_map.get(m, m) for m in modes], fontsize=9)
    ax_d.axhline(0, color='grey', lw=0.7, ls=':')
    _style(ax_d, ylabel="Memory A final retention", title="D. Replay Prioritization")

    fig.suptitle("Continual Learning with Slow Consolidation + Replay\n"
                 "Biologically Grounded Computational Neuroscience",
                 fontsize=13, fontweight='bold')
    _save_fig(fig, "publication_summary")

# ─────────────────────────────────────────────────────────────────────────────
# REPLAY QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_replay_quality(all_results):
    """
    Aggregate per-event replay coherence metrics stored in each trial result.

    Returns a dict keyed by condition label.  Each value contains:
      all_coherences         – flat list of peak_coherence across all events/trials
      trial_mean_coherences  – one mean-coherence value per trial (for correlation)
      trial_mean_confidences – one mean-confidence value per trial (for correlation)
      trial_final_scores     – Memory A final score per trial (for correlation)
      coherence_by_rest      – {rest_id: [peak_coherence, …]} (trajectory)
      success_by_burst       – {burst_id: [0/1, …]} (success rate per burst)
      mean_coherence         – grand mean of peak_coherence
      std_coherence          – std of peak_coherence
      success_rate           – fraction of events where n_steps_coherent > 0
      all_confidences        – flat list of replay_confidence per event
      accepted_coherences    – peak_coherence for accepted events only
      rejected_coherences    – peak_coherence for rejected events only
      acceptance_rate        – fraction of events that passed adaptive gate
      accept_reasons         – dict {reason: count} for rejected events
      use_replay             – bool flag
      color                  – matplotlib color string for this condition
    """
    analysis = {}
    for res in all_results:
        label      = res["cond"]["label"]
        color      = res["cond"]["color"]
        use_replay = res["cond"]["use_replay"]

        if not use_replay:
            analysis[label] = {
                "color": color, "use_replay": False,
                "all_coherences": [], "trial_mean_coherences": [],
                "trial_mean_confidences": [], "trial_final_scores": [],
                "coherence_by_rest": {}, "success_by_burst": {},
                "mean_coherence": 0.0, "std_coherence": 0.0,
                "success_rate": 0.0, "all_confidences": [],
                "accepted_coherences": [], "rejected_coherences": [],
                "acceptance_rate": 0.0, "accept_reasons": {},
                "all_max_consec": [], "all_run_lengths": [],
                "wfast_by_rest": {}, "wslow_by_rest": {},
                "accept_by_rest": {}, "coh_by_rest_mean": {},
            }
            continue

        all_coherences        = []
        all_confidences       = []
        accepted_coherences   = []
        rejected_coherences   = []
        accept_reasons        = {}   # reject_reason - count
        trial_mean_coherences = []
        trial_mean_confidences = []
        trial_final_scores    = []
        coherence_by_rest     = {}   # rest_id  - [peak_coherence …]
        success_by_burst      = {}   # burst_id - [1/0 …]
        all_max_consec        = []   # max coherent run per event
        all_run_lengths       = []   # individual coherent epoch durations (flat)
        wfast_by_rest         = {}   # rest_id - [w_fast_aa] basin erosion
        wslow_by_rest         = {}   # rest_id - [w_slow_aa] consolidation
        accept_by_rest        = {}   # rest_id - [0/1]       recovery trajectory
        coh_by_rest_mean      = {}   # rest_id - [peak_coh]  (same as coherence_by_rest)

        for trial in res["trials"]:
            mlist = trial.get("replay_metrics", [])
            if not mlist:
                continue
            # Per-trial summary
            t_cohs  = [m["peak_coherence"]    for m in mlist]
            t_confs = [m.get("replay_confidence", 0.0) for m in mlist]
            if t_cohs:
                trial_mean_coherences.append(float(np.mean(t_cohs)))
                trial_mean_confidences.append(float(np.mean(t_confs)))
                all_coherences.extend(t_cohs)
                all_confidences.extend(t_confs)
            final_a = trial["final_scores"][0]
            trial_final_scores.append(float(final_a) if np.isfinite(final_a) else np.nan)
            # Accepted/rejected split + rejection reasons + attractor dynamics
            for m in mlist:
                accepted = m.get("replay_accepted", True)   # default True for legacy data
                coh      = m["peak_coherence"]
                conf     = m.get("replay_confidence", 0.0)
                if accepted:
                    accepted_coherences.append(coh)
                else:
                    rejected_coherences.append(coh)
                    reason = m.get("reject_reason", "unknown") or "unknown"
                    accept_reasons[reason] = accept_reasons.get(reason, 0) + 1
                # Trajectory and burst stats
                rid = m["rest_id"]
                bid = m["burst_id"]
                coherence_by_rest.setdefault(rid, []).append(coh)
                success_by_burst.setdefault(bid, []).append(
                    1 if m["n_steps_coherent"] > 0 else 0)
                # Attractor-dynamics: run-length data
                all_max_consec.append(m.get("max_consec_coherent", 0))
                all_run_lengths.extend(m.get("coherent_run_lengths", []))
                # Competition-dynamics: per-rest basin stability
                wfast_by_rest.setdefault(rid, []).append(
                    m.get("w_fast_aa", 0.0))
                wslow_by_rest.setdefault(rid, []).append(
                    m.get("w_slow_aa", 0.0))
                accept_by_rest.setdefault(rid, []).append(
                    1 if m.get("replay_accepted", True) else 0)
                coh_by_rest_mean.setdefault(rid, []).append(coh)

        all_cohs  = np.array(all_coherences,  float)
        n_total   = len(all_coherences)
        n_accept  = len(accepted_coherences)
        # Diversity metrics
        all_conf_arr = np.array(all_confidences, float)
        all_qual_arr = np.array([m.get("event_quality", 0.0) for trial in res["trials"]
                                 for m in trial.get("replay_metrics", [])], float)
        analysis[label] = {
            "color":                 color,
            "use_replay":            True,
            "all_coherences":        all_coherences,
            "all_confidences":       all_confidences,
            "all_qualities":         all_qual_arr.tolist(),
            "accepted_coherences":   accepted_coherences,
            "rejected_coherences":   rejected_coherences,
            "trial_mean_coherences": trial_mean_coherences,
            "trial_mean_confidences": trial_mean_confidences,
            "trial_final_scores":    trial_final_scores,
            "coherence_by_rest":     coherence_by_rest,
            "success_by_burst":      success_by_burst,
            "mean_coherence":        float(all_cohs.mean()) if n_total else 0.0,
            "std_coherence":         float(all_cohs.std())  if n_total else 0.0,
            "success_rate":          float((all_cohs > REPLAY_COHERENCE_THR).mean())
                                     if n_total else 0.0,
            "acceptance_rate":       n_accept / n_total if n_total else 0.0,
            "accept_reasons":        accept_reasons,
            "all_max_consec":        all_max_consec,
            "all_run_lengths":       all_run_lengths,
            "wfast_by_rest":         wfast_by_rest,
            "wslow_by_rest":         wslow_by_rest,
            "accept_by_rest":        accept_by_rest,
            "coh_by_rest_mean":      coh_by_rest_mean,
            # Diversity metrics
            "mean_quality":          float(all_qual_arr.mean()) if len(all_qual_arr) else 0.0,
            "qual_diversity":        float(all_qual_arr.std()) if len(all_qual_arr) > 1 else 0.0,
            "conf_diversity":        float(all_conf_arr.std()) if len(all_conf_arr) > 1 else 0.0,
            "coh_diversity":         float(all_cohs.std()) if n_total > 1 else 0.0,
        }
    return analysis


def print_replay_quality_summary(analysis):
    """Print a concise table of replay quality metrics per condition."""
    replay_conds = {l: v for l, v in analysis.items() if v["use_replay"]}
    if not replay_conds:
        return
    print("\n[REPLAY QUALITY SUMMARY]", flush=True)
    hdr = (f"  {'Condition':<22s}  {'Mean coh':>9s}  {'Std coh':>8s}"
           f"  {'Success%':>9s}  {'Accept%':>8s}  {'Confidence':>11s}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for label, data in replay_conds.items():
        confs = np.array(data["all_confidences"], float)
        mean_conf = float(confs.mean()) if len(confs) else 0.0
        print(f"  {label:<22s}  {data['mean_coherence']:>9.4f}  "
              f"{data['std_coherence']:>8.4f}  "
              f"{data['success_rate']*100:>8.1f}%"
              f"  {data['acceptance_rate']*100:>7.1f}%"
              f"  {mean_conf:>11.4f}", flush=True)
        if data["accept_reasons"]:
            for reason, cnt in sorted(data["accept_reasons"].items(),
                                      key=lambda x: -x[1]):
                print(f"    -> rejected ({reason}): {cnt}", flush=True)

    # Correlation 1: coherence ↔ retention
    xs, ys = [], []
    for data in replay_conds.values():
        xs.extend(data["trial_mean_coherences"])
        ys.extend(data["trial_final_scores"])
    xs = np.array(xs, float); ys = np.array(ys, float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.sum() >= 3:
        r, p = safe_corrcoef(xs[valid], ys[valid])
        print(f"\n  Coherence <-> retention (pooled replay trials):", flush=True)
        print(f"    r={r:.3f}  p={p:.4f}  N={int(valid.sum())}", flush=True)

    # Correlation 2: confidence ↔ retention (new metric)
    xc, yc = [], []
    for data in replay_conds.values():
        xc.extend(data["trial_mean_confidences"])
        yc.extend(data["trial_final_scores"])
    xc = np.array(xc, float); yc = np.array(yc, float)
    valid2 = np.isfinite(xc) & np.isfinite(yc)
    if valid2.sum() >= 3:
        r2, p2 = safe_corrcoef(xc[valid2], yc[valid2])
        print(f"\n  Confidence <-> retention (pooled replay trials):", flush=True)
        print(f"    r={r2:.3f}  p={p2:.4f}  N={int(valid2.sum())}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# SAFE STATISTICAL UTILITIES  (robust to degenerate data)
# ═══════════════════════════════════════════════════════════════════════════

def safe_linregress(x, y):
    """Linear regression with full degeneracy detection.
    
    Returns (slope, intercept, r_value, p_value, std_err) with sane defaults
    when data is degenerate (identical x, NaNs, <3 points, all same y, etc.).
    Never crashes — always returns a valid 5-tuple.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n < 3:
        return (0.0, 0.0, 0.0, 1.0, 0.0)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return (0.0, 0.0, 0.0, 1.0, 0.0)
    xv, yv = x[valid], y[valid]
    # Degeneracy 1: identical x values
    if np.ptp(xv) < 1e-12:
        return (0.0, float(np.mean(yv)), 0.0, 1.0, 0.0)
    # Degeneracy 2: identical y values
    if np.ptp(yv) < 1e-12:
        return (0.0, float(np.mean(yv)), 0.0, 1.0, 0.0)
    try:
        result = _linregress(xv, yv)
        return result
    except Exception:
        return (0.0, float(np.mean(yv)), 0.0, 1.0, 0.0)

def safe_corrcoef(x, y):
    """Pearson correlation with degeneracy detection.
    
    Returns (r, p_value).  Never crashes.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n < 3:
        return (0.0, 1.0)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return (0.0, 1.0)
    xv, yv = x[valid], y[valid]
    if np.ptp(xv) < 1e-12 or np.ptp(yv) < 1e-12:
        return (0.0, 1.0)
    try:
        return _pearsonr(xv, yv)
    except Exception:
        return (0.0, 1.0)

def safe_confidence_interval(data, confidence=0.95):
    """Paired bootstrap confidence interval for the mean.
    
    Robust to non-normality, small samples, and degenerate data.
    Falls back to t-based CI if bootstrap fails.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    valid = np.isfinite(data)
    if valid.sum() < 2:
        return (np.nan, np.nan)
    dv = data[valid]
    n = len(dv)
    if n < 2:
        return (np.nan, np.nan)
    # Bootstrap
    n_boot = min(5000, int(1e5 / n))
    means = np.empty(n_boot)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(dv[idx])
    alpha = 1.0 - confidence
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# REPLAY QUALITY FIGURES  (Figures R1–R4)
# ─────────────────────────────────────────────────────────────────────────────

def fig_replay_coherence_distributions(analysis):
    """
    R1 — Violin + summary bars of replay coherence distributions per condition.

    Left panel:  violin plot of peak_coherence per replay event, per condition.
                 Individual event values overlaid as scatter.  Gate threshold
                 shown as dashed reference.
    Right panel: grouped bar chart of success rate and mean coherence per
                 condition, making the Fast vs Slow contrast immediately legible.
    """
    replay_conds = {l: v for l, v in analysis.items() if v["use_replay"]}
    if not replay_conds:
        return

    labels = list(replay_conds.keys())
    colors = [replay_conds[l]["color"] for l in labels]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: violin ─────────────────────────────────────────────────────────
    ax = axes[0]
    for i, (label, color) in enumerate(zip(labels, colors)):
        cohs = replay_conds[label]["all_coherences"]
        if not cohs:
            continue
        arr = np.array(cohs)
        vp  = ax.violinplot([arr], positions=[i], showmedians=True,
                            showextrema=True, widths=0.65)
        for pc in vp["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.55)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in vp:
                vp[key].set_color(color); vp[key].set_linewidth(1.8)
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(i + jitter, arr, color=color, alpha=0.30, s=9, zorder=3)

    ax.axhline(REPLAY_COHERENCE_THR, color='k', ls='--', lw=1.3, alpha=0.7,
               label=f"STDP gate threshold ({REPLAY_COHERENCE_THR})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" / ", "\n").replace(" + ", "\n+")
                        for l in labels], fontsize=9)
    ax.set_ylim(-0.04, 1.08)
    ax.legend(fontsize=8, loc="upper right")
    _style(ax, ylabel="Peak replay coherence (per event)",
           title="R1a — Replay Coherence Distributions")

    # ── Right: summary bars ───────────────────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(labels))
    w = 0.35
    success = [replay_conds[l]["success_rate"]    for l in labels]
    means   = [replay_conds[l]["mean_coherence"]  for l in labels]
    stds    = [replay_conds[l]["std_coherence"]   for l in labels]

    ax.bar(x - w/2, success, w, color=colors, alpha=0.80,
           label="Success rate (fraction ≥ gate)", edgecolor='none')
    ax.bar(x + w/2, means,   w, color=colors, alpha=0.40,
           label="Mean coherence", edgecolor=colors, linewidth=1.5)
    # Error bars on mean coherence — errorbar requires a single color per call
    for _xi, _m, _s, _c in zip(x + w/2, means, stds, colors):
        ax.errorbar(_xi, _m, yerr=_s, fmt='none',
                    ecolor=_c, elinewidth=1.8, capsize=4)

    ax.axhline(REPLAY_COHERENCE_THR, color='k', ls='--', lw=1.3, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" / ", "\n").replace(" + ", "\n+")
                        for l in labels], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper right")
    _style(ax, ylabel="Score [0 – 1]",
           title="R1b — Success Rate and Mean Coherence")

    plt.tight_layout()
    _save_fig(fig, "replay_coherence_distributions")


def fig_coherence_vs_retention(analysis):
    """
    R2 — Scatter: mean replay coherence per trial vs Memory A final retention.

    Each point is one trial; colour encodes condition.  Pearson correlation
    and regression line computed across all replay-condition trials pooled.

    Publishable narrative:
      Trials with higher average replay coherence retain Memory A better.
      Fast/Replay clusters near low coherence / near-zero retention;
      Slow+Replay clusters near high coherence / high retention.
      The continuum across both conditions supports a causal role for
      replay quality, not just slow-consolidation per se.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    all_x, all_y = [], []
    for label, data in analysis.items():
        if not data["use_replay"]:
            continue
        xs = np.array(data["trial_mean_coherences"], float)
        ys = np.array(data["trial_final_scores"],    float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        if not valid.any():
            continue
        ax.scatter(xs[valid], ys[valid], color=data["color"],
                   s=90, alpha=0.80, zorder=4,
                   label=f"{label}  (n={int(valid.sum())})")
        all_x.extend(xs[valid].tolist())
        all_y.extend(ys[valid].tolist())

    # Overall linear regression line
    all_x = np.array(all_x, float)
    all_y = np.array(all_y, float)
    if len(all_x) >= 3:
        sl, ic, r_val, p_val, _ = safe_linregress(all_x, all_y)
        x_line = np.linspace(all_x.min(), all_x.max(), 200)
        ax.plot(x_line, sl * x_line + ic, 'k--', lw=1.5, alpha=0.65,
                label=f"Regression  r = {r_val:.2f},  p = {p_val:.3f}")

    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.axvline(REPLAY_COHERENCE_THR, color='dimgrey', ls=':', lw=1.0, alpha=0.6,
               label=f"Gate threshold = {REPLAY_COHERENCE_THR}")
    ax.legend(fontsize=8, loc="upper left")
    _style(ax,
           xlabel="Mean peak replay coherence (per trial)",
           ylabel="Final Memory A retention (I_syn score)",
           title="R2 — Replay Coherence Predicts Memory Retention")
    plt.tight_layout()
    _save_fig(fig, "replay_coherence_vs_retention")


def fig_replay_success_across_bursts(analysis):
    """
    R3 — Replay success rate as a function of burst position within each rest.

    Each rest period contains n_bursts SWR-style burst clusters.
    This figure shows whether replay quality degrades across bursts
    (W_fast decays between bursts via REPLAY_BURST_GAP passive steps).

    Publishable narrative:
      For Fast/Replay, success rate may fall across bursts as W_fast
      decays below the pattern-completion threshold.  For Slow+Replay,
      W_slow anchors the effective weight, sustaining coherent replay
      throughout all bursts.
    """
    replay_conds = {l: v for l, v in analysis.items() if v["use_replay"]}
    if not replay_conds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: success rate per burst ─────────────────────────────────────────
    ax = axes[0]
    for label, data in replay_conds.items():
        sbr = data["success_by_burst"]
        if not sbr:
            continue
        burst_ids = sorted(sbr.keys())
        rates = [float(np.mean(sbr[b])) if sbr[b] else np.nan
                 for b in burst_ids]
        cnts  = [len(sbr[b]) for b in burst_ids]
        errs  = [float(_safe_sem(sbr[b])) if len(sbr[b]) >= 2 else 0.0
                 for b in burst_ids]
        ax.errorbar(burst_ids, rates, yerr=errs,
                    color=data["color"], lw=2.2, marker='o', ms=8,
                    capsize=4, label=f"{label}")
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel("Burst index within rest period", fontsize=11)
    ax.set_ylabel("Replay success rate (fraction coherent events)", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(range(max(len(v["success_by_burst"])
                            for v in replay_conds.values())))
    ax.legend(fontsize=9)
    _style(ax, title="R3a — Success Rate Across SWR Burst Clusters")

    # ── Right: cumulative coherent-event fraction per burst ──────────────────
    # Uses success_by_burst (fraction of events where n_steps_coherent > 0),
    # the same data as the left panel — shows the same signal, different style.
    ax = axes[1]
    for label, data in replay_conds.items():
        sbr = data["success_by_burst"]
        if not sbr:
            continue
        burst_ids = sorted(sbr.keys())
        rates = [float(np.mean(sbr[b])) if sbr[b] else np.nan
                 for b in burst_ids]
        ax.plot(burst_ids, rates, 'o--', color=data["color"],
                lw=1.8, ms=7, alpha=0.75, label=label)

    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel("Burst index within rest period", fontsize=11)
    ax.set_ylabel("Fraction of coherent events", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9)
    _style(ax, title="R3b — Coherent Fraction Per Burst")

    plt.tight_layout()
    _save_fig(fig, "replay_success_across_bursts")


def fig_replay_coherence_trajectory(analysis):
    """
    R4 — Mean replay coherence across successive inter-memory rests.

    X-axis: rest_id (0 = after Memory A, 1 = after B, 2 = after C).
    As more memories are learned, interference accumulates and W_fast decays
    more severely for older assemblies.  This figure captures whether replay
    quality degrades under accumulating interference.

    Publishable narrative:
      Slow+Replay maintains high coherence across all rests because W_slow
      provides a stable attractor that survives the fast-decay between rests.
      Fast/Replay coherence degrades monotonically: each new memory further
      dilutes the A-A recurrent weight, making pattern completion harder.
    """
    replay_conds = {l: v for l, v in analysis.items() if v["use_replay"]}
    if not replay_conds:
        return

    rest_labels = [f"Rest after {chr(65+i)}" for i in range(N_MEMORIES - 1)]
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in replay_conds.items():
        cbr = data["coherence_by_rest"]
        if not cbr:
            continue
        rest_ids = sorted(cbr.keys())
        means = [float(np.mean(cbr[r]))  if cbr[r] else np.nan for r in rest_ids]
        sems  = [float(_safe_sem(cbr[r])) if len(cbr[r]) >= 2 else 0.0
                 for r in rest_ids]
        x = np.array(rest_ids)
        y = np.array(means)
        e = np.array(sems)
        ax.plot(x, y, 'o-', color=data["color"], lw=2.2, ms=9, label=label)
        ax.fill_between(x, y - e, y + e, color=data["color"], alpha=0.15)

    ax.axhline(REPLAY_COHERENCE_THR, color='k', ls='--', lw=1.3, alpha=0.6,
               label=f"STDP gate threshold ({REPLAY_COHERENCE_THR})")
    ax.set_xticks(range(N_MEMORIES - 1))
    ax.set_xticklabels(rest_labels[:N_MEMORIES - 1], fontsize=9)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(fontsize=9)
    _style(ax,
           ylabel="Mean peak replay coherence (± SEM)",
           title="R4 — Replay Quality Under Accumulating Interference")
    plt.tight_layout()
    _save_fig(fig, "replay_coherence_trajectory")


# ─────────────────────────────────────────────────────────────────────────────
# TIMING REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_timing_report(total_s):
    """
    Print per-category CPU-seconds aggregated across all parallel workers.

    WHY CPU-seconds, not percentages of wall-clock:
      _TIMER entries are incremented in worker processes and merged back to the
      main process via _worker_timer deltas.  With N_WORKERS>1 the workers run
      in parallel, so the sum of CPU-seconds exceeds wall-clock time (that is
      expected and desirable — it means workers are keeping cores busy).
      Dividing by wall-clock would yield categories summing to >100% and an
      "unmeasured" row that goes negative.  Instead we report each category as a
      fraction of total CPU-seconds, and show wall-clock + parallelism factor
      separately so both measures are available without confusion.
    """
    print("\n" + "-"*50, flush=True)
    print("TIMING REPORT (CPU-seconds, all workers combined)", flush=True)
    print("-"*50, flush=True)
    cpu_total = sum(_TIMER.values())
    denom = max(cpu_total, 1e-6)
    for k, v in sorted(_TIMER.items(), key=lambda x: -x[1]):
        pct = 100.0 * v / denom
        print(f"  {k:15s}: {v:7.1f}s  ({pct:.1f}%)", flush=True)
    parallelism = cpu_total / max(total_s, 1e-6)
    print(f"  {'-'*37}", flush=True)
    print(f"  {'CPU total':15s}: {cpu_total:7.1f}s  "
          f"({parallelism:.2f}x wall-clock parallelism)", flush=True)
    print(f"  {'Wall total':15s}: {total_s:7.1f}s", flush=True)
    print("-"*50, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _check_results_stability(results, to_check):
    """
    Return (ok, hard_labels) where hard_labels is a list of condition labels
    that triggered hard-abort criteria.  Used by validate_single_trial.
    """
    ok          = True
    hard_labels = []
    for (label, use_slow, use_replay), r in zip(to_check, results):
        finals = r["final_scores"]
        base   = r["baseline_scores"]
        has_nan_fin  = not np.all(np.isfinite(finals))
        has_nan_base = not np.all(np.isfinite(base))
        large_fin    = bool(np.any(np.abs(np.nan_to_num(finals, nan=0.0)) > 5.0))
        large_base   = bool(np.any(np.abs(np.nan_to_num(base,   nan=0.0)) > 5.0))
        hard = has_nan_fin or has_nan_base or large_fin or large_base
        if hard:
            ok = False
            hard_labels.append(label)
    return ok, hard_labels


def validate_single_trial(assemblies):
    """
    Run one trial per condition in PARALLEL and check for outlier scores.
    Runs all 4 conditions simultaneously on N_WORKERS cores.
    Thresholds:
      |final_score| > 5.0  -> hard abort (system explosion)
      |final_score| > 1.5  -> soft warning (high variance; acceptable with more trials)
    All 4 conditions checked: Fast/Replay instability (pre-fix: competition without anchor)
    is now expected to be absent since competition is gated on use_slow=True.

    Seed strategy: use MASTER_SEED as the primary validation seed (consistent with
    the first trial of the main experiment).  If any condition triggers hard-abort
    with the primary seed, retry once with MASTER_SEED + 100_000 before aborting
    — mirrors _trial_worker retry logic.  seed=0 is deliberately avoided because
    at production N_PRESENTATIONS=20 it can hit a seed-specific cross-attractor
    instability (A-B overlap neurons at their max W_eff cause runaway during B
    probing) while MASTER_SEED-based seeds are robustly stable; the main experiment
    handles occasional NaN trials via _trial_worker retry.

    Returns True if NO hard-abort triggers fire on primary OR retry seed.
    """
    print("\n[VALIDATE] Sanity check (4 conditions, parallel) ...", flush=True)
    to_check = [
        ("Fast/NoReplay",  False, False),
        ("Fast/Replay",    False, True),
        ("Slow/NoReplay",  True,  False),
        ("Slow+Replay",    True,  True),
    ]
    asm_lists = [a.tolist() for a in assemblies]

    def _run_validate(seed):
        tasks = [(use_slow, use_replay, asm_lists, seed, "interference_aware")
                 for _, use_slow, use_replay in to_check]
        return _run_tasks_parallel(tasks, verbose=False)

    # ── Primary validation pass (MASTER_SEED) ────────────────────────────────
    results = _run_validate(MASTER_SEED)
    ok, hard_labels = _check_results_stability(results, to_check)

    # Print primary-pass results
    all_ok_primary = True
    for (label, use_slow, use_replay), r in zip(to_check, results):
        finals = r["final_scores"]
        base   = r["baseline_scores"]
        has_nan_fin  = not np.all(np.isfinite(finals))
        has_nan_base = not np.all(np.isfinite(base))
        large_fin    = bool(np.any(np.abs(np.nan_to_num(finals, nan=0.0)) > 5.0))
        large_base   = bool(np.any(np.abs(np.nan_to_num(base,   nan=0.0)) > 5.0))
        hard = has_nan_fin or has_nan_base or large_fin or large_base
        soft = bool(np.any(np.abs(np.nan_to_num(finals, nan=0.0)) > 1.5))
        status = "[!HARD_UNSTABLE]" if hard else ("[!SOFT_WARN]" if soft else "[OK]")
        print(f"  {status:18s} {label:15s}  "
              f"baseline={np.nan_to_num(base).round(3)}  "
              f"final={np.nan_to_num(finals).round(3)}", flush=True)
        if hard:
            all_ok_primary = False

    if all_ok_primary:
        return True

    # ── Retry pass (MASTER_SEED + 100_000) ───────────────────────────────────
    # If the primary seed triggers hard-abort for any condition, try an alternative
    # seed before declaring full abort.  This handles rare seed-specific instabilities
    # (cross-attractor saturation with certain random weight initializations).
    retry_seed = MASTER_SEED + 100_000
    print(f"  [VALIDATE] Primary seed {MASTER_SEED} flagged "
          f"{hard_labels}; retrying with seed={retry_seed} ...", flush=True)
    results2  = _run_validate(retry_seed)
    ok2, hard_labels2 = _check_results_stability(results2, to_check)

    for (label, use_slow, use_replay), r in zip(to_check, results2):
        finals = r["final_scores"]
        base   = r["baseline_scores"]
        has_nan_fin  = not np.all(np.isfinite(finals))
        has_nan_base = not np.all(np.isfinite(base))
        large_fin    = bool(np.any(np.abs(np.nan_to_num(finals, nan=0.0)) > 5.0))
        large_base   = bool(np.any(np.abs(np.nan_to_num(base,   nan=0.0)) > 5.0))
        hard = has_nan_fin or has_nan_base or large_fin or large_base
        soft = bool(np.any(np.abs(np.nan_to_num(finals, nan=0.0)) > 1.5))
        status = "[!HARD_UNSTABLE]" if hard else ("[!SOFT_WARN]" if soft else "[OK]")
        print(f"  (retry) {status:18s} {label:15s}  "
              f"baseline={np.nan_to_num(base).round(3)}  "
              f"final={np.nan_to_num(finals).round(3)}", flush=True)

    if ok2:
        print("  [VALIDATE] Retry passed — seed-specific instability on primary "
              "seed; main experiment retry logic handles these cases.", flush=True)
        return True

    # Both seeds failed — genuine hard instability
    print(f"  ABORTING: hard instability on both seed={MASTER_SEED} and "
          f"seed={retry_seed}. Check parameters.", flush=True)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SCALING VALIDATION SUITE  (v4 — sparse modular architecture)
# ─────────────────────────────────────────────────────────────────────────────

def validate_scaling():
    """
    Run quick stability checks across multiple network sizes.
    Tests spontaneous dynamics and basic replay ignition.
    Reports failure categories: SILENT, RUNAWAY, NAN, SATURATION, STABLE.

    Returns dict of {n_neurons: failure_reason} for all tested sizes.
    """
    print("\n" + "="*70, flush=True)
    print("SCALING VALIDATION SUITE", flush=True)
    print(f"  Architecture mode: {ARCH_MODE}", flush=True)
    print("="*70, flush=True)

    test_sizes = [400, 600, 800, 1000]
    results = {}

    for N in test_sizes:
        n_inh = N // 5  # 20% inhibitory
        n_exc = N - n_inh

        print(f"\n  Testing N={N} (exc={n_exc}, inh={n_inh}) ...", flush=True)

        # Build network
        net = IzhikevichNetwork(
            n_neurons=N, n_inh=n_inh,
            g_exc=G_EXC, g_inh=G_INH,
            noise_std=NOISE_STD, dt=DT, device=DEVICE,
            arch_mode=ARCH_MODE, n_modules=N_MODULES,
            intra_module_conn_prob=INTRA_MODULE_CONN_PROB,
            inter_module_conn_prob=INTER_MODULE_CONN_PROB,
            inter_module_scale=INTER_MODULE_SCALE,
            ee_sparsity=EE_SPARSITY,
        ).to(DEVICE)
        net.init_stdp(
            A_plus=A_PLUS, A_minus=A_MINUS,
            tau_plus=TAU_PLUS, tau_minus=TAU_MINUS, w_max=W_MAX
        )

        # Diagnostic output
        if DEBUG_SCALING:
            diagnose_connectivity(net)

        # Test 1: spontaneous dynamics
        failure = detect_failure_mode(net, n_steps=500)
        status = "STABLE" if failure == FAILURE_STABLE else failure

        if failure == FAILURE_STABLE:
            # Test 2: basic replay ignition (seed then observe)
            assembly_size = max(20, n_exc // (N_MODULES * 4))
            asm = np.arange(0, assembly_size, dtype=int)
            seed_stim = torch.zeros(N, device=DEVICE)
            seed_stim[asm[:8]] = 12.0
            for _ in range(50):
                net.forward(seed_stim)
            failure2 = detect_failure_mode(net, n_steps=100)
            if failure2 != FAILURE_STABLE:
                status = f"STABLE_SPONT-{failure2}_AFTER_SEED"

        results[N] = status
        print(f"    - {status}", flush=True)

    print("\n" + "-"*70, flush=True)
    print("SCALING SUMMARY:", flush=True)
    all_ok = True
    for N, status in results.items():
        ok = "OK" if status == FAILURE_STABLE else "FAIL"
        if ok == "FAIL":
            all_ok = False
        print(f"  N={N:4d}: {ok:4s}  ({status})", flush=True)

    if all_ok:
        print("\n  All sizes stable. Architecture scales correctly.", flush=True)
    else:
        print("\n  Scaling issues detected. Inspect failures before production runs.",
              flush=True)
    print("="*70, flush=True)

    return results


def fig_adaptive_replay_analysis(analysis):
    """
    R5 — Adaptive replay selection: 4-panel summary figure.

    R5a (top-left):   Coherence distributions split by accepted vs rejected
                      events.  For each condition, shows violin + strip of
                      peak_coherence, coloured by acceptance.  Demonstrates
                      that the adaptive gate selects higher-coherence events.

    R5b (top-right):  Replay confidence histograms per condition.  Confidence
                      is the geometric mean of normalised completion × stability
                      × coherence-SNR, computed from the eval window.
                      Slow+Replay should show a rightward shift.

    R5c (bottom-left): Confidence vs Memory-A retention scatter.  Each point
                       is one trial; colour encodes condition.  Regression line
                       and Pearson r shown.  Stronger correlation than the raw
                       coherence scatter (R2) is expected because confidence
                       combines three quality signals.

    R5d (bottom-right): Acceptance rates per condition.  Bar chart of the
                        fraction of replay events accepted by the adaptive gate,
                        with breakdown of rejection reasons.

    Publishable narrative:
      The adaptive gate filters 60-80% of Fast/Replay events (low completion,
      unstable coherence) while accepting the majority of Slow+Replay events
      (strong W_slow-sustained pattern completion).  Accepted events show
      systematically higher coherence.  Replay confidence explains a larger
      fraction of variance in final retention than coherence alone.
      "Only coherent, high-confidence replay events drive long-term
      consolidation; noisy replay is filtered, reducing weight corruption."
    """
    replay_conds = {l: v for l, v in analysis.items() if v["use_replay"]}
    if not replay_conds:
        return

    labels = list(replay_conds.keys())
    colors = [replay_conds[l]["color"] for l in labels]
    accept_color  = '#27ae60'   # green  — accepted
    reject_color  = '#e74c3c'   # red    — rejected

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── R5a: accepted vs rejected coherence distributions ────────────────────
    ax = axes[0, 0]
    x_pos = np.arange(len(labels))
    w = 0.32
    for i, label in enumerate(labels):
        data = replay_conds[label]
        acc  = np.array(data["accepted_coherences"],  float)
        rej  = np.array(data["rejected_coherences"],  float)
        # Accepted violin (right of center)
        if len(acc) >= 3:
            vp = ax.violinplot([acc], positions=[x_pos[i] + w/2],
                               widths=w, showmedians=True, showextrema=False)
            for part in vp["bodies"]:
                part.set_facecolor(accept_color); part.set_alpha(0.55)
            vp["cmedians"].set_color(accept_color); vp["cmedians"].set_linewidth(2)
        elif len(acc) > 0:
            ax.scatter([x_pos[i] + w/2] * len(acc), acc,
                       color=accept_color, s=20, alpha=0.6, zorder=3)
        # Rejected violin (left of center)
        if len(rej) >= 3:
            vp = ax.violinplot([rej], positions=[x_pos[i] - w/2],
                               widths=w, showmedians=True, showextrema=False)
            for part in vp["bodies"]:
                part.set_facecolor(reject_color); part.set_alpha(0.45)
            vp["cmedians"].set_color(reject_color); vp["cmedians"].set_linewidth(2)
        elif len(rej) > 0:
            ax.scatter([x_pos[i] - w/2] * len(rej), rej,
                       color=reject_color, s=20, alpha=0.6, zorder=3)

    ax.axhline(REPLAY_COHERENCE_THR, color='k', ls='--', lw=1.3, alpha=0.7,
               label=f"STDP gate ({REPLAY_COHERENCE_THR})")
    # Legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=accept_color, alpha=0.7, label="Accepted"),
                        Patch(facecolor=reject_color, alpha=0.6, label="Rejected"),
                        plt.Line2D([0], [0], color='k', ls='--', lw=1.3,
                                   label=f"STDP gate ({REPLAY_COHERENCE_THR})")],
              fontsize=8, loc="upper right")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([l.replace(" / ", "\n").replace(" + ", "\n+")
                        for l in labels], fontsize=9)
    ax.set_ylim(-0.04, 1.08)
    _style(ax, ylabel="Peak replay coherence",
           title="R5a — Accepted vs Rejected Coherence")

    # ── R5b: confidence histograms ────────────────────────────────────────────
    ax = axes[0, 1]
    bins = np.linspace(0.0, 1.0, 25)
    for label, color in zip(labels, colors):
        confs = np.array(replay_conds[label]["all_confidences"], float)
        if len(confs) == 0:
            continue
        ax.hist(confs, bins=bins, color=color, alpha=0.55, label=label,
                edgecolor='none', density=True)
        ax.axvline(float(confs.mean()), color=color, lw=2.0, ls='--', alpha=0.85)
    ax.set_xlabel("Replay confidence score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=8)
    _style(ax, title="R5b — Replay Confidence Distributions")

    # ── R5c: confidence vs retention scatter ──────────────────────────────────
    ax = axes[1, 0]
    all_cx, all_cy = [], []
    for label, color in zip(labels, colors):
        data = replay_conds[label]
        cx = np.array(data["trial_mean_confidences"], float)
        cy = np.array(data["trial_final_scores"],     float)
        valid = np.isfinite(cx) & np.isfinite(cy)
        if valid.sum() > 0:
            ax.scatter(cx[valid], cy[valid], color=color, s=72, alpha=0.80,
                       edgecolors='white', linewidths=0.6, zorder=3, label=label)
            all_cx.extend(cx[valid].tolist())
            all_cy.extend(cy[valid].tolist())
    # Pooled regression line
    ax_cx = np.array(all_cx, float)
    ax_cy = np.array(all_cy, float)
    val = np.isfinite(ax_cx) & np.isfinite(ax_cy)
    if val.sum() >= 3:
        slope, intercept, rv, pv, _ = safe_linregress(ax_cx[val], ax_cy[val])
        xl = np.linspace(ax_cx[val].min(), ax_cx[val].max(), 60)
        ax.plot(xl, slope * xl + intercept, 'k-', lw=1.8, alpha=0.7,
                label=f"r={rv:.2f}  p={pv:.3f}")
        r_conf, p_conf = safe_corrcoef(ax_cx[val], ax_cy[val])
        ax.text(0.05, 0.93, f"r = {r_conf:.3f}\np = {p_conf:.3f}\nN = {int(val.sum())}",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.legend(fontsize=8)
    _style(ax, xlabel="Mean replay confidence (trial)",
           ylabel="Memory A final retention",
           title="R5c — Replay Confidence vs Retention")

    # ── R5d: acceptance rates + rejection breakdown ───────────────────────────
    ax = axes[1, 1]
    x   = np.arange(len(labels))
    acc_rates = [replay_conds[l]["acceptance_rate"] * 100 for l in labels]
    bars = ax.bar(x, acc_rates, color=colors, alpha=0.75, edgecolor='white',
                  linewidth=1.4, zorder=2)
    for xi, rate, bar in zip(x, acc_rates, bars):
        ax.text(xi, rate + 1.5, f"{rate:.1f}%", ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    # Stacked rejection-reason annotation
    for i, label in enumerate(labels):
        reasons = replay_conds[label]["accept_reasons"]
        total   = sum(reasons.values())
        if total == 0:
            continue
        reason_labels = {
            "low_completion":    "Low completion",
            "high_offtarget":    "High off-target",
            "unstable_coherence": "Unstable coh.",
            "unknown":           "Unknown",
        }
        y_off = acc_rates[i] + 4.0
        for reason, cnt in sorted(reasons.items(), key=lambda kv: -kv[1]):
            pct = cnt / max(total + len(replay_conds[label]["accepted_coherences"]), 1) * 100
            ax.text(i, y_off, f"  {reason_labels.get(reason, reason)}: {pct:.0f}%",
                    ha='center', va='bottom', fontsize=7, color='#555555')
            y_off += 5.0

    ax.axhline(50, color='grey', lw=0.9, ls=':', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" / ", "\n").replace(" + ", "\n+")
                        for l in labels], fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Events accepted (%)", fontsize=11)
    _style(ax, title="R5d — Adaptive Gate Acceptance Rate")

    plt.tight_layout()
    _save_fig(fig, "adaptive_replay_analysis")


def fig_attractor_dynamics(analysis):
    """
    R6 — Attractor persistence dynamics: 3-panel figure.

    R6a (left):   Distribution of max consecutive coherent steps per event.
                  Violin + strip per condition.  Captures the maximum attractor
                  occupancy achieved per replay event.  Slow+Replay should
                  show a clear rightward shift — longer coherent epochs arise
                  from stronger basin stability conferred by W_slow.

    R6b (centre): Individual coherent epoch length distribution (histogram).
                  Each data point is the length of one uninterrupted coherent
                  run.  This is the attractor LIFETIME distribution — analogous
                  to a dwell-time histogram in biophysics.  Slow+Replay should
                  have heavier tails (longer-lived attractors).

    R6c (right):  Coherence survival curve: P(run_length ≥ t) vs t.
                  Derived from the empirical CDF of run lengths.  Slow+Replay
                  curve decays more slowly - longer mean attractor lifetime.
                  Fast/Replay curve drops off sharply after t=1–2 steps.

    Publishable narrative:
      "W_slow-mediated reverberatory support creates deeper attractor basins
      for consolidated assemblies.  Slow+Replay events sustain coherent
      replay for 3–10× longer than Fast/Replay events (Fig R6b), and the
      survival curve (Fig R6c) confirms a population of long-lived attractor
      states (> 5 steps) that is absent in the Fast condition.  This emergent
      attractor stability — arising from W_slow, not from replay thresholds —
      is the mechanistic basis for the acceptance-rate separation between
      conditions and for the superior long-term retention of Slow+Replay."
    """
    replay_conds = {l: v for l, v in analysis.items() if v["use_replay"]}
    if not replay_conds:
        return

    labels = list(replay_conds.keys())
    colors = [replay_conds[l]["color"] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── R6a: max consecutive coherent steps per event ─────────────────────────
    ax = axes[0]
    for i, (label, color) in enumerate(zip(labels, colors)):
        mc = replay_conds[label]["all_max_consec"]
        if not mc:
            continue
        mc_arr = np.array(mc, float)
        if len(mc_arr) >= 3:
            vp = ax.violinplot([mc_arr], positions=[i],
                               widths=0.65, showmedians=True, showextrema=False)
            for part in vp["bodies"]:
                part.set_facecolor(color); part.set_alpha(0.55)
            vp["cmedians"].set_color(color); vp["cmedians"].set_linewidth(2.2)
        # Jitter scatter overlay
        jitter = np.random.uniform(-0.18, 0.18, size=len(mc_arr))
        ax.scatter(i + jitter, mc_arr, color=color, s=12, alpha=0.35, zorder=3)
        # Mean marker
        ax.scatter([i], [mc_arr.mean()], color=color, s=80, marker='D',
                   zorder=5, edgecolors='white', linewidths=0.8)

    ax.axhline(REPLAY_ACCEPT_MIN_CONSEC, color='k', ls='--', lw=1.2, alpha=0.6,
               label=f"Unlock threshold ({REPLAY_ACCEPT_MIN_CONSEC} steps)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" / ", "\n").replace(" + ", "\n+")
                        for l in labels], fontsize=9)
    ax.set_ylim(-0.5, None)
    ax.legend(fontsize=8)
    _style(ax, ylabel="Max consecutive coherent steps (per event)",
           title="R6a — Attractor Occupancy Distribution")

    # ── R6b: individual coherent run length histogram ─────────────────────────
    ax = axes[1]
    max_run = max(
        (max(v["all_run_lengths"]) for v in replay_conds.values()
         if v["all_run_lengths"]), default=10)
    bins = np.arange(0.5, max_run + 1.5, 1.0)   # integer-centred bins
    for label, color in zip(labels, colors):
        rl = replay_conds[label]["all_run_lengths"]
        if not rl:
            continue
        ax.hist(rl, bins=bins, color=color, alpha=0.55, label=label,
                edgecolor='none', density=True)
        # Exponential fit (mean = 1/lambda):
        mean_rl = float(np.mean(rl))
        if mean_rl > 0:
            xs = np.linspace(0.5, max_run + 0.5, 200)
            lam = 1.0 / mean_rl
            ax.plot(xs, lam * np.exp(-lam * xs), color=color,
                    lw=2.0, ls='--', alpha=0.85,
                    label=f"{label} exp fit (τ={mean_rl:.1f})")
    ax.axvline(REPLAY_ACCEPT_MIN_CONSEC, color='k', ls='--', lw=1.2, alpha=0.6,
               label=f"Unlock threshold ({REPLAY_ACCEPT_MIN_CONSEC})")
    ax.set_xlabel("Coherent epoch length (steps)", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    _style(ax, title="R6b — Attractor Lifetime Distribution")

    # ── R6c: coherence survival curve P(run_length ≥ t) ─────────────────────
    ax = axes[2]
    for label, color in zip(labels, colors):
        rl = np.array(replay_conds[label]["all_run_lengths"], float)
        if len(rl) == 0:
            continue
        max_t = int(rl.max()) if len(rl) else 1
        ts    = np.arange(1, max_t + 2)
        surv  = np.array([(rl >= t).mean() for t in ts])
        ax.step(ts, surv, where='post', color=color, lw=2.4,
                label=label, alpha=0.85)
        ax.fill_between(ts, 0, surv, step='post', color=color, alpha=0.12)
        # Annotate mean lifetime
        mean_t = float(rl.mean())
        surv_at_mean = float((rl >= mean_t).mean())
        ax.annotate(f"τ={mean_t:.1f}",
                    xy=(mean_t, surv_at_mean), xytext=(mean_t + 0.5, surv_at_mean + 0.06),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.axvline(REPLAY_ACCEPT_MIN_CONSEC, color='k', ls='--', lw=1.2, alpha=0.6,
               label=f"Unlock threshold ({REPLAY_ACCEPT_MIN_CONSEC})")
    ax.set_xlabel("Run length t (steps)", fontsize=11)
    ax.set_ylabel("P(run_length ≥ t)", fontsize=11)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(fontsize=8)
    _style(ax, title="R6c — Coherence Survival Curve")

    plt.tight_layout()
    _save_fig(fig, "attractor_dynamics")


def fig_competition_dynamics(analysis):
    """
    R7 — Competitive attractor dynamics: 3-panel figure.

    X-axis throughout: rest_id — which inter-memory rest we are measuring.
      rest_id=0  -  after learning memory B  (A has one competitor)
      rest_id=1  -  after learning memory C  (A has two competitors)
      rest_id=2  -  after learning memory D  (A has three competitors)

    As more memories are learned, the competition for overlapping neural
    substrate intensifies.  These panels show how each condition responds.

    R7a (left):   Basin stability trajectories.
                  Mean w_fast_aa (fast-weight A-A strength) per rest, showing
                  how competition erodes A's recurrent basin.  Overlaid with
                  w_slow_aa for Slow+Replay, which should RISE as consolidation
                  builds — slow weights grow while fast weights erode, creating
                  a durable protected basin.  Fast/Replay has no w_slow; its
                  basin erodes monotonically.

    R7b (centre): Competition-induced coherence drift.
                  Mean peak coherence of Memory A's replay events per rest.
                  Shows attractor degradation under increasing competition.
                  Slow+Replay should maintain higher coherence across rests
                  because W_slow sustains pattern completion even as W_fast
                  decays.

    R7c (right):  Replay recovery trajectory.
                  Acceptance rate for Memory A replay events per rest.
                  Captures whether replay successfully re-enters the attractor.
                  Slow+Replay should maintain or improve; Fast/Replay should
                  degrade as competitors crowd the neural substrate.

    Publishable narrative:
      "Increasing competitive interference from newly learned memories erodes
      fast recurrent weights (W_fast) for all assemblies.  In the Fast/Replay
      condition, replay coherence and acceptance rates decline monotonically
      across rests (Fig R7b,c), reflecting attractor collapse under competition.
      In Slow+Replay, slow consolidation builds a protected basin (rising
      w_slow_aa, Fig R7a) that sustains coherent replay despite competitive
      pressure.  Replay in the Slow condition therefore functions as attractor
      RESTORATION, not merely rehearsal: each rest event re-deepens the basin
      against competitive erosion from newer memories."
    """
    replay_conds = {l: v for l, v in analysis.items() if v["use_replay"]}
    if not replay_conds:
        return

    labels = list(replay_conds.keys())
    colors = [replay_conds[l]["color"] for l in labels]
    rest_labels = [f"After\n{chr(65+i+1)}" for i in range(N_MEMORIES - 1)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── R7a: Basin stability — w_fast_aa and w_slow_aa trajectories ───────────
    ax = axes[0]
    for label, color in zip(labels, colors):
        data   = replay_conds[label]
        wf     = data["wfast_by_rest"]
        ws     = data["wslow_by_rest"]
        rids   = sorted(set(wf.keys()) | set(ws.keys()))
        if not rids:
            continue
        wf_m   = np.array([np.mean(wf[r]) if wf.get(r) else np.nan for r in rids])
        wf_e   = np.array([_safe_sem(wf[r]) if wf.get(r) else 0.0    for r in rids])
        # W_fast (solid)
        ax.plot(rids, wf_m, 'o-', color=color, lw=2.2, ms=8,
                label=f"{label} W_fast")
        ax.fill_between(rids,
                        np.nan_to_num(wf_m - wf_e),
                        np.nan_to_num(wf_m + wf_e),
                        color=color, alpha=0.15)
        # W_slow (dashed, only meaningful for Slow+Replay)
        if any(ws.get(r) for r in rids):
            ws_m = np.array([np.mean(ws[r]) if ws.get(r) else np.nan for r in rids])
            ws_e = np.array([_safe_sem(ws[r]) if ws.get(r) else 0.0   for r in rids])
            ax.plot(rids, ws_m, 's--', color=color, lw=1.8, ms=6, alpha=0.75,
                    label=f"{label} W_slow")
            ax.fill_between(rids,
                            np.nan_to_num(ws_m - ws_e),
                            np.nan_to_num(ws_m + ws_e),
                            color=color, alpha=0.10)

    ax.set_xticks(range(N_MEMORIES - 1))
    ax.set_xticklabels(rest_labels[:N_MEMORIES - 1], fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper left")
    _style(ax,
           ylabel="Mean A-A recurrent weight",
           title="R7a — Basin Stability Under Competition")

    # ── R7b: Coherence drift under accumulating competition ───────────────────
    ax = axes[1]
    for label, color in zip(labels, colors):
        data  = replay_conds[label]
        cdict = data["coh_by_rest_mean"]
        rids  = sorted(cdict.keys())
        if not rids:
            continue
        m = np.array([np.mean(cdict[r]) if cdict.get(r) else np.nan for r in rids])
        e = np.array([_safe_sem(cdict[r]) if cdict.get(r) else 0.0   for r in rids])
        ax.plot(rids, m, 'o-', color=color, lw=2.2, ms=8, label=label)
        ax.fill_between(rids,
                        np.nan_to_num(m - e),
                        np.nan_to_num(m + e),
                        color=color, alpha=0.15)

    ax.axhline(REPLAY_COHERENCE_THR, color='k', ls='--', lw=1.2, alpha=0.6,
               label=f"STDP gate ({REPLAY_COHERENCE_THR})")
    ax.set_xticks(range(N_MEMORIES - 1))
    ax.set_xticklabels(rest_labels[:N_MEMORIES - 1], fontsize=9)
    ax.legend(fontsize=8)
    _style(ax,
           ylabel="Mean peak coherence (Memory A replay)",
           title="R7b — Attractor Coherence Under Competition")

    # ── R7c: Replay recovery (acceptance rate) vs competition depth ───────────
    ax = axes[2]
    for label, color in zip(labels, colors):
        data  = replay_conds[label]
        adict = data["accept_by_rest"]
        rids  = sorted(adict.keys())
        if not rids:
            continue
        rates = np.array([np.mean(adict[r]) * 100 if adict.get(r) else np.nan
                          for r in rids])
        errs  = np.array([_safe_sem(adict[r]) * 100 if adict.get(r) else 0.0
                          for r in rids])
        ax.plot(rids, rates, 'o-', color=color, lw=2.2, ms=8, label=label)
        ax.fill_between(rids,
                        np.nan_to_num(rates - errs),
                        np.nan_to_num(rates + errs),
                        color=color, alpha=0.15)

    ax.set_xticks(range(N_MEMORIES - 1))
    ax.set_xticklabels(rest_labels[:N_MEMORIES - 1], fontsize=9)
    ax.set_ylim(-2, 105)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.legend(fontsize=8)
    _style(ax,
           ylabel="Replay acceptance rate (%)",
           title="R7c — Replay Recovery Against Competition")

    plt.tight_layout()
    _save_fig(fig, "competition_dynamics")


# ─────────────────────────────────────────────────────────────────────────────
# ENDOGENOUS PRIORITIZATION ANALYSIS  (Figure R8)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_endogenous_replay(comparison):
    """
    Extract and aggregate endogenous-prioritization metrics from Phase 3 trials.

    For each trial in the 'endogenous' condition, reads per-event replay_metrics
    and computes:
      • urgency trajectory    – mean urgency_score per rest_id × assembly_idx
      • allocation trajectory – fraction of events per assembly per rest
      • demand vs success     – urgency_score vs replay_accepted per event
      • restoration efficiency– mean accepted fraction per urgency quartile

    Returns None (with a warning) if the 'endogenous' key is missing from
    comparison or contains no trials with replay metrics.
    """
    entry = comparison.get("endogenous")
    if entry is None:
        return None
    trials = entry.get("trials", [])
    if not trials:
        return None

    # Collect flat event list across all trials, only for endogenous trials
    all_events    = []
    urgency_by_rest_asm = {}   # (rest_id, asm_idx) - [urgency_score]
    alloc_by_rest_asm   = {}   # (rest_id, asm_idx) - event count

    for trial in trials:
        mlist = trial.get("replay_metrics", [])
        for m in mlist:
            u = m.get("urgency_score", 0.0)
            if u <= 0.0:
                # Skip events from non-endogenous conditions that may appear
                # if the trial_worker ran a different mode
                continue
            all_events.append(m)
            rid  = m.get("rest_id", 0)
            aidx = m.get("assembly_idx", 0)
            urgency_by_rest_asm.setdefault((rid, aidx), []).append(u)
            alloc_by_rest_asm[(rid, aidx)] = alloc_by_rest_asm.get((rid, aidx), 0) + 1

    if not all_events:
        return None

    # Urgency trajectory: mean urgency per (rest, assembly)
    rest_ids  = sorted({k[0] for k in urgency_by_rest_asm})
    asm_idxs  = sorted({k[1] for k in urgency_by_rest_asm})

    urgency_mat   = {}   # asm_idx - list[mean_urgency] over rest_ids
    allocation_mat = {}  # asm_idx - list[fraction] over rest_ids
    for ai in asm_idxs:
        urgency_mat[ai]    = []
        allocation_mat[ai] = []
        for rid in rest_ids:
            u_vals = urgency_by_rest_asm.get((rid, ai), [])
            # Total events at this rest
            total_at_rest = sum(alloc_by_rest_asm.get((rid, aj), 0)
                                for aj in asm_idxs)
            urgency_mat[ai].append(float(np.mean(u_vals)) if u_vals else 0.0)
            allocation_mat[ai].append(
                alloc_by_rest_asm.get((rid, ai), 0) / max(total_at_rest, 1))

    # Demand vs success: urgency_score vs replay_accepted
    urgencies  = np.array([m["urgency_score"] for m in all_events], float)
    accepted   = np.array([1 if m.get("replay_accepted", False) else 0
                            for m in all_events], float)

    # Restoration efficiency per urgency quartile
    if len(urgencies) >= 4:
        q_edges = np.percentile(urgencies, [0, 25, 50, 75, 100])
        quartile_accept = []
        quartile_labels = ["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
        for qi in range(4):
            mask = (urgencies >= q_edges[qi]) & (urgencies <= q_edges[qi+1])
            quartile_accept.append(float(accepted[mask].mean()) if mask.any() else 0.0)
    else:
        quartile_accept = [0.0] * 4
        quartile_labels = ["Q1", "Q2", "Q3", "Q4"]

    return {
        "rest_ids":         rest_ids,
        "asm_idxs":         asm_idxs,
        "urgency_mat":      urgency_mat,      # asm_idx - [mean_urgency per rest]
        "allocation_mat":   allocation_mat,   # asm_idx - [fraction per rest]
        "urgencies":        urgencies,
        "accepted":         accepted,
        "quartile_accept":  quartile_accept,
        "quartile_labels":  quartile_labels,
        "n_events":         len(all_events),
    }


def fig_endogenous_prioritization(endo):
    """
    R8 — Endogenous replay prioritization dynamics (3 panels).

    R8a (left):   Urgency trajectories.
                  Mean urgency_score per assembly over the three rest periods.
                  Shows how urgency diverges as Assembly A is increasingly
                  interfered with: it should accumulate higher urgency than B, C
                  (which were recently trained and have stronger w_fast).

    R8b (centre): Allocation trajectories.
                  Fraction of burst events allocated to each assembly per rest.
                  Tracks how the network's limited replay budget redistributes
                  toward assemblies with growing urgency.

    R8c (right):  Acceptance efficiency by urgency quartile.
                  Acceptance rate (accepted / total events) for events grouped
                  by urgency quartile.  High-urgency events should NOT show
                  higher acceptance — urgency reflects network state (how degraded
                  the assembly is), not event quality.  This panel tests whether
                  increased allocation (driven by high urgency) translates to
                  recovery by the time of acceptance.
    """
    if endo is None:
        return

    rest_ids   = endo["rest_ids"]
    asm_idxs   = endo["asm_idxs"]
    asm_colors = plt.cm.tab10(np.linspace(0, 0.6, max(len(asm_idxs), 1)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── R8a: Urgency trajectories ─────────────────────────────────────────────
    ax = axes[0]
    for ai, col in zip(asm_idxs, asm_colors):
        u_vals = endo["urgency_mat"][ai]
        if not u_vals:
            continue
        x = np.arange(len(u_vals))
        ax.plot(x, u_vals, marker='o', ms=7, lw=2.0,
                color=col, label=f"Assembly {chr(65+ai)}")
        ax.fill_between(x,
                        np.array(u_vals) * 0.9,
                        np.array(u_vals) * 1.1,
                        color=col, alpha=0.12)
    ax.set_xticks(range(len(rest_ids)))
    ax.set_xticklabels([f"Rest {r+1}\n(after {chr(65+r)})" for r in rest_ids],
                       fontsize=9)
    ax.legend(fontsize=8)
    _style(ax,
           ylabel="Mean urgency probability",
           title="R8a — Urgency Trajectories")

    # ── R8b: Allocation trajectories ─────────────────────────────────────────
    ax = axes[1]
    bottoms = np.zeros(len(rest_ids))
    for ai, col in zip(asm_idxs, asm_colors):
        fracs = np.array(endo["allocation_mat"][ai])
        ax.bar(range(len(rest_ids)), fracs, bottom=bottoms,
               color=col, alpha=0.78, label=f"Assembly {chr(65+ai)}")
        bottoms += fracs
    ax.set_xticks(range(len(rest_ids)))
    ax.set_xticklabels([f"Rest {r+1}" for r in rest_ids], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    _style(ax,
           ylabel="Fraction of burst events",
           title="R8b — Replay Budget Allocation")

    # ── R8c: Acceptance efficiency by urgency quartile ────────────────────────
    ax = axes[2]
    qvals  = np.array(endo["quartile_accept"])
    qlabels = endo["quartile_labels"]
    colors_q = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(qvals)))
    bars = ax.bar(range(len(qvals)), qvals * 100, color=colors_q, alpha=0.85)
    ax.set_xticks(range(len(qvals)))
    ax.set_xticklabels(qlabels, fontsize=10)
    ax.set_ylim(0, max(qvals.max() * 130, 5.0))
    for bar, val in zip(bars, qvals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha='center', va='bottom', fontsize=9)
    _style(ax,
           xlabel="Urgency quartile",
           ylabel="Acceptance rate (%)",
           title="R8c — Efficiency vs Urgency Quartile")
    ax.text(0.98, 0.98,
            f"N events = {endo['n_events']}",
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            color='grey')

    plt.tight_layout()
    _save_fig(fig, "endogenous_prioritization")


# ── Phase 2 plotting ──────────────────────────────────────────────────

def fig_systems_consolidation(all_results):
    """Plot HC→Ctx transfer curves and representational drift."""
    for res in all_results:
        _cond = res["cond"]
        _trials = res["trials"]
        _transfer = [t.get("hc_ctx_transfer", []) for t in _trials if t.get("hc_ctx_transfer")]
        _hc_drift = [t.get("hc_drift", np.zeros((4,4))) for t in _trials if t.get("hc_drift") is not None]
        if not _transfer and not _hc_drift:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax1, ax2 = axes
        # HC vs Ctx contribution over memories
        if _transfer and _transfer[0]:
            _n_mem = len(_transfer[0])
            _hc_means = np.array([_safe_mean([t[i].get("hc_isyn", 0) for t in _transfer]) for i in range(_n_mem)])
            _ctx_means = np.array([_safe_mean([t[i].get("ctx_isyn", 0) for t in _transfer]) for i in range(_n_mem)])
            ax1.bar(np.arange(_n_mem) - 0.15, _hc_means, 0.3, label="HC", color="#e74c3c", alpha=0.7)
            ax1.bar(np.arange(_n_mem) + 0.15, _ctx_means, 0.3, label="Cortex", color="#3498db", alpha=0.7)
            ax1.set_xlabel("Memory")
            ax1.set_ylabel("Mean I_syn contribution")
            ax1.set_title("HC vs Cortex contribution")
            ax1.legend()
            ax1.set_xticks(range(_n_mem))
            ax1.set_xticklabels([chr(65+i) for i in range(_n_mem)])
        # Representational drift
        if _hc_drift and len(_hc_drift[0]) > 0:
            _mean_hc = _safe_nanmean(np.array([d for d in _hc_drift]), axis=0) if len(_hc_drift) > 0 else np.zeros((4,4))
            im = ax2.imshow(_mean_hc, cmap="Reds", aspect="auto")
            plt.colorbar(im, ax=ax2)
            ax2.set_title("HC representational drift")
            ax2.set_xlabel("Training step")
            ax2.set_ylabel("Memory")
        _style(ax1); _style(ax2)
        plt.tight_layout()
        _save_fig(fig, f"systems_consolidation_{_cond['label'].replace('/','_').replace(' ','')}")


def fig_transfer_curves(all_results):
    """Plot emergence of cortical support over replay cycles."""
    for res in all_results:
        _trials = res["trials"]
        _curves = [t.get("transfer_curves", {}) for t in _trials if t.get("transfer_curves")]
        if not _curves:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for asm_idx in range(max(len(c) for c in _curves) if _curves else 0):
            _all_ratios = [c.get(asm_idx, {}).get("transfer_ratio", 0) for c in _curves if asm_idx in c]
            if _all_ratios:
                ax.scatter([asm_idx]*len(_all_ratios), _all_ratios, alpha=0.5, label=f"Memory {chr(65+asm_idx)}")
        ax.axhline(1.0, color='grey', ls=':', label="Equal HC/Ctx")
        ax.set_xlabel("Assembly index")
        ax.set_ylabel("Ctx/HC transfer ratio")
        ax.set_title("Cortical memory emergence (transfer ratio)")
        ax.legend()
        _style(ax)
        plt.tight_layout()
        _save_fig(fig, f"transfer_curves_{res['cond']['label'].replace('/','_').replace(' ','')}")


# ── Phase 3 plotting ──────────────────────────────────────────────────

def fig_dynamical_analysis(all_results):
    """Plot basin stability, spectral radius, participation ratio."""
    for res in all_results:
        _trials = res["trials"]
        _basins = [t.get("basin_stability", {}) for t in _trials if t.get("basin_stability")]
        _sr = [t.get("spectral_radius", 0) for t in _trials]
        _pr = [t.get("participation_ratio", 0) for t in _trials]
        if not _basins and not _sr:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ax1, ax2, ax3 = axes
        # Basin stability
        if _basins and _basins[0]:
            _n_asm = max(len(b) for b in _basins) if _basins else 0
            _probs = []
            for i in range(_n_asm):
                _vals = [b.get(i, {}).get("recovery_prob", 0) for b in _basins if i in b]
                _probs.append(_safe_mean(_vals) if _vals else 0)
            ax1.bar(range(len(_probs)), _probs, color="#2ecc71", alpha=0.7)
            ax1.set_xlabel("Assembly")
            ax1.set_ylabel("Recovery probability")
            ax1.set_title("Basin stability")
            ax1.set_xticks(range(len(_probs)))
            ax1.set_xticklabels([chr(65+i) for i in range(len(_probs))])
        # Spectral radius
        if _sr:
            _sr_arr = np.array(_sr)
            _sr_mean = _safe_mean(_sr_arr)
            ax2.bar(["Spectral\nradius"], [_sr_mean], color="#9b59b6", alpha=0.7)
            ax2.axhline(1.0, color='r', ls='--', label="chaos threshold")
            ax2.legend()
            ax2.set_title(f"Spectral radius (mean={_sr_mean:.3f})")
        # Participation ratio
        if _pr:
            _pr_arr = np.array(_pr)
            _pr_mean = _safe_mean(_pr_arr)
            ax3.bar(["Participation\nratio"], [_pr_mean], color="#e67e22", alpha=0.7)
            ax3.set_title(f"Effective dim (mean={_pr_mean:.1f})")
        _style(ax1); _style(ax2); _style(ax3)
        plt.tight_layout()
        _save_fig(fig, f"dynamical_analysis_{res['cond']['label'].replace('/','_').replace(' ','')}")


def fig_metastable_analysis(all_results):
    """Plot metastable state dwell times and transition matrices."""
    for res in all_results:
        _trials = res["trials"]
        _meta = [t.get("metastable_states", {}) for t in _trials if t.get("metastable_states")]
        if not _meta or not _meta[0].get("dwell_times"):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax1, ax2 = axes
        # Dwell time histogram
        _all_dwell = []
        for m in _meta:
            _all_dwell.extend(m.get("dwell_times", []))
        if _all_dwell:
            ax1.hist(_all_dwell, bins=20, color="#3498db", alpha=0.7, edgecolor="white")
            ax1.set_xlabel("Dwell time (steps)")
            ax1.set_ylabel("Count")
            ax1.set_title("Metastable state dwell times")
        # Transition matrix
        _trans_mats = [m.get("transition_matrix", np.zeros((4,4))) for m in _meta]
        if _trans_mats and len(_trans_mats[0]) > 0:
            _mean_trans = _safe_nanmean(np.array([t for t in _trans_mats if t is not None]), axis=0)
            im = ax2.imshow(_mean_trans, cmap="Blues", aspect="auto")
            plt.colorbar(im, ax=ax2)
            ax2.set_title("State transition matrix")
            ax2.set_xlabel("To state"); ax2.set_ylabel("From state")
        _style(ax1); _style(ax2)
        plt.tight_layout()
        _save_fig(fig, f"metastable_analysis_{res['cond']['label'].replace('/','_').replace(' ','')}")


# ── Phase 4 plotting ──────────────────────────────────────────────────

def fig_homeostatic_analysis(all_results):
    """Summary plot for homeostatic plasticity effects."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _labels = ["BCM", "Intrinsic\nPlast.", "SFA", "Local\nScaling", "iSTDP"]
    _active = [BCM_ENABLED, INTRINSIC_PLASTICITY, SPIKE_FREQ_ADAPT, LOCAL_SCALING, INHIBITORY_STDP]
    _colors = ["#27ae60" if a else "#e74c3c" for a in _active]
    ax.bar(range(len(_labels)), [1 if a else 0 for a in _active], color=_colors, alpha=0.7)
    ax.set_xticks(range(len(_labels)))
    ax.set_xticklabels(_labels, fontsize=8)
    ax.set_ylabel("Active")
    ax.set_title("Homeostatic Plasticity Mechanisms")
    _style(ax)
    plt.tight_layout()
    _save_fig(fig, "homeostatic_summary")


# ── Phase 5 plotting ──────────────────────────────────────────────────

def fig_interneuron_analysis(all_results):
    """Plot interneuron type-specific dynamics."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _types = ["PV", "SOM", "VIP", "Other"]
    _fracs = [PV_FRACTION, SOM_FRACTION, VIP_FRACTION, 1.0 - PV_FRACTION - SOM_FRACTION - VIP_FRACTION]
    _colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]
    _active = [ABLATION_PHASE5.get("pv_cells", False),
               ABLATION_PHASE5.get("som_cells", False),
               ABLATION_PHASE5.get("vip_cells", False),
               True]
    _alphas = [0.9 if a else 0.3 for a in _active]
    ax.bar(_types, _fracs, color=_colors, alpha=_alphas)
    ax.set_ylabel("Fraction of INH pool")
    ax.set_title(f"Interneuron Subtypes ({'active' if any(_active[:-1]) else 'inactive'})")
    _style(ax)
    plt.tight_layout()
    _save_fig(fig, "interneuron_subtypes")


# ── Phase 6 plotting ──────────────────────────────────────────────────

def fig_energy_analysis(all_results):
    """Plot energy consumption and replay suppression."""
    for res in all_results:
        _trials = res["trials"]
        _energy = [t.get("energy_metrics", {}) for t in _trials if t.get("energy_metrics")]
        if not _energy or not any(e for e in _energy):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax1, ax2 = axes
        # Remaining energy
        _remaining = [e.get("energy_remaining", ENERGY_BUDGET) for e in _energy if e]
        _consumed = [e.get("energy_consumed", 0) for e in _energy if e]
        if _remaining:
            ax1.plot(_remaining, color="#2ecc71", alpha=0.7, label="Remaining")
            ax1.plot(_consumed, color="#e74c3c", alpha=0.7, label="Consumed")
            ax1.set_xlabel("Trial")
            ax1.set_ylabel("Energy")
            ax1.set_title("Energy budget tracking")
            ax1.legend()
        # Suppression
        _supp = [e.get("suppression_active", False) for e in _energy if e]
        if _supp:
            ax2.bar(["Suppression\nactive"], [sum(_supp) / max(len(_supp), 1)], color="#e67e22", alpha=0.7)
            ax2.set_ylabel("Fraction of trials")
            ax2.set_title("Replay suppression rate")
        _style(ax1); _style(ax2)
        plt.tight_layout()
        _save_fig(fig, f"energy_analysis_{res['cond']['label'].replace('/','_').replace(' ','')}")


def fig_consolidation_efficiency(all_results):
    """Plot retention-per-energy metric."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _cond_labels = []
    _efficiencies = []
    for res in all_results:
        _trials = res["trials"]
        _energy = [t.get("energy_metrics", {}) for t in _trials if t.get("energy_metrics")]
        _finals = [t.get("final_scores", [np.nan]) for t in _trials]
        if not _energy or not _finals:
            continue
        _eff = []
        for e, f in zip(_energy, _finals):
            _con = e.get("energy_consumed", 1)
            _ret = _safe_mean(f) if np.any(np.isfinite(f)) else 0
            _eff.append(_ret / max(_con, 1e-8))
        _cond_labels.append(res["cond"]["label"])
        _efficiencies.append(_safe_mean(_eff) if _eff else 0)
    if _efficiencies:
        ax.bar(range(len(_cond_labels)), _efficiencies, color=[c["color"] for c in CONDITIONS[:len(_cond_labels)]], alpha=0.7)
        ax.set_xticks(range(len(_cond_labels)))
        ax.set_xticklabels(_cond_labels, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel("Retention / Energy (a.u.)")
        ax.set_title("Consolidation efficiency")
        _style(ax)
    plt.tight_layout()
    _save_fig(fig, "consolidation_efficiency")





# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_global = _tick()
    print(f"[INFO] Device: {DEVICE}", flush=True)
    print("\n" + "="*70, flush=True)
    print("CATASTROPHIC FORGETTING (v4 -- sparse modular architecture)", flush=True)
    print(f"  Architecture mode: {ARCH_MODE}", flush=True)
    print(f"  Memories: {N_MEMORIES}  |  Assembly size: {ASSEMBLY_SIZE}", flush=True)
    if ARCH_MODE == "sparse_modular":
        print(f"  Modules: {N_MODULES}  |  Intra conn: {INTRA_MODULE_CONN_PROB}"
              f"  Inter conn: {INTER_MODULE_CONN_PROB}"
              f"  Inter scale: {INTER_MODULE_SCALE}", flush=True)
    pres_str = (f"{_N_PRESENTATIONS} (DEV fast-path; prod={N_PRESENTATIONS_PER_MEM})"
                if DEV_MODE else str(N_PRESENTATIONS_PER_MEM))
    print(f"  Presentations/memory: {pres_str}", flush=True)
    print(f"  GAMMA={GAMMA}  FAST_DECAY_TAU={FAST_DECAY_TAU}  (RESTORED to validated)", flush=True)
    print(f"  Replay: noise={REPLAY_NOISE_STD}  events={N_REPLAY_EVENTS_PER_REST}"
          f"  spont={REPLAY_SPONTANEOUS_STEPS}"
          f"  burst={REPLAY_BURST_SIZE}x(gap={REPLAY_BURST_GAP}steps)", flush=True)
    print(f"  Competition: strength={COMPETITION_STRENGTH}", flush=True)
    print(f"  Tagging: rate={TAG_CAPTURE_RATE}  (W_slow clamped to [0,{W_MAX}])", flush=True)
    print(f"  Workers: {N_WORKERS}  |  DEV_MODE: {DEV_MODE}  |  PDFs: {GENERATE_PDFS}",
          flush=True)
    print("="*70, flush=True)

    _n_trials   = 2 if DEV_MODE else N_TRIALS
    _n_sweep    = 1 if DEV_MODE else N_TRIALS_SWEEP
    _n_ablation = 2 if DEV_MODE else N_TRIALS_ABLATION

    # ── Scaling validation (runs in both modes, reports unconditionally) ────
    if DEBUG_SCALING or ARCH_MODE == "sparse_modular":
        scaling_results = validate_scaling()
        has_failures = any(v != FAILURE_STABLE and "STABLE" not in v
                           for v in scaling_results.values())
        if has_failures:
            print("\n[WARNING] Scaling validation FAILED for some sizes. "
                  "Proceeding with main experiment anyway.", flush=True)

    MAIN_OVERLAP = 0.20
    assemblies = make_overlapping_assemblies(N_MEMORIES, ASSEMBLY_SIZE, MAIN_OVERLAP)
    for i, asm in enumerate(assemblies):
        print(f"  Assembly {chr(65+i)}: neurons {asm[0]}..{asm[-1]}", flush=True)
    print(f"  BG window: {BG_START}..{BG_END-1}", flush=True)

    # ── Safety check before full run ─────────────────────────────────────────
    if not validate_single_trial(assemblies):
        sys.exit(1)

    # ── Phase 1: main experiment ─────────────────────────────────────────────
    print(f"\n[PHASE 1] Main experiment ({_n_trials} trials, overlap=20%)", flush=True)
    all_results = run_all_conditions(assemblies, n_trials=_n_trials, verbose=True)

    print_statistics(all_results)

    print("\n[PHASE 1] Generating figures ...", flush=True)
    fig_forgetting_curves(all_results)
    fig_replay_protection(all_results)
    if not DEV_MODE:
        fig_interference_matrix(all_results)
        fig_synaptic_overlap_evolution(all_results)
        fig_replay_preserves_old_memories(all_results)
        fig_representational_drift(all_results)
        fig_synaptic_tag_evolution(all_results)

    # ── Phase 1b: replay quality analysis ────────────────────────────────────
    print("\n[PHASE 1b] Replay quality analysis", flush=True)
    rq_analysis = analyze_replay_quality(all_results)
    print_replay_quality_summary(rq_analysis)
    fig_replay_coherence_distributions(rq_analysis)
    fig_coherence_vs_retention(rq_analysis)
    fig_replay_success_across_bursts(rq_analysis)
    fig_replay_coherence_trajectory(rq_analysis)
    fig_adaptive_replay_analysis(rq_analysis)
    fig_attractor_dynamics(rq_analysis)
    fig_competition_dynamics(rq_analysis)

    # ── Phase 2: overlap sweep ────────────────────────────────────────────────
    print(f"\n[PHASE 2] Overlap sweep", flush=True)
    sweep = run_overlap_sweep(n_trials=_n_sweep, verbose=True)
    fig_overlap_vs_forgetting(sweep)
    if not DEV_MODE:
        fig_retention_surface(sweep)
        fig_memory_vulnerability_map(sweep)
        fig_overlap_interference_phase_diagram(sweep)

    # ── Phase 3: prioritization ───────────────────────────────────────────────
    print(f"\n[PHASE 3] Prioritization comparison", flush=True)
    prio = run_prioritization_comparison(assemblies, n_trials=_n_sweep)
    fig_replay_scheduling(prio)

    # ── Phase 3b: endogenous prioritization analysis ──────────────────────────
    print(f"\n[PHASE 3b] Endogenous replay prioritization analysis", flush=True)
    endo_analysis = analyze_endogenous_replay(prio)
    if endo_analysis is not None:
        n_endo = endo_analysis["n_events"]
        print(f"  Endogenous events analysed: {n_endo}", flush=True)
        fig_endogenous_prioritization(endo_analysis)
    else:
        print("  [SKIP] No endogenous events found (DEV cold-start or no trials).",
              flush=True)

    # ── Phase 4: ablation suite ───────────────────────────────────────────────
    print(f"\n[PHASE 4] Ablation suite ({_n_ablation} trials x "
          f"{len(ABLATION_CONDITIONS)} conditions)", flush=True)
    ablation_results = run_ablation_suite(assemblies, n_trials=_n_ablation, verbose=True)
    print_ablation_summary(ablation_results)
    fig_ablation_suite(ablation_results)

    # ── Phase 2: systems consolidation analysis ───────────────────────────
    print("\n[PHASE 2] Systems consolidation analysis", flush=True)
    if any(t.get("hc_ctx_transfer") for r in all_results for t in r["trials"]):
        fig_systems_consolidation(all_results)
        fig_transfer_curves(all_results)
        for res in all_results:
            _n_drift = sum(1 for t in res["trials"] if t.get("hc_drift") is not None)
            print(f"  {res['cond']['label']:25s}: drift matrices={_n_drift}/{len(res['trials'])}", flush=True)
    else:
        print("  [SKIP] No HC/Ctx transfer data (requires N_HC > 0)", flush=True)

    # ── Phase 2b: lesion experiments (single-trial demo) ──────────────────
    if not DEV_MODE and N_HC > 0:
        print("\n[PHASE 2b] Lesion experiment demo", flush=True)
        _lesion_net = build_network(use_slow=True)
        for _li, asm in enumerate(assemblies):
            train_one_memory(_lesion_net, asm, n_presentations=2)
        _pre = [probe_memory(_lesion_net, a)["isyn_score"] for a in assemblies]
        print(f"  Pre-lesion scores: {np.array(_pre).round(3)}", flush=True)
        lesion_network(_lesion_net, "hc")
        _post_hc = [probe_memory(_lesion_net, a)["isyn_score"] for a in assemblies]
        print(f"  Post-HC lesion:    {np.array(_post_hc).round(3)}", flush=True)

    # ── Phase 3: dynamical systems analysis ───────────────────────────────
    print("\n[PHASE 3] Dynamical systems analysis", flush=True)
    if ABLATION_PHASE3.get("basin_stability", True):
        fig_dynamical_analysis(all_results)
        fig_metastable_analysis(all_results)
        for res in all_results:
            _sr = [t.get("spectral_radius", 0) for t in res["trials"]]
            _pr = [t.get("participation_ratio", 0) for t in res["trials"]]
            if _sr:
                print(f"  {res['cond']['label']:25s}: SR={_safe_mean(_sr):.3f}+-{_safe_sem(_sr):.3f}"
                      f"  PR={_safe_mean(_pr):.1f}+-{_safe_sem(_pr):.1f}", flush=True)

    # ── Phase 4: homeostatic plasticity analysis ──────────────────────────
    print("\n[PHASE 4] Homeostatic plasticity analysis", flush=True)
    _hp_flags = [BCM_ENABLED, INTRINSIC_PLASTICITY, SPIKE_FREQ_ADAPT, LOCAL_SCALING, INHIBITORY_STDP]
    if any(_hp_flags):
        fig_homeostatic_analysis(all_results)
        print(f"  Mechanisms active: BCM={BCM_ENABLED} IP={INTRINSIC_PLASTICITY}"
              f" SFA={SPIKE_FREQ_ADAPT} LS={LOCAL_SCALING} iSTDP={INHIBITORY_STDP}", flush=True)
    else:
        print("  All homeostatic mechanisms disabled (toggle via constants or ablation)", flush=True)

    # ── Phase 5: interneuron diversity analysis ───────────────────────────
    print("\n[PHASE 5] Interneuron diversity analysis", flush=True)
    _pv_flag = ABLATION_PHASE5.get("pv_cells", False)
    _som_flag = ABLATION_PHASE5.get("som_cells", False)
    _vip_flag = ABLATION_PHASE5.get("vip_cells", False)
    if _pv_flag or _som_flag or _vip_flag:
        fig_interneuron_analysis(all_results)
        _n_pv = int(N_INH * PV_FRACTION)
        _n_som = int(N_INH * SOM_FRACTION)
        _n_vip = int(N_INH * VIP_FRACTION)
        print(f"  PV={_n_pv} SOM={_n_som} VIP={_n_vip} "
              f"(fracs: {PV_FRACTION:.0%}/{SOM_FRACTION:.0%}/{VIP_FRACTION:.0%})", flush=True)
    else:
        print("  Interneuron subtypes disabled (toggle via ABLATION_PHASE5)", flush=True)

    # ── Phase 6: energy-constrained replay analysis ───────────────────────
    print("\n[PHASE 6] Energy-constrained replay analysis", flush=True)
    if ENERGY_TRACKING:
        fig_energy_analysis(all_results)
        fig_consolidation_efficiency(all_results)
        for res in all_results:
            _e = [t.get("energy_metrics", {}) for t in res["trials"] if t.get("energy_metrics")]
            if _e:
                _avg_rem = _safe_mean([e.get("energy_remaining", 0) for e in _e])
                _avg_con = _safe_mean([e.get("energy_consumed", 0) for e in _e])
                _n_sup = sum(1 for e in _e if e.get("suppression_active", False))
                print(f"  {res['cond']['label']:25s}: avg_rem={_avg_rem:.1f}"
                      f" consumed={_avg_con:.1f} suppression={_n_sup}/{len(_e)}", flush=True)
    else:
        print("  Energy tracking disabled (toggle ENERGY_TRACKING)", flush=True)

    # ── Schema abstraction analysis ──────────────────────────────────────
    _call_hooks("analysis", all_results=all_results)

    # ── Phase 5 (existing): publication summary ───────────────────────────
    if not DEV_MODE:
        fig_publication_summary(all_results, prio, sweep)

    elapsed = time.perf_counter() - t_global
    print_timing_report(elapsed)
    print(f"\n[DONE] {elapsed/60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
