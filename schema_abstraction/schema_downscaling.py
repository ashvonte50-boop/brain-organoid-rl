"""Synaptic Downscaling with Synaptic Tagging & Capture, Consolidation Mask,
and distinct NREM/REM sleep phases.

Implements the full sleep-dependent consolidation pipeline:
  1. Synaptic Tagging (Frey & Morris 1997): synapses that changed during
     encoding are tagged; tags decay exponentially.
  2. Global downscaling (Tononi & Cirelli 2014): all E→E weights are
     weakened multiplicatively during offline periods.
  3. Replay protection (Gulati et al. 2017): synapses that were active
     during replay get a protection boost.
  4. Consolidation mask: synapses repeatedly activated during coherent
     replay accumulate consolidation strength; highly consolidated
     synapses become partially resistant (STC-consistent).
  5. NREM/REM distinction: NREM prunes, REM strengthens schema-related
     connections via targeted replay.

Reference:
  Frey & Morris (1997) Synaptic tagging and long-term potentiation.
  Tononi & Cirelli (2014) Sleep and synaptic homeostasis.
  Gulati et al. (2017) Replay protects consolidated synapses.
"""

import numpy as np
import torch

from compare_catastrophic_forgetting import DEVICE
import compare_catastrophic_forgetting as _ccf

# DEV_MODE speed-up
try:
    from compare_catastrophic_forgetting import DEV_MODE as _DEV
except ImportError:
    _DEV = False

# ── Config ──────────────────────────────────────────────────────────────
DOWNSCALE_ENABLED       = True
DOWNSCALE_RATE          = 0.0008        # per-step multiplicative weakening (balanced for retention + protection gap)
DOWNSCALE_NREM_DUR      = 200.0         # ms — NREM phase duration (reduced from 400)
DOWNSCALE_REM_DUR       = 100.0         # ms — REM phase duration (reduced from 200)
DOWNSCALE_PROTECT       = 2.0           # protection multiplier for replayed synapses
DOWNSCALE_FLOOR         = 1e-6          # prevent true zero
STDP_WINDOW             = 50.0          # ms — causal window for marking replay activity

# Synaptic Tagging & Capture (STC)
TAG_THRESHOLD           = 0.005         # weight change threshold for tagging (reduced from 0.01)
TAG_TAU                 = 300.0         # tag decay time constant (steps)
TAG_SLOW_DECAY          = np.exp(-1.0 / TAG_TAU)  # per-step tag decay
PROTECTED_DECAY         = 0.9998        # decay multiplier for tagged/protected synapses (was 0.9995)
TAG_ACTIVATION          = 0.5           # tag value above which synapse is "captured"

# Consolidation mask
CONSOLIDATION_TAU       = 500.0         # accumulation time constant
CONSOLIDATION_THR       = 3.0           # protection events to reach full strength
CONSOLIDATION_MAX       = 0.8           # max protection fraction (1.0 = fully immune)
CONSOLIDATION_DECAY     = np.exp(-1.0 / CONSOLIDATION_TAU)  # per-step decay

# NREM/REM config
NREM_NOISE_STD          = 0.5           # elevated Gaussian noise during NREM
REM_NOISE_STD           = 0.2           # baseline noise during REM
REM_REPLAY_PROB_CORE    = 0.85          # replay probability for schema-core assemblies (increased from 0.7)
GATING_OFF_SCALE        = 0.1           # hippocampal input scale during NREM

# Replay diversity analysis
REPLAY_ENTROPY_WINDOW   = 10            # events for sliding entropy computation


class SynapticTagTracker:
    """Tracks synaptic tags (Frey & Morris STC model).

    Tags are set on synapses whose weight changed significantly during
    encoding, then decay exponentially.  Tagged synapses receive
    preferential protection during downscaling.
    """

    def __init__(self, n_exc, device):
        self.n_exc = n_exc
        self.device = device
        self.synaptic_tag = torch.zeros(n_exc, n_exc, dtype=torch.float32, device=device)
        self.weight_snapshot = None

    def snapshot_weights(self, net):
        """Record current W_ee for tag computation."""
        ei = slice(0, self.n_exc)
        self.weight_snapshot = net.W.data[ei, ei].clone()

    def detect_changes(self, net):
        """Set tags on synapses whose weight changed above TAG_THRESHOLD."""
        if self.weight_snapshot is None:
            self.snapshot_weights(net)
            return
        ei = slice(0, self.n_exc)
        current = net.W.data[ei, ei]
        delta = (current - self.weight_snapshot).abs()
        self.synaptic_tag.masked_fill_(delta > TAG_THRESHOLD, 1.0)
        self.weight_snapshot = current.clone()

    def decay_tags(self):
        """Decay all tags exponentially."""
        self.synaptic_tag.mul_(TAG_SLOW_DECAY)

    def protection_mask(self):
        """Return a float mask: 1.0 where tag > TAG_ACTIVATION, 0 elsewhere."""
        return (self.synaptic_tag > TAG_ACTIVATION).float()

    def summary(self):
        return {
            "mean_tag": float(self.synaptic_tag.mean().item()),
            "n_tagged": int((self.synaptic_tag > TAG_ACTIVATION).sum().item()),
            "tag_frac": float((self.synaptic_tag > TAG_ACTIVATION).float().mean().item()),
        }


class ConsolidationMask:
    """Tracks which synapses have been consolidated through repeated replay.

    Each coherent replay event increments consolidation strength for
    active synapses.  Highly consolidated synapses are partially resistant
    to global downscaling.
    """

    def __init__(self, n_exc, device):
        self.n_exc = n_exc
        self.device = device
        self.strength = torch.zeros(n_exc, n_exc, dtype=torch.float32, device=device)
        self.protection_events = torch.zeros(n_exc, n_exc, dtype=torch.float32, device=device)

    def increment(self, active_mask):
        """Increment consolidation strength for synapses in active_mask."""
        self.strength += active_mask.float()
        self.strength.mul_(CONSOLIDATION_DECAY).clamp_(max=CONSOLIDATION_MAX)
        self.protection_events += active_mask.float()
        self.protection_events.clamp_(max=100.0)

    def protection_multiplier(self):
        """Return protection factor in [1.0, 1.0/CONSOLIDATION_MAX].

        A fully consolidated synapse (strength == CONSOLIDATION_MAX)
        gets the maximum protection boost.
        """
        return 1.0 + self.strength * (1.0 / max(CONSOLIDATION_MAX, 1e-6) - 1.0)

    def summary(self):
        n_cons = int((self.strength > 0.5 * CONSOLIDATION_MAX).sum().item())
        return {
            "mean_strength": float(self.strength.mean().item()),
            "n_consolidated": n_cons,
            "mean_protection_events": float(self.protection_events.mean().item()),
        }


class DownscaleTracker:
    """Tracks replay-active synapses and manages the downscaling schedule.

    Attached to the net as ``net._downscale``.  The tracker maintains:
      - ``replay_active_mask`` : bool array (n_exc, n_exc) — synapses that
        carried replay activity in the current epoch.
      - ``replay_potentiation`` : float array — cumulative protection tag.
      - ``cumulative_downscale`` : total multiplicative decay applied.
      - ``synaptic_tag`` : SynapticTagTracker for STC.
      - ``consolidation_mask`` : ConsolidationMask for long-term protection.
      - ``replay_history`` : list of replay-event metrics for diversity analysis.
    """

    def __init__(self, net):
        self.n_exc = net.n_exc
        self.device = net.W.device
        self.reset()

    def reset(self):
        self.replay_active_mask = torch.zeros(
            self.n_exc, self.n_exc, dtype=torch.bool, device=self.device)
        self.replay_potentiation = torch.zeros(
            self.n_exc, self.n_exc, dtype=torch.float32, device=self.device)
        self.cumulative_downscale = 0.0
        self.downscale_events = 0
        self.synaptic_tag = SynapticTagTracker(self.n_exc, self.device)
        self.consolidation_mask = ConsolidationMask(self.n_exc, self.device)
        self.replay_history = []
        self.nrem_count = 0
        self.rem_count = 0
        self.rem_nrem_ratios = []
        self.schema_core_firing_nrem = []
        self.schema_core_firing_rem = []

    def mark_replay_synapses(self, spike_log, assemblies_in_sequence):
        """Scan recent spike activity and mark causal E→E pairs as replay-active."""
        if len(spike_log) < 2:
            return
        times = np.array([s[0] for s in spike_log], dtype=np.float32)
        neurons = np.array([s[1] for s in spike_log], dtype=np.int64)
        exc_only = neurons < self.n_exc
        times = times[exc_only]
        neurons = neurons[exc_only]
        if len(neurons) < 2:
            return
        for post_i in range(1, len(neurons)):
            post_n = int(neurons[post_i])
            pre_n = int(neurons[post_i - 1])
            dt = times[post_i] - times[post_i - 1]
            if 0 < dt < STDP_WINDOW:
                self.replay_active_mask[post_n, pre_n] = True
                self.replay_potentiation[post_n, pre_n] += 0.5

    def apply_nrem_downscale(self, net):
        """NREM phase: global downscaling with tag-based and consolidation protection.

        - Weights are weakened globally.
        - Synapses with active tags or high consolidation strength get
          reduced weakening (PROTECTED_DECAY instead of (1-DOWNSCALE_RATE)).
        - No replay is triggered.
        - Elevated Gaussian noise.
        """
        n_steps = int(DOWNSCALE_NREM_DUR / net.dt)
        d = 1.0 - DOWNSCALE_RATE
        dN = d ** n_steps
        ei = slice(0, self.n_exc)

        # Build protection mask: synapses that are tagged OR consolidated
        tag_protection = self.synaptic_tag.protection_mask()
        cons_protection = (self.consolidation_mask.strength > 0.5 * CONSOLIDATION_MAX).float()
        # Prevent any synapse from being both tagged and consolidated counting double
        protection = torch.maximum(tag_protection, cons_protection)

        with torch.no_grad():
            sub = net.W.data[ei, ei]

            # Apply protection: protected synapses use PROTECTED_DECAY^n_steps
            if protection.any():
                prot_decay = PROTECTED_DECAY ** n_steps
                # Unprotected: use normal decay
                unprotected = (protection == 0)
                sub.mul_(torch.where(unprotected, dN, prot_decay))
            else:
                sub.mul_(dN)

            # Replay potentiation boost
            if self.replay_active_mask.any():
                protect = self.replay_active_mask.float() * DOWNSCALE_PROTECT
                boost_per_step = protect * self.replay_potentiation
                boost_total = boost_per_step * (1.0 - dN) / max(DOWNSCALE_RATE, 1e-10)
                sub.add_(boost_total.to(sub.device))

            # Consolidation-driven boost: consolidated synapses get extra help
            if self.consolidation_mask.strength.any():
                cons_boost = self.consolidation_mask.strength * 0.001 * n_steps
                sub.add_(cons_boost.to(sub.device))

            sub.clamp_(min=DOWNSCALE_FLOOR)

        self.cumulative_downscale += 1.0 - dN
        self.downscale_events += 1
        self.nrem_count += 1

        # Decay tags after NREM
        self.synaptic_tag.decay_tags()

        # Record schema core firing rate for NREM
        _record_schema_firing(net, "nrem", self)

    def apply_rem_replay(self, net, assemblies, core_mask):
        """REM phase: pause downscaling, trigger targeted replay.

        - Downscaling is paused (rate = 0.0).
        - Replay is biased toward schema-core assemblies.
        - Cortical noise reduced to baseline.
        - Hippocampal-cortical input gated ON.
        """
        n_steps = int(DOWNSCALE_REM_DUR / net.dt)
        self.rem_count += 1

        # No downscaling during REM (rate = 0.0)
        # Instead, run targeted replay for schema-core assemblies
        for step in range(n_steps):
            if step % 50 == 0 and assemblies is not None:
                # Select assembly with probability weighted by schema-core overlap
                n_mem = len(assemblies)
                probs = np.ones(n_mem) / n_mem  # default uniform
                if core_mask is not None:
                    # Bias toward assemblies with core overlap
                    core_sizes = []
                    for asm in assemblies:
                        core_overlap = len(np.intersect1d(asm, core_mask))
                        core_sizes.append(core_overlap)
                    core_sizes = np.array(core_sizes, dtype=float)
                    if core_sizes.max() > 0:
                        probs = core_sizes / core_sizes.sum()
                aidx = np.random.choice(n_mem, p=probs)
                asm = assemblies[aidx]

                # Brief pulse to reactivate assembly
                stim = torch.zeros(_ccf.N_NEURONS, device=self.device)
                cue = np.random.choice(asm[asm < self.n_exc],
                                       size=min(10, len(asm)), replace=False)
                stim[cue] = 1.0
                net.forward(stim)

        # Record schema core firing for REM
        _record_schema_firing(net, "rem", self)

        # Compute REM/NREM ratio
        if self.nrem_count > 0 and self.schema_core_firing_rem:
            nrem_mean = np.mean(self.schema_core_firing_nrem[-self.rem_count:]) if self.schema_core_firing_nrem else 0.0
            rem_mean = np.mean(self.schema_core_firing_rem[-self.rem_count:]) if self.schema_core_firing_rem else 0.0
            ratio = rem_mean / max(nrem_mean, 1e-10)
            self.rem_nrem_ratios.append(ratio)

    def apply_downscale(self, net, duration_ms=None):
        """Legacy single-phase downscale (used when NREM/REM not active)."""
        if duration_ms is None:
            duration_ms = DOWNSCALE_NREM_DUR
        n_steps = int(duration_ms / net.dt)
        d = 1.0 - DOWNSCALE_RATE
        dN = d ** n_steps
        ei = slice(0, self.n_exc)

        tag_protection = self.synaptic_tag.protection_mask()
        cons_protection = (self.consolidation_mask.strength > 0.5 * CONSOLIDATION_MAX).float()
        protection = torch.maximum(tag_protection, cons_protection)

        with torch.no_grad():
            sub = net.W.data[ei, ei]
            if protection.any():
                prot_decay = PROTECTED_DECAY ** n_steps
                unprotected = (protection == 0)
                sub.mul_(torch.where(unprotected, dN, prot_decay))
            else:
                sub.mul_(dN)
            if self.replay_active_mask.any():
                protect = self.replay_active_mask.float() * DOWNSCALE_PROTECT
                boost_per_step = protect * self.replay_potentiation
                boost_total = boost_per_step * (1.0 - dN) / max(DOWNSCALE_RATE, 1e-10)
                sub.add_(boost_total.to(sub.device))
            if self.consolidation_mask.strength.any():
                cons_boost = self.consolidation_mask.strength * 0.001 * n_steps
                sub.add_(cons_boost.to(sub.device))
            sub.clamp_(min=DOWNSCALE_FLOOR)

        self.cumulative_downscale += 1.0 - dN
        self.downscale_events += 1
        self.synaptic_tag.decay_tags()

    def update_consolidation(self, coherence):
        """Update consolidation mask based on replay coherence."""
        active_mask = self.replay_active_mask.float()
        if active_mask.any():
            # Scale increment by coherence quality
            scaled = active_mask * max(coherence, 0.1)
            self.consolidation_mask.increment(scaled)

    def compute_replay_entropy(self):
        """Compute replay diversity entropy from recent event history."""
        if len(self.replay_history) < 2:
            return 0.0
        recent = self.replay_history[-REPLAY_ENTROPY_WINDOW:]
        assembly_counts = {}
        for ev in recent:
            idx = ev.get("assembly_idx", -1)
            assembly_counts[idx] = assembly_counts.get(idx, 0) + 1
        total = sum(assembly_counts.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in assembly_counts.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def clear_marks(self):
        self.replay_active_mask.zero_()
        self.replay_potentiation.zero_()

    def summary(self):
        tag_summary = self.synaptic_tag.summary()
        cons_summary = self.consolidation_mask.summary()
        replay_entropy = self.compute_replay_entropy()
        rem_nrem = float(np.mean(self.rem_nrem_ratios)) if self.rem_nrem_ratios else 0.0
        return {
            "active_synapses": int(self.replay_active_mask.sum().item()),
            "cumulative_downscale": self.cumulative_downscale,
            "downscale_events": self.downscale_events,
            "mean_tag": tag_summary["mean_tag"],
            "n_tagged": tag_summary["n_tagged"],
            "tag_frac": tag_summary["tag_frac"],
            "n_consolidated": cons_summary["n_consolidated"],
            "mean_consolidation": cons_summary["mean_strength"],
            "replay_entropy": replay_entropy,
            "rem_nrem_ratio": rem_nrem,
            "nrem_count": self.nrem_count,
            "rem_count": self.rem_count,
        }


def _record_schema_firing(net, phase, tracker):
    """Record mean firing rate of excitatory neurons for this phase."""
    with torch.no_grad():
        fr = float(net.spikes[:_ccf.N_EXC].float().mean().item())
    if phase == "nrem":
        tracker.schema_core_firing_nrem.append(fr)
    else:
        tracker.schema_core_firing_rem.append(fr)


def init_downscale(net):
    """Attach a DownscaleTracker to the net if not already present."""
    if not hasattr(net, "_downscale"):
        net._downscale = DownscaleTracker(net)


def extract_replay_spikes(net, replay_start_time, replay_duration_ms):
    """Extract spike log from the network's recent forward passes."""
    if not hasattr(net, "_spike_log"):
        net._spike_log = []
    return net._spike_log


# ── Multi-step sleep cycle ──────────────────────────────────────────────

def run_sleep_cycle(net, assemblies, core_mask=None, use_nrem_rem=True):
    """Run one full sleep cycle with NREM and REM phases.

    Args:
        net: The network.
        assemblies: List of assembly index arrays.
        core_mask: Array of schema-core neuron indices (or None).
        use_nrem_rem: If True, use distinct NREM/REM phases.
                      If False, use legacy single downscale phase.
    """
    dt = getattr(net, "_downscale", None)
    if dt is None:
        return

    if not use_nrem_rem:
        # Legacy mode
        dt.apply_downscale(net)
        return

    # Phase 1: NREM — global downscale, elevated noise, no replay
    orig_noise = net.noise_std
    net.noise_std = NREM_NOISE_STD

    # Snapshot tags before NREM
    dt.synaptic_tag.snapshot_weights(net)

    # Apply NREM downscale
    dt.apply_nrem_downscale(net)

    # Phase 2: REM — no downscale, targeted replay
    net.noise_std = REM_NOISE_STD
    dt.apply_rem_replay(net, assemblies, core_mask)

    # Restore noise
    net.noise_std = orig_noise


# ── Hook callbacks ─────────────────────────────────────────────────────

def _downscale_pre_replay_hook(net, assemblies, n_mem, j, **_):
    """Initialize or reset downscale tracking before replay."""
    init_downscale(net)
    dt = net._downscale
    dt.clear_marks()
    # Snapshot weights before encoding for tag detection
    dt.synaptic_tag.snapshot_weights(net)


def _downscale_post_replay_hook(net, assemblies, n_mem, j, **_):
    """Mark replay-active synapses, update consolidation, apply downscaling."""
    dt = getattr(net, "_downscale", None)
    if dt is None:
        return

    # Detect synaptic changes from encoding
    dt.synaptic_tag.detect_changes(net)

    # Mark replay spikes (only when there was actual replay — not last memory)
    if j < n_mem - 1:
        if hasattr(net, "_spike_log") and net._spike_log:
            dt.mark_replay_synapses(net._spike_log, assemblies[:j + 1])
        replay_coh = getattr(net, "_last_replay_coherence", 0.5)
        dt.update_consolidation(replay_coh)

    # Sleep cycle: run only once after the last memory is encoded
    core_mask = getattr(net, "_schema_core_mask", None)
    if j == n_mem - 1:
        run_sleep_cycle(net, assemblies, core_mask=core_mask)


def _downscale_final_hook(net, assemblies, n_mem, **_):
    """Store downscale summary in hook_extra."""
    dt = getattr(net, "_downscale", None)
    if dt is None:
        return
    extra = getattr(net, "_hook_extra", None)
    if extra is not None:
        extra["downscale_summary"] = dt.summary()
