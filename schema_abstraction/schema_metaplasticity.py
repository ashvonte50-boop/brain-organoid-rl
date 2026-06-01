"""Metaplasticity and hidden-state tracking for replay-dependent consolidation.

Metaplasticity — the modulation of plasticity based on recent activity
history — implements the "sliding threshold" (BCM theory) where frequently
active synapses require stronger co-activation to potentiate further.

Hidden states track the consolidation status of each memory, providing
a continuous measure of how "cortical" vs "hippocampal" each memory is.
"""

import numpy as np
import torch

from compare_catastrophic_forgetting import N_EXC, N_NEURONS, DEVICE


# ── Config ──────────────────────────────────────────────────────────────
META_THETA_P  = 0.3    # sliding threshold for potentiation
META_THETA_D  = 0.1    # sliding threshold for depression
META_TAU      = 1000.0 # time constant for threshold dynamics (ms)
META_ETA      = 0.01   # learning rate for threshold adjustment
META_ACT_WINDOW = 50.0 # ms window for activity tracking

HIDDEN_TAU    = 500.0  # time constant for hidden-state accumulation


class MetaplasticityController:
    """Implements BCM-style sliding threshold for STDP.

    Each excitatory synapse tracks its recent activity and adjusts its
    plasticity threshold accordingly.  Frequently active synapses become
    harder to potentiate (protection from runaway excitation) and easier
    to depress (forgetting of over-learned patterns).

    Attached to the net as ``net._meta``.
    """

    def __init__(self, net):
        self.n_exc = net.n_exc
        self.device = net.W.device

        # Sliding thresholds per excitatory neuron
        self.theta_p = torch.full((self.n_exc,), META_THETA_P, device=self.device)
        self.theta_d = torch.full((self.n_exc,), META_THETA_D, device=self.device)

        # Recent activity trace (low-pass filtered spike count per neuron)
        self.activity_trace = torch.zeros(self.n_exc, device=self.device)

        # Metaplastic modulation factor (applied to STDP learning rate)
        # 1.0 = default, < 1.0 = suppressed, > 1.0 = enhanced
        self.modulation = torch.ones(self.n_exc, self.n_exc, device=self.device)

        self.step_count = 0
        self.meta_history = []

    def update(self, net):
        """Update sliding thresholds based on recent activity.

        Call after each forward step during the training/replay phase.
        """
        with torch.no_grad():
            spikes_exc = net.spikes[:self.n_exc].float()
            self.activity_trace += (spikes_exc - self.activity_trace) / META_TAU * net.dt

            # Adjust thresholds: high activity → raise theta_p (harder to potentiate)
            act = self.activity_trace.clamp(min=0.0, max=1.0)
            self.theta_p += META_ETA * (act - self.theta_p)
            self.theta_d += META_ETA * (act - self.theta_d)

            # Compute modulation: synapses onto highly-active postsynaptic neurons
            # get suppressed potentiation
            self.modulation = torch.ones(self.n_exc, self.n_exc, device=self.device)
            for post_idx in range(self.n_exc):
                if self.activity_trace[post_idx] > self.theta_p[post_idx]:
                    self.modulation[post_idx, :] *= 0.7
                if self.activity_trace[post_idx] < self.theta_d[post_idx]:
                    self.modulation[post_idx, :] *= 1.3

            self.step_count += 1

    def get_effective_lr(self, base_lr):
        """Return metaplasticity-modulated learning rate matrix."""
        return base_lr * self.modulation

    def summary(self):
        return {
            "n_steps": self.step_count,
            "mean_theta_p": float(self.theta_p.mean().item()),
            "mean_theta_d": float(self.theta_d.mean().item()),
            "mean_activity": float(self.activity_trace.mean().item()),
            "frac_modulated": float((self.modulation != 1.0).float().mean().item()),
        }


class HiddenStateTracker:
    """Tracks the consolidation status of each memory as a hidden state.

    Each memory (assembly) has a hidden state value between 0 and 1:
      - 0.0 = fully hippocampal (requires hippocampus for retrieval)
      - 1.0 = fully cortical (can be retrieved without hippocampus)

    The hidden state evolves according to:
      dH/dt = (replay_benefit - H) / HIDDEN_TAU + noise
    where replay_benefit = 1 if the memory was replayed recently, else 0.

    Attached to the net as ``net._hidden_state``.
    """

    def __init__(self, n_memories):
        self.n = n_memories
        self.hidden_states = np.zeros(n_memories, dtype=np.float32)
        self.last_replay_time = np.full(n_memories, -1e6, dtype=np.float32)
        self.time = 0.0
        self.history = []

    def step(self, dt_ms, replayed_memories=None):
        """Advance hidden states by dt_ms.

        Args:
            dt_ms: time step in milliseconds.
            replayed_memories: list of assembly indices replayed in this step.
        """
        if replayed_memories is None:
            replayed_memories = []
        for ai in range(self.n):
            replay_benefit = 1.0 if ai in replayed_memories else 0.0
            dh = (replay_benefit - self.hidden_states[ai]) / HIDDEN_TAU * dt_ms
            self.hidden_states[ai] = np.clip(self.hidden_states[ai] + dh, 0.0, 1.0)
        self.time += dt_ms
        self.history.append(self.hidden_states.copy())

    def get_consolidation_status(self):
        """Return dict mapping assembly index to consolidation level."""
        return {i: float(self.hidden_states[i]) for i in range(self.n)}

    def summary(self):
        return {
            "final_states": self.hidden_states.tolist(),
            "mean_consolidation": float(self.hidden_states.mean()),
            "n_cortical": int((self.hidden_states > 0.7).sum()),
            "n_hippocampal": int((self.hidden_states < 0.3).sum()),
        }


# ── Hook callbacks ─────────────────────────────────────────────────────

def _meta_baseline_hook(net, assemblies, n_mem, j=-1, **_):
    """Initialise metaplasticity controller and hidden state tracker."""
    if not hasattr(net, "_meta"):
        net._meta = MetaplasticityController(net)
    if not hasattr(net, "_hidden_state"):
        net._hidden_state = HiddenStateTracker(n_mem)


def _meta_post_encode_hook(net, assemblies, n_mem, j, **_):
    """Update metaplasticity after each encoding phase."""
    meta = getattr(net, "_meta", None)
    if meta is not None:
        meta.update(net)
    hs = getattr(net, "_hidden_state", None)
    if hs is not None:
        hs.step(net.dt)


def _meta_replay_hook(net, assemblies, n_mem, j, **_):
    """Update metaplasticity and hidden states after replay."""
    meta = getattr(net, "_meta", None)
    if meta is not None:
        meta.update(net)
    hs = getattr(net, "_hidden_state", None)
    if hs is not None:
        replayed = list(range(min(j + 1, hs.n)))
        hs.step(net.dt, replayed_memories=replayed)


def _meta_final_hook(net, assemblies, n_mem, **_):
    """Store metaplasticity and hidden state summaries in hook_extra."""
    extra = getattr(net, "_hook_extra", None)
    if extra is None:
        return
    meta = getattr(net, "_meta", None)
    if meta is not None:
        extra["metaplasticity"] = meta.summary()
    hs = getattr(net, "_hidden_state", None)
    if hs is not None:
        extra["hidden_state"] = hs.summary()
