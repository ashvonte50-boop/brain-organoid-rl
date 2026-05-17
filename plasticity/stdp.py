"""
Spike-Timing-Dependent Plasticity (STDP) learning rules.

PairSTDP             — Classical nearest-neighbour pair STDP.
RewardModulatedSTDP  — Three-factor rule: eligibility trace gated by a reward signal.
                       This is the primary bridge between plasticity and RL.

Biological basis:
    Long-term potentiation (LTP) occurs when pre fires before post (causal).
    Long-term depression (LTD) occurs when post fires before pre (anti-causal).
    Reward modulation reflects dopaminergic signalling in striatum / PFC circuits.

Future work:
    - Triplet STDP (Pfister & Gerstner 2006) for better rate-dependence
    - Voltage-based STDP (Clopath et al. 2010) for organoid-scale simulations
    - Weight-dependent updates with soft upper/lower bounds
"""

import torch
import torch.nn as nn


class PairSTDP(nn.Module):
    """Classical pair-based STDP using exponential pre/post traces.

    Maintains low-pass filtered spike traces for pre- and post-synaptic neurons.
    Weight updates are applied after each timestep based on coincident activity.

    Args:
        n_pre:      Number of pre-synaptic neurons.
        n_post:     Number of post-synaptic neurons.
        A_plus:     LTP learning rate.
        A_minus:    LTD learning rate.
        tau_plus:   Pre-synaptic trace decay (ms).
        tau_minus:  Post-synaptic trace decay (ms).
        w_min:      Hard minimum weight clip.
        w_max:      Hard maximum weight clip.
        dt:         Simulation timestep (ms).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max

        self.decay_pre = torch.exp(torch.tensor(-dt / tau_plus))
        self.decay_post = torch.exp(torch.tensor(-dt / tau_minus))

        self.weight = nn.Parameter(
            torch.rand(n_post, n_pre) * (w_max - w_min) + w_min
        )

    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        trace_pre: torch.Tensor,
        trace_post: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one STDP update step.

        Args:
            pre_spikes:  (batch, n_pre)  binary spike tensor.
            post_spikes: (batch, n_post) binary spike tensor.
            trace_pre:   (batch, n_pre)  pre-synaptic eligibility trace.
            trace_post:  (batch, n_post) post-synaptic eligibility trace.

        Returns:
            dW:          Weight update matrix (n_post, n_pre).
            trace_pre:   Updated pre trace.
            trace_post:  Updated post trace.
        """
        trace_pre = trace_pre * self.decay_pre + pre_spikes
        trace_post = trace_post * self.decay_post + post_spikes

        # LTP: post fires, potentiate based on recent pre activity
        ltp = self.A_plus * torch.einsum("bp,bq->pq", post_spikes, trace_pre)
        # LTD: pre fires, depress based on recent post activity
        ltd = self.A_minus * torch.einsum("bp,bq->pq", trace_post, pre_spikes)

        dW = ltp - ltd
        with torch.no_grad():
            self.weight.data = torch.clamp(self.weight.data + dW, self.w_min, self.w_max)

        return dW, trace_pre, trace_post

    def init_traces(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        n_pre = self.weight.shape[1]
        n_post = self.weight.shape[0]
        return (
            torch.zeros(batch, n_pre, device=device),
            torch.zeros(batch, n_post, device=device),
        )


class RewardModulatedSTDP(nn.Module):
    """Three-factor reward-modulated STDP.

    Eligibility traces accumulate as in pair STDP, but weight updates are
    gated by a scalar reward prediction error (RPE) signal `delta`.

    This implements: dW = eta * delta * eligibility_trace
    where eligibility_trace = A_plus * x_pre * post_spike - A_minus * x_post * pre_spike.

    Biological interpretation: dopamine release (delta > 0) potentiates synapses
    that were recently co-active; dopamine dip (delta < 0) causes depression.

    Args:
        n_pre, n_post:   Network dimensions.
        tau_e:           Eligibility trace decay time constant (ms).
        eta:             Global learning rate.
        Other params:    Same as PairSTDP.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        tau_e: float = 100.0,
        eta: float = 1e-3,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.eta = eta
        self.w_min = w_min
        self.w_max = w_max

        self.decay_pre = torch.exp(torch.tensor(-dt / tau_plus))
        self.decay_post = torch.exp(torch.tensor(-dt / tau_minus))
        self.decay_e = torch.exp(torch.tensor(-dt / tau_e))

        self.weight = nn.Parameter(
            torch.rand(n_post, n_pre) * (w_max - w_min) + w_min
        )

    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        trace_pre: torch.Tensor,
        trace_post: torch.Tensor,
        eligibility: torch.Tensor,
        reward_signal: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single timestep update with reward modulation.

        Args:
            pre_spikes:     (batch, n_pre).
            post_spikes:    (batch, n_post).
            trace_pre:      (batch, n_pre) spike trace.
            trace_post:     (batch, n_post) spike trace.
            eligibility:    (n_post, n_pre) eligibility trace carried across time.
            reward_signal:  Scalar or (batch,) RPE from the RL agent.

        Returns:
            dW:          Weight update (n_post, n_pre).
            trace_pre, trace_post, eligibility: Updated states.
        """
        trace_pre = trace_pre * self.decay_pre + pre_spikes
        trace_post = trace_post * self.decay_post + post_spikes

        # Instantaneous eligibility kernel (batch-averaged)
        ltp_kernel = self.A_plus * torch.einsum("bp,bq->pq", post_spikes, trace_pre)
        ltd_kernel = self.A_minus * torch.einsum("bp,bq->pq", trace_post, pre_spikes)
        e_instant = (ltp_kernel - ltd_kernel) / max(pre_spikes.shape[0], 1)

        eligibility = eligibility * self.decay_e + e_instant

        dW = self.eta * reward_signal * eligibility
        with torch.no_grad():
            self.weight.data = torch.clamp(self.weight.data + dW, self.w_min, self.w_max)

        return dW, trace_pre, trace_post, eligibility

    def init_states(
        self, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_pre = self.weight.shape[1]
        n_post = self.weight.shape[0]
        return (
            torch.zeros(batch, n_pre, device=device),
            torch.zeros(batch, n_post, device=device),
            torch.zeros(n_post, n_pre, device=device),
        )
