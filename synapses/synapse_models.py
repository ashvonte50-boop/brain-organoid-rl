"""
Synapse models ranging from simple current injection to full conductance dynamics.

CurrentSynapse    — instantaneous current-based synapse (baseline).
ConductanceSynapse — exponential conductance with reversal potential (E/I asymmetry).
STPSynapse        — short-term plasticity (Tsodyks-Markram: depression & facilitation).

Future work:
    - Gap junctions (electrical synapses) for organoid synchrony
    - NMDA receptor model with voltage-gated Mg2+ block
    - Multi-compartment dendritic filtering
"""

import torch
import torch.nn as nn


class CurrentSynapse(nn.Module):
    """Instantaneous current-based synapse.

    Computes I_syn = W @ z where z are pre-synaptic spike counts.
    No temporal dynamics; useful as a fast baseline.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (batch, in_features) pre-synaptic spike tensor.
        Returns:
            current: (batch, out_features) post-synaptic current.
        """
        return torch.nn.functional.linear(spikes, self.weight)


class ConductanceSynapse(nn.Module):
    """Conductance-based synapse with exponential decay.

    Models AMPA/GABA-A dynamics:
        dg/dt = -g / tau_syn  (solved analytically per step)
        I_syn = g * (v - E_rev)

    The post-synaptic voltage `v` must be supplied externally from the neuron state.

    Args:
        tau_syn_ms: Synaptic time constant in milliseconds.
        e_rev:      Reversal potential (mV). +0 for excitatory, -70 for inhibitory.
        dt:         Simulation timestep in ms.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_syn_ms: float = 5.0,
        e_rev: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.e_rev = e_rev
        self.decay = torch.tensor(torch.exp(torch.tensor(-dt / tau_syn_ms)))
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.05
        )

    def forward(
        self,
        spikes: torch.Tensor,
        conductance: torch.Tensor,
        voltage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spikes:      (batch, in_features) spike tensor.
            conductance: (batch, out_features) conductance state from previous step.
            voltage:     (batch, out_features) post-synaptic membrane voltage.

        Returns:
            current:         Synaptic current (batch, out_features).
            new_conductance: Updated conductance state.
        """
        delta_g = torch.nn.functional.linear(spikes, self.weight)
        new_conductance = conductance * self.decay + delta_g
        current = new_conductance * (voltage - self.e_rev)
        return current, new_conductance


class STPSynapse(nn.Module):
    """Short-term plasticity synapse (Tsodyks-Markram model).

    Captures both synaptic depression (resource depletion) and facilitation
    (residual calcium accumulation) on timescales of 100–1000 ms.

    Variables:
        x — fraction of available synaptic resources  (depression)
        u — utilisation parameter                      (facilitation)

    Args:
        tau_rec_ms:  Recovery time constant (ms) — controls depression.
        tau_fac_ms:  Facilitation time constant (ms). 0 disables facilitation.
        U:           Baseline utilisation fraction.
        dt:          Simulation timestep in ms.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_rec_ms: float = 200.0,
        tau_fac_ms: float = 0.0,
        U: float = 0.5,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.U = U
        self.tau_fac_ms = tau_fac_ms
        self.decay_rec = torch.exp(torch.tensor(-dt / tau_rec_ms))
        self.decay_fac = (
            torch.exp(torch.tensor(-dt / tau_fac_ms))
            if tau_fac_ms > 0
            else torch.tensor(0.0)
        )
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )

    def forward(
        self,
        spikes: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step STP update.

        Args:
            spikes: (batch, in_features) pre-synaptic spikes.
            x:      (batch, in_features) resource availability state.
            u:      (batch, in_features) utilisation state.

        Returns:
            current: (batch, out_features) effective synaptic current.
            x_new:   Updated resource state.
            u_new:   Updated utilisation state.
        """
        # Facilitation update
        u_new = u * self.decay_fac + self.U * (1 - u * self.decay_fac) * spikes

        # Depression update: released resources are consumed
        released = u_new * x * spikes
        x_new = x * self.decay_rec + (1 - self.decay_rec) - released

        effective_spikes = released
        current = torch.nn.functional.linear(effective_spikes, self.weight)
        return current, x_new, u_new
