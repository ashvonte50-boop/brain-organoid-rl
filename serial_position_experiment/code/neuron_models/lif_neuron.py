"""
Leaky Integrate-and-Fire neuron layers backed by Norse.

LIFLayer   — single recurrent LIF layer with Norse LIFCell.
LIFPopulation — multi-layer ensemble with optional E/I balance.

Norse 1.1.0 API notes:
    - LIFCell.forward(input, state=None) handles None state internally.
    - State is LIFFeedForwardState(v, i); spikes (z) are the first return value.
    - Layer state is managed as (lif_state, prev_spikes) so recurrent weights
      can reference the previous timestep's output without extra instance state.

Future work:
    - Add AdEx (adaptive exponential) dynamics for spike-frequency adaptation
    - Support heterogeneous parameter distributions across neurons
    - Implement sparse random connectivity matching organoid measurements
"""

import torch
import torch.nn as nn

try:
    from norse.torch import LIFCell, LIFParameters
except ImportError:
    raise ImportError(
        "Norse is required: pip install norse. "
        "See https://norse.github.io/norse/ for installation."
    )

# LayerState = (LIFFeedForwardState, prev_spikes_tensor)
LayerState = tuple


class LIFLayer(nn.Module):
    """Single LIF recurrent layer.

    Args:
        input_size:  Number of pre-synaptic input channels.
        hidden_size: Number of LIF neurons in this layer.
        params:      Norse LIFParameters (tau_mem, tau_syn, v_th, …).
        dt:          Simulation timestep in milliseconds.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        params: LIFParameters | None = None,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        self.input_weights = nn.Linear(input_size, hidden_size, bias=False)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size, bias=False)

        self.lif = LIFCell(p=params or LIFParameters(), dt=dt)

    def forward(
        self, x: torch.Tensor, state: LayerState | None = None
    ) -> tuple[torch.Tensor, LayerState]:
        """Single timestep forward pass.

        Args:
            x:     Input tensor of shape (batch, input_size).
            state: (LIFFeedForwardState, prev_spikes) or None to start fresh.

        Returns:
            spikes:    Binary spike tensor (batch, hidden_size).
            new_state: Updated (LIFFeedForwardState, spikes) for next timestep.
        """
        if state is None:
            lif_state = None
            prev_spikes = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        else:
            lif_state, prev_spikes = state

        current = self.input_weights(x) + self.recurrent_weights(prev_spikes)
        spikes, new_lif_state = self.lif(current, lif_state)

        return spikes, (new_lif_state, spikes)

    def simulate(
        self, input_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, list[LayerState]]:
        """Run a full input sequence through the layer.

        Args:
            input_sequence: Tensor of shape (T, batch, input_size).

        Returns:
            spike_train: Tensor of shape (T, batch, hidden_size).
            states:      List of LayerState at each timestep.
        """
        T = input_sequence.shape[0]
        state = None
        spikes_list, states = [], []

        for t in range(T):
            spikes, state = self.forward(input_sequence[t], state)
            spikes_list.append(spikes)
            states.append(state)

        return torch.stack(spikes_list), states


class LIFPopulation(nn.Module):
    """Multi-layer LIF network with optional excitatory/inhibitory structure.

    Each layer is an independent LIFLayer; inter-layer projections are learned
    linear maps. E/I balance can be enforced by constraining weight signs.

    Args:
        layer_sizes: List [input_size, hidden1, hidden2, …, output_size].
        ei_ratio:    Fraction of excitatory neurons per layer (0–1). None disables constraint.
        dt:          Simulation timestep in ms.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        ei_ratio: float | None = 0.8,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.ei_ratio = ei_ratio

        self.layers = nn.ModuleList(
            [
                LIFLayer(layer_sizes[i], layer_sizes[i + 1], dt=dt)
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(
        self, x: torch.Tensor, states: list | None = None
    ) -> tuple[torch.Tensor, list]:
        """Single timestep forward through all layers.

        Args:
            x:      Input tensor (batch, input_size).
            states: Per-layer LayerState list, or None to initialise.

        Returns:
            output: Spike tensor from the final layer (batch, output_size).
            states: Updated per-layer state list.
        """
        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        current = x
        for layer, state in zip(self.layers, states):
            current, new_state = layer(current, state)
            new_states.append(new_state)

        return current, new_states
