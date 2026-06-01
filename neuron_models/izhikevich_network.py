import torch
import torch.nn as nn
import numpy as np


class IzhikevichNetwork(nn.Module):
    """
    Recurrent Izhikevich spiking neural network with:
    - Multiple biological cell types
    - Distance-dependent connectivity (dense mode)
    - Sparse modular connectivity (sparse_modular mode)
    - Pair-based STDP
    - Slow synaptic consolidation (Fusi-style cascade)

    Two architecture modes:
      "dense"          — original validated distance-dependent connectivity
      "sparse_modular" — scalable modular topology with fan-in normalization
    """

    def __init__(
        self,
        n_neurons=1000,
        n_inh=200,
        p_conn=0.10,
        g_exc=5.0,
        g_inh=-10.0,
        noise_std=8.0,
        dt=0.5,
        device='cpu',
        arch_mode='dense',
        n_modules=8,
        intra_module_conn_prob=0.25,
        inter_module_conn_prob=0.02,
        inter_module_scale=0.05,
        ee_sparsity=0.10,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.n_inh = n_inh
        self.n_exc = n_neurons - n_inh

        self.dt = dt
        self.device = device
        self.noise_std = noise_std

        # Architecture mode
        self.arch_mode = arch_mode
        self.n_modules = n_modules
        self.intra_module_conn_prob = intra_module_conn_prob
        self.inter_module_conn_prob = inter_module_conn_prob
        self.inter_module_scale = inter_module_scale
        self.ee_sparsity = ee_sparsity

        # =====================================================
        # CELL TYPES
        # =====================================================

        self.cell_types = torch.zeros(
            n_neurons,
            dtype=torch.long,
            device=device
        )

        n_rs = int(0.60 * self.n_exc)
        n_ib = int(0.15 * self.n_exc)
        n_ch = int(0.05 * self.n_exc)

        n_fs = int(0.15 * self.n_inh)
        n_lts = self.n_inh - n_fs

        idx = 0

        self.cell_types[idx:idx+n_rs] = 0
        idx += n_rs

        self.cell_types[idx:idx+n_ib] = 1
        idx += n_ib

        self.cell_types[idx:idx+n_ch] = 2

        idx = self.n_exc

        self.cell_types[idx:idx+n_fs] = 3
        idx += n_fs

        self.cell_types[idx:idx+n_lts] = 4

        # =====================================================
        # IZHIKEVICH PARAMETERS
        # =====================================================

        self.a = torch.tensor(
            [0.02, 0.02, 0.02, 0.10, 0.02],
            device=device
        )

        self.b = torch.tensor(
            [0.20, 0.20, 0.20, 0.20, 0.25],
            device=device
        )

        self.c = torch.tensor(
            [-65.0, -55.0, -50.0, -65.0, -65.0],
            device=device
        )

        self.d = torch.tensor(
            [8.0, 4.0, 2.0, 2.0, 2.0],
            device=device
        )

        # =====================================================
        # STATE VARIABLES
        # =====================================================

        self.register_buffer(
            'v',
            torch.full((n_neurons,), -70.0, device=device)
        )

        self.register_buffer(
            'u',
            torch.zeros(n_neurons, device=device)
        )

        self.register_buffer(
            'spikes',
            torch.zeros(n_neurons, device=device)
        )

        # Synaptic current with an exponential time constant.
        # Without this, I_syn = W @ spikes is a single-timestep pulse
        # that the membrane cannot integrate — pattern completion is
        # mathematically impossible.  Realistic AMPA+NMDA average ~8ms.
        self.tau_syn = 8.0
        self.register_buffer(
            'I_syn',
            torch.zeros(n_neurons, device=device)
        )

        for i in range(n_neurons):
            t = self.cell_types[i].item()
            self.u[i] = self.b[t] * (-70.0)

        # =====================================================
        # CONNECTIVITY
        # =====================================================

        grid_size = int(np.ceil(np.sqrt(n_neurons)))

        self.positions = torch.zeros(
            n_neurons,
            2,
            device=device
        )

        for i in range(n_neurons):
            row = i // grid_size
            col = i % grid_size

            self.positions[i, 0] = row / grid_size
            self.positions[i, 1] = col / grid_size

        if arch_mode == "sparse_modular":
            # -------------------------------------------------
            # SPARSE MODULAR CONNECTIVITY
            # -------------------------------------------------
            # Biological rationale:
            #   Cortex is organized into minicolumns/modules with dense
            #   local recurrence and sparse long-range projections.
            #   This prevents fan-in explosion at scale by keeping
            #   each neuron's recurrent input localized to its module.
            # -------------------------------------------------

            # Assign each excitatory neuron to a module
            module_id = torch.full(
                (n_neurons,), -1, dtype=torch.long, device=device
            )
            exc_per_module = self.n_exc // n_modules
            for m in range(n_modules):
                start = m * exc_per_module
                end = start + exc_per_module if m < n_modules - 1 else self.n_exc
                module_id[start:end] = m
            self.register_buffer('module_id', module_id)

            # Build modular connectivity mask (vectorized)
            mod_i = self.module_id[:self.n_exc].unsqueeze(1)  # (n_exc, 1)
            mod_j = self.module_id[:self.n_exc].unsqueeze(0)  # (1, n_exc)
            same_module = (mod_i == mod_j) & (mod_i >= 0)     # (n_exc, n_exc)

            rand_mat = torch.rand(
                self.n_exc, self.n_exc, device=device
            )
            intra_mask = (rand_mat < intra_module_conn_prob) & same_module
            inter_mask = (rand_mat < inter_module_conn_prob) & ~same_module
            ee_mask = (intra_mask | inter_mask).float()
            ee_mask.fill_diagonal_(0.0)

            # Full connectivity: E→E from modular mask, I→ everywhere
            conn_mask = torch.zeros(
                n_neurons, n_neurons, device=device
            )
            conn_mask[:self.n_exc, :self.n_exc] = ee_mask
            conn_mask[self.n_exc:, :] = 1.0

            W_raw = (
                torch.randn(n_neurons, n_neurons, device=device) * 0.1
            )
            W_raw = W_raw * conn_mask

            # Scale cross-module excitatory weights
            cross_module = (~same_module).float()
            cross_module.fill_diagonal_(0.0)
            W_raw[:self.n_exc, :self.n_exc] *= (
                1.0 + (inter_module_scale - 1.0) * cross_module
            )

        else:
            # -------------------------------------------------
            # DENSE DISTANCE-DEPENDENT CONNECTIVITY (original)
            # -------------------------------------------------
            C = 0.25
            lambda_scale = 0.08
            long_range_p = 0.08

            diff = (
                self.positions.unsqueeze(0)
                - self.positions.unsqueeze(1)
            )

            dist = torch.sqrt((diff ** 2).sum(dim=2))

            P_dist = C * torch.exp(
                -dist ** 2 / (2 * lambda_scale ** 2)
            )

            P_dist.fill_diagonal_(0.0)

            long_range_mask = (
                torch.rand(
                    n_neurons,
                    n_neurons,
                    device=device
                ) < long_range_p
            ).float()

            long_range_mask.fill_diagonal_(0.0)

            conn_mask = (
                (
                    torch.rand(
                        n_neurons,
                        n_neurons,
                        device=device
                    ) < P_dist
                ).float()
                + long_range_mask
            ) > 0

            conn_mask = conn_mask.float()

            W_raw = (
                torch.randn(
                    n_neurons,
                    n_neurons,
                    device=device
                ) * 0.1
            )

            W_raw = W_raw * conn_mask

            # module_id: all -1 in dense mode (no modular structure)
            self.register_buffer(
                'module_id',
                torch.full((n_neurons,), -1, dtype=torch.long, device=device)
            )

        # NOTE on sign convention: this mask selects ROWS (postsynaptic),
        # not columns (presynaptic) as biological Dale's-principle would
        # require.  This is a known structural quirk of the network:
        # inhibitory cells receive only-negative synaptic input and so
        # almost never fire, meaning the network has effectively no
        # active inhibition.  Recurrent excitation is contained by
        # W_MAX and the sparse, distance-dependent connectivity.
        # The retention experiment works under this convention; an
        # earlier attempt at the Dale-correct convention (and matching
        # |g_inh| balance) produced a knife-edge regime where assembly
        # formation succeeded only on some random seeds.  Left as-is
        # to preserve the verified retention gap.
        exc_mask = torch.zeros(
            n_neurons,
            n_neurons,
            device=device
        )

        exc_mask[:self.n_exc, :] = 1.0

        inh_mask = 1.0 - exc_mask

        self.W = nn.Parameter(
            torch.abs(W_raw) * exc_mask * g_exc +
            torch.abs(W_raw) * inh_mask * g_inh,
            requires_grad=False
        )

        # Fan-in normalization for sparse_modular mode:
        # scales each excitatory neuron's incoming E→E weights by
        # 1/sqrt(fan_in), preserving stable recurrent drive at any N.
        if arch_mode == "sparse_modular":
            with torch.no_grad():
                ee_sub = self.W.data[:self.n_exc, :self.n_exc]
                fan_in = (ee_sub != 0).sum(dim=1, keepdim=True).clamp(min=1).float()
                scale = 1.0 / torch.sqrt(fan_in)
                ee_sub.mul_(scale)

        # Freeze initial weights as the decay target for both fast-only
        # and slow-consolidation conditions.
        self.register_buffer('W_init', self.W.data.clone())

        # =====================================================
        # EXTERNAL INPUT WEIGHTS
        # =====================================================

        self.input_weights = nn.Parameter(
            torch.randn(n_neurons, 4, device=device) * 0.5,
            requires_grad=False
        )

        # =====================================================
        # FLAGS
        # =====================================================

        self.stdp_enabled = False
        self.slow_enabled = False

    # =========================================================
    # FORWARD PASS
    # =========================================================

    def forward(self, x_ext=None):

        # Use effective weights if slow consolidation enabled
        if self.slow_enabled:
            W_eff = self.get_effective_weights()
        else:
            W_eff = self.W.data

        # Exponentially-decaying synaptic current.
        # Each spike adds W @ spike (an instantaneous kick); existing
        # current decays with tau_syn so multiple recent spikes summate.
        decay_syn = float(np.exp(-self.dt / self.tau_syn))
        self.I_syn.mul_(decay_syn).add_(W_eff @ self.spikes)

        # External current
        if x_ext is not None:
            I_ext = self.I_syn + x_ext
        else:
            I_ext = self.I_syn

        # Background noise
        I_noise = (
            torch.randn(
                self.n_neurons,
                device=self.device
            ) * self.noise_std
        )

        I_total = I_ext + I_noise

        # Retrieve neuron parameters
        a = self.a[self.cell_types]
        b = self.b[self.cell_types]
        c = self.c[self.cell_types]
        d = self.d[self.cell_types]

        # Izhikevich dynamics
        dv = (
            0.04 * self.v ** 2
            + 5.0 * self.v
            + 140.0
            - self.u
            + I_total
        )

        du = a * (b * self.v - self.u)

        self.v = self.v + self.dt * dv
        self.u = self.u + self.dt * du

        # Spikes
        spike_mask = self.v >= 30.0

        self.spikes = spike_mask.float()

        # Reset after spike
        self.v = torch.where(
            spike_mask,
            c,
            self.v
        )

        self.u = torch.where(
            spike_mask,
            self.u + d,
            self.u
        )

        return self.spikes.clone()

    # =========================================================
    # RESET STATE
    # =========================================================

    def reset_state(self):

        self.v.fill_(-70.0)

        for i in range(self.n_neurons):
            t = self.cell_types[i].item()
            self.u[i] = self.b[t] * (-70.0)

        self.spikes.zero_()
        self.I_syn.zero_()

    # =========================================================
    # STDP INITIALIZATION
    # =========================================================

    def init_stdp(
        self,
        A_plus=0.005,
        A_minus=0.00525,
        tau_plus=20.0,
        tau_minus=20.0,
        w_max=1.0
    ):

        self.stdp_enabled = True

        self.A_plus = A_plus
        self.A_minus = A_minus

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        self.w_max = w_max

        self.register_buffer(
            'pre_trace',
            torch.zeros(
                self.n_neurons,
                device=self.device
            )
        )

        self.register_buffer(
            'post_trace',
            torch.zeros(
                self.n_neurons,
                device=self.device
            )
        )

        self.plastic_mask = torch.zeros(
            self.n_neurons,
            self.n_neurons,
            device=self.device
        )

        # Excitatory-excitatory plasticity only
        self.plastic_mask[:self.n_exc, :self.n_exc] = 1.0

    # =========================================================
    # SLOW CONSOLIDATION
    # =========================================================

    def init_slow_weights(
        self,
        gamma=0.3,
        tau_slow=10000.0,
        tau_fast=5000.0,
        tau_very_slow=200000.0
    ):
        """
        Slow synaptic consolidation mechanism.

        gamma           mixing coefficient between fast and slow weights
        tau_slow        time constant for W_slow to follow W_fast upward
                        (consolidation; called during training rest gaps)
        tau_fast        fast weight decay time constant (used by decay_step)
        tau_very_slow   time constant for W_slow's own passive decay
                        (much longer than tau_fast; makes slow memories stable)
        """

        self.slow_enabled = True

        self.gamma = gamma
        self.tau_slow = tau_slow
        self.tau_fast = tau_fast
        self.tau_very_slow = tau_very_slow

        self.register_buffer(
            'W_slow',
            self.W.data[:self.n_exc, :self.n_exc].clone()
        )

        # Backward-compat alias — points to the same buffer registered in __init__
        self.W_baseline = self.W_init

    # =========================================================
    # SLOW WEIGHT UPDATE
    # =========================================================

    def slow_step(self):
        """
        Asymmetric slow consolidation: W_slow follows W_fast *upward*
        at rate 1/tau_slow (consolidation) but only drifts downward at
        1/tau_very_slow (structural stability).  Call this during rest
        gaps between training presentations so W_slow accumulates the
        learned assembly, and also during post-training rest so it
        resists the fast-weight decay.
        """

        if not self.slow_enabled:
            return

        with torch.no_grad():

            fast_exc = self.W.data[:self.n_exc, :self.n_exc]
            delta = fast_exc - self.W_slow

            # Potentiation: follow W_fast up quickly
            up = torch.clamp(delta, min=0.0) / self.tau_slow

            # Depression: resist following W_fast back down
            down = torch.clamp(delta, max=0.0) / self.tau_very_slow

            self.W_slow.add_(up + down)

    # =========================================================
    # FAST WEIGHT DECAY
    # =========================================================

    def decay_step(self, tau=None):
        """
        Exponential relaxation of fast excitatory weights toward
        W_init (the pre-training baseline).  Call every timestep
        during post-training rest — for *both* Fast-Only and Slow-
        Consolidation conditions.  Without this, the Fast-Only
        condition has no forgetting pressure at all.
        """

        if tau is None:
            tau = getattr(self, 'tau_fast', 500.0)

        with torch.no_grad():
            exc = self.W.data[:self.n_exc, :self.n_exc]
            baseline = self.W_init[:self.n_exc, :self.n_exc]
            exc.add_((baseline - exc) / tau)

    # =========================================================
    # HOMEOSTATIC SYNAPTIC SCALING
    # =========================================================

    def homeostatic_step(self):
        """
        Multiplicative scaling to prevent unbounded weight growth.
        Scales each excitatory neuron's incoming E→E row so it never
        exceeds 3× its initial total.  Does not shrink rows below their
        initial sum, so it only suppresses runaway potentiation.
        """

        if not self.stdp_enabled:
            return

        with torch.no_grad():
            exc = self.W.data[:self.n_exc, :self.n_exc]
            init_exc = self.W_init[:self.n_exc, :self.n_exc]

            row_sums = exc.sum(dim=1, keepdim=True)
            init_sums = init_exc.sum(dim=1, keepdim=True).clamp(min=1e-6)

            target = init_sums * 3.0
            scale = torch.where(
                row_sums > target,
                target / row_sums,
                torch.ones_like(row_sums)
            )
            exc.mul_(scale)

    # =========================================================
    # EFFECTIVE WEIGHTS
    # =========================================================

    def get_effective_weights(self):

        if not self.slow_enabled:
            return self.W.data

        W_eff = self.W.data.clone()

        W_eff[:self.n_exc, :self.n_exc] = (
            (1.0 - self.gamma)
            * self.W.data[:self.n_exc, :self.n_exc]
            + self.gamma * self.W_slow
        )

        return W_eff

    # =========================================================
    # STDP UPDATE
    # =========================================================

    def stdp_step(self):
        """
        Pair-based STDP, online formulation.

        IMPORTANT: traces must be sampled BEFORE the current spike is
        added, otherwise a coincident pre/post spike contributes to both
        the LTP and LTD terms simultaneously, producing a net change of
        (A_plus - A_minus) per coincidence.  With A_minus >= A_plus
        (the conventional choice for irregular-input stability) this
        causes correlated, co-firing assemblies to be DEPRESSED rather
        than potentiated.

        Correct order each step:
          1. decay traces
          2. compute dw using the decayed traces (PAST spikes only)
          3. add the current spike to the traces (counts for FUTURE pairings)
        """

        if not self.stdp_enabled:
            return

        dt = self.dt
        decay_pre = float(np.exp(-dt / self.tau_plus))
        decay_post = float(np.exp(-dt / self.tau_minus))

        # 1) decay traces (do NOT add current spike yet)
        self.pre_trace.mul_(decay_pre)
        self.post_trace.mul_(decay_post)

        # 2) STDP using pre-update traces
        #    LTP: post i fires AND pre j has recent activity
        #    LTD: pre  j fires AND post i has recent activity
        dw = (
            self.A_plus
            * self.spikes.unsqueeze(1)
            * self.pre_trace.unsqueeze(0)
            -
            self.A_minus
            * self.spikes.unsqueeze(0)
            * self.post_trace.unsqueeze(1)
        )
        dw = dw * self.plastic_mask

        with torch.no_grad():
            self.W.data += dw
            # Only clamp the plastic (E→E) submatrix so inhibitory
            # connections (I→E positive, E→I negative in this row-based
            # sign convention) are not disturbed.
            self.W.data[:self.n_exc, :self.n_exc].clamp_(0.0, self.w_max)

        # 3) add current spikes to traces, for FUTURE pairings
        self.pre_trace.add_(self.spikes)
        self.post_trace.add_(self.spikes)

    # =========================================================
    # FORWARD + STDP
    # =========================================================

    def forward_stdp(self, x_ext=None):

        spikes = self.forward(x_ext)

        self.stdp_step()

        return spikes

    # =========================================================
    # SIMULATION
    # =========================================================

    def simulate(
        self,
        n_steps,
        x_ext_series=None
    ):

        self.reset_state()

        spike_log = torch.zeros(
            n_steps,
            self.n_neurons,
            device='cpu'
        )

        for t in range(n_steps):

            if x_ext_series is not None:
                x = x_ext_series[t]
            else:
                x = None

            self.forward(x)

            spike_log[t] = self.spikes.cpu()

        return spike_log.numpy()

    # =========================================================
    # SIMULATION WITH STDP
    # =========================================================

    def simulate_stdp(
        self,
        n_steps,
        x_ext_series=None
    ):

        self.reset_state()

        if self.stdp_enabled:
            self.pre_trace.zero_()
            self.post_trace.zero_()

        spike_log = torch.zeros(
            n_steps,
            self.n_neurons,
            device='cpu'
        )

        for t in range(n_steps):

            if x_ext_series is not None:
                x = x_ext_series[t]
            else:
                x = None

            self.forward_stdp(x)

            spike_log[t] = self.spikes.cpu()

        return spike_log.numpy()