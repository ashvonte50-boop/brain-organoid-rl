import torch
import torch.nn as nn
import numpy as np

class IzhikevichNetwork(nn.Module):
    """
    Recurrent network of Izhikevich neurons with cell-type heterogeneity.
    Replaces the LIF network in your pipeline.
    """
    def __init__(self, n_neurons=1000, n_inh=200, p_conn=0.10, 
                 g_exc=5.0, g_inh=-20.0, noise_std=5.0,
                 dt=0.5, device='cpu'):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_inh = n_inh
        self.n_exc = n_neurons - n_inh
        self.dt = dt
        self.device = device
        self.noise_std = noise_std
        
        # Cell type parameters (Izhikevich)
        self.cell_types = torch.zeros(n_neurons, dtype=torch.long, device=device)
        # 60% RS, 15% IB, 5% CH, 15% FS, 5% LTS among excitatory
        n_rs = int(0.60 * self.n_exc)
        n_ib = int(0.15 * self.n_exc)
        n_ch = int(0.05 * self.n_exc)
        n_fs = int(0.15 * self.n_inh)
        n_lts = self.n_inh - n_fs
        
        idx = 0
        self.cell_types[idx:idx+n_rs] = 0; idx += n_rs      # RS
        self.cell_types[idx:idx+n_ib] = 1; idx += n_ib      # IB
        self.cell_types[idx:idx+n_ch] = 2; idx += n_ch      # CH
        # Inhibitory
        idx = self.n_exc
        self.cell_types[idx:idx+n_fs] = 3; idx += n_fs       # FS
        self.cell_types[idx:idx+n_lts] = 4; idx += n_lts    # LTS
        
        # Izhikevich parameters per cell type
        self.a = torch.tensor([0.02, 0.02, 0.02, 0.10, 0.02], device=device)
        self.b = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.25], device=device)
        self.c = torch.tensor([-65.0, -55.0, -50.0, -65.0, -65.0], device=device)
        self.d = torch.tensor([8.0, 4.0, 2.0, 2.0, 2.0], device=device)
        
        # State variables
        self.register_buffer('v', torch.full((n_neurons,), -70.0, device=device))
        self.register_buffer('u', torch.zeros(n_neurons, device=device))
        self.register_buffer('spikes', torch.zeros(n_neurons, device=device))
        
        # Initialize u according to cell type
        for i in range(n_neurons):
            t = self.cell_types[i].item()
            self.u[i] = self.b[t] * (-70.0)
        
        # Random sparse weight matrix
        W_raw = torch.randn(n_neurons, n_neurons, device=device) * 0.1
        mask = (torch.rand(n_neurons, n_neurons, device=device) < p_conn).float()
        W_raw = W_raw * mask
        W_raw.fill_diagonal_(0.0)
        
        # Excitatory / inhibitory split
        exc_mask = torch.zeros(n_neurons, n_neurons, device=device)
        exc_mask[:self.n_exc, :] = 1.0
        inh_mask = 1.0 - exc_mask
        
        self.W = nn.Parameter(
            torch.abs(W_raw) * exc_mask * g_exc + 
            torch.abs(W_raw) * inh_mask * g_inh,
            requires_grad=False
        )
        
        # Input weights (for external drive)
        self.input_weights = nn.Parameter(
            torch.randn(n_neurons, 4, device=device) * 0.5,  # 4 inputs for cartpole
            requires_grad=False
        )
        
    def forward(self, x_ext=None):
        """
        One Euler step for all neurons.
        x_ext: (n_neurons,) external current, or None for noise only
        """
        # Synaptic current
        I_syn = self.W @ self.spikes
        
        # External input
        if x_ext is not None:
            I_ext = I_syn + x_ext
        else:
            I_ext = I_syn
        
        # Noise
        I_noise = torch.randn(self.n_neurons, device=self.device) * self.noise_std
        
        I_total = I_ext + I_noise
        
        # Izhikevich dynamics per cell type
        a = self.a[self.cell_types]
        b = self.b[self.cell_types]
        c = self.c[self.cell_types]
        d = self.d[self.cell_types]
        
        dv = 0.04 * self.v ** 2 + 5.0 * self.v + 140.0 - self.u + I_total
        du = a * (b * self.v - self.u)
        
        self.v = self.v + self.dt * dv
        self.u = self.u + self.dt * du
        
        # Spike detection
        spike_mask = self.v >= 30.0
        self.spikes = spike_mask.float()
        
        # Reset
        self.v = torch.where(spike_mask, c, self.v)
        self.u = torch.where(spike_mask, self.u + d, self.u)
        
        return self.spikes.clone()
    
    def reset_state(self):
        """Reset to resting state."""
        self.v.fill_(-70.0)
        for i in range(self.n_neurons):
            t = self.cell_types[i].item()
            self.u[i] = self.b[t] * (-70.0)
        self.spikes.zero_()
    
    def simulate(self, n_steps, x_ext_series=None):
        """
        Run simulation for n_steps.
        x_ext_series: (n_steps, n_neurons) or None
        Returns: (n_steps, n_neurons) spike raster
        """
        self.reset_state()
        spike_log = torch.zeros(n_steps, self.n_neurons, device='cpu')
        
        for t in range(n_steps):
            if x_ext_series is not None:
                x = x_ext_series[t]
            else:
                x = None
            self.forward(x)
            spike_log[t] = self.spikes.cpu()
        
        return spike_log.numpy()