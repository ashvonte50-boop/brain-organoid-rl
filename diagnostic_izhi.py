import torch
import numpy as np
import matplotlib.pyplot as plt

# Izhikevich parameters for 5 cell types
CELL_TYPES = {
    'RS':  {'a': 0.02, 'b': 0.2,  'c': -65.0, 'd': 8.0,  'label': 'Regular Spiking'},
    'IB':  {'a': 0.02, 'b': 0.2,  'c': -55.0, 'd': 4.0,  'label': 'Intrinsically Bursting'},
    'CH':  {'a': 0.02, 'b': 0.2,  'c': -50.0, 'd': 2.0,  'label': 'Chattering'},
    'FS':  {'a': 0.1,  'b': 0.2,  'c': -65.0, 'd': 2.0,  'label': 'Fast Spiking'},
    'LTS': {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.0,  'label': 'Low-Threshold Spiking'},
}

DT = 0.5  # ms
T_MS = 1000
N_STEPS = int(T_MS / DT)

def run_izhi(I_const, params, n_steps=N_STEPS, dt=DT):
    v = torch.tensor(-70.0)
    u = torch.tensor(params['b'] * -70.0)
    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    
    spikes = 0
    v_trace = []
    
    for t in range(n_steps):
        dv = 0.04 * v ** 2 + 5.0 * v + 140.0 - u + I_const
        du = a * (b * v - u)
        v = v + dt * dv
        u = u + dt * du
        
        if v >= 30.0:
            spikes += 1
            v = torch.tensor(float(c))
            u = u + d
        
        v_trace.append(v.item())
    
    rate = spikes / (T_MS / 1000.0)
    return rate, v_trace

# Sweep currents for each cell type
currents = np.linspace(0.0, 10.0, 41)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx, (name, params) in enumerate(CELL_TYPES.items()):
    ax = axes.flatten()[idx]
    rates = []
    for I in currents:
        rate, _ = run_izhi(torch.tensor(I), params)
        rates.append(rate)
    
    ax.plot(currents, rates, 'o-', markersize=3, label=params['label'])
    ax.set_xlabel("Input current")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"{name}: {params['label']}")
    ax.set_ylim(0, 150)

axes.flatten()[-1].axis('off')
plt.tight_layout()
plt.savefig("diagnostic_izhi_fI.png", dpi=150)
print("Saved: diagnostic_izhi_fI.png")

# Demo trace for RS at I=5
rate, v_trace = run_izhi(torch.tensor(5.0), CELL_TYPES['RS'])
plt.figure(figsize=(10, 3))
plt.plot(np.arange(len(v_trace)) * DT, v_trace)
plt.axhline(30, color='r', linestyle='--', label='threshold')
plt.xlabel("Time (ms)")
plt.ylabel("v (mV)")
plt.title(f"RS neuron at I=5, rate={rate:.1f} Hz")
plt.legend()
plt.tight_layout()
plt.savefig("diagnostic_izhi_trace.png", dpi=150)
print("Saved: diagnostic_izhi_trace.png")