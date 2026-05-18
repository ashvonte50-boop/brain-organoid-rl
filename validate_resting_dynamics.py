import torch
import numpy as np
import matplotlib.pyplot as plt
from neuron_models.izhikevich_network import IzhikevichNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# CORRECTED PARAMETERS — Excitatory neurons were silent (0 Hz)
# Increased G_EXC and NOISE_STD, decreased inhibition magnitude
G_EXC = 5.0       # WAS 2.0
G_INH = -10.0     # WAS -15.0
NOISE_STD = 8.0   # WAS 3.0
N_NEURONS = 1000
N_INH = 200
P_CONN = 0.10
G_EXC = 5.0       # WAS 2.0 — too weak, excitatory neurons never fired
G_INH = -10.0     # WAS -15.0 — too strong, suppression dominated
NOISE_STD = 8.0   # WAS 3.0 — too low, no spontaneous excitatory crossings
DT = 0.5
T_MS = 2000
N_STEPS = int(T_MS / DT)

print(f"[INFO] Parameters: G_EXC={G_EXC}, G_INH={G_INH}, NOISE_STD={NOISE_STD}")
print("[INFO] Building network...")
net = IzhikevichNetwork(
    n_neurons=N_NEURONS,
    n_inh=N_INH,
    p_conn=P_CONN,
    g_exc=G_EXC,
    g_inh=G_INH,
    noise_std=NOISE_STD,
    dt=DT,
    device=DEVICE
).to(DEVICE)

print("[INFO] Simulating resting dynamics...")
spikes = net.simulate(N_STEPS)

# Analysis
times, neurons = np.where(spikes > 0)
rates = spikes.sum(axis=0) / (T_MS / 1000.0)

print(f"\n[STATS] Mean rate (all):  {rates.mean():.2f} Hz")
print(f"[STATS] Mean rate (exc):  {rates[:N_NEURONS-N_INH].mean():.2f} Hz")
print(f"[STATS] Mean rate (inh):  {rates[N_NEURONS-N_INH:].mean():.2f} Hz")

# Cell-type breakdown
print("\n[STATS] By cell type:")
type_names = {0: "RS", 1: "IB", 2: "CH", 3: "FS", 4: "LTS"}
for t in range(5):
    mask = net.cell_types.cpu().numpy() == t
    if mask.sum() > 0:
        type_rate = rates[mask].mean()
        print(f"  {type_names[t]}: {type_rate:.2f} Hz (n={mask.sum()})")

# Plot raster
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

axes[0].scatter(times * DT, neurons, s=1, c='black', alpha=0.3)
axes[0].set_ylabel("Neuron index")
axes[0].set_title(f"Resting dynamics: {N_NEURONS} Izhikevich neurons ({N_NEURONS-N_INH} exc, {N_INH} inh)")
axes[0].set_xlim(0, T_MS)
axes[0].set_ylim(0, N_NEURONS)
axes[0].axhline(N_NEURONS - N_INH, color='red', linestyle='--', alpha=0.5, label='inh boundary')
axes[0].legend()

# Population firing rate
window = 50  # ms
bin_edges = np.arange(0, T_MS + window, window)
pop_rate, _ = np.histogram(times * DT, bins=bin_edges)
pop_rate = pop_rate / (window / 1000.0) / N_NEURONS

axes[1].plot(bin_edges[:-1], pop_rate, 'k-', linewidth=1)
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Pop rate (Hz/neuron)")
axes[1].set_title("Population firing rate")

plt.tight_layout()
plt.savefig("validate_resting_dynamics.png", dpi=150)
print("[SAVED] validate_resting_dynamics.png")

# Power spectral density
from scipy import signal
fs = 1000.0 / DT
f, Pxx = signal.welch(pop_rate, fs, nperseg=min(256, len(pop_rate)//2))
plt.figure(figsize=(8, 4))
plt.semilogy(f, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("Population power spectrum")
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig("validate_power_spectrum.png", dpi=600)
print("[SAVED] validate_power_spectrum.png")