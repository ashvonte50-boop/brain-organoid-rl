import torch
import numpy as np
import matplotlib.pyplot as plt
from neuron_models.izhikevich_network import IzhikevichNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

N_NEURONS = 1000
N_INH = 200
P_CONN = 0.10  # legacy parameter, not used with distance-dependent
G_EXC = 5.0
G_INH = -10.0
NOISE_STD = 8.0
DT = 0.5
T_MS = 2000
N_STEPS = int(T_MS / DT)

print("[INFO] Building network with distance-dependent connectivity...")
net = IzhikevichNetwork(
    n_neurons=N_NEURONS,
    n_inh=N_INH,
    p_conn=P_CONN,  # still passed but ignored internally
    g_exc=G_EXC,
    g_inh=G_INH,
    noise_std=NOISE_STD,
    dt=DT,
    device=DEVICE
).to(DEVICE)

# Validate connectivity structure
print("\n[CONNECTIVITY STATS]")
W = net.W.detach().cpu().numpy()
n_connections = (W != 0).sum()
print(f"  Total connections: {n_connections} / {N_NEURONS**2} ({100*n_connections/N_NEURONS**2:.1f}%)")

# Excitatory vs inhibitory out-degree
exc_out = (W[:net.n_exc, :] != 0).sum(axis=1)
inh_out = (W[net.n_exc:, :] != 0).sum(axis=1)
print(f"  Excitatory out-degree: {exc_out.mean():.1f} ± {exc_out.std():.1f}")
print(f"  Inhibitory out-degree: {inh_out.mean():.1f} ± {inh_out.std():.1f}")

# Distance distribution of connections
pos = net.positions.cpu().numpy()
connected_pairs = np.argwhere(W != 0)
distances = []
for i, j in connected_pairs:
    d = np.sqrt(((pos[i] - pos[j])**2).sum())
    distances.append(d)
distances = np.array(distances)

print(f"  Connection distance: {distances.mean():.3f} ± {distances.std():.3f} (max possible = {np.sqrt(2):.3f})")

# Plot distance distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(distances, bins=50, color='steelblue', edgecolor='black')
axes[0].set_xlabel("Connection distance (normalized)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of connection distances")

# 2D visualization of a single neuron's connections
neuron_idx = 0  # first excitatory neuron
targets = np.where(W[neuron_idx, :] != 0)[0]
axes[1].scatter(pos[:, 0], pos[:, 1], s=5, c='lightgray', label='all neurons')
axes[1].scatter(pos[neuron_idx, 0], pos[neuron_idx, 1], s=100, c='red', marker='*', label='source')
axes[1].scatter(pos[targets, 0], pos[targets, 1], s=20, c='blue', label='targets')
axes[1].set_xlabel("X position")
axes[1].set_ylabel("Y position")
axes[1].set_title(f"Connections from neuron {neuron_idx}")
axes[1].legend()

plt.tight_layout()
plt.savefig("validate_distance_structure.png", dpi=150)
print("[SAVED] validate_distance_structure.png")

# Simulate resting dynamics
print("\n[INFO] Simulating resting dynamics...")
spikes = net.simulate(N_STEPS)

times, neurons = np.where(spikes > 0)
rates = spikes.sum(axis=0) / (T_MS / 1000.0)

print(f"\n[STATS] Mean rate (all):  {rates.mean():.2f} Hz")
print(f"[STATS] Mean rate (exc):  {rates[:N_NEURONS-N_INH].mean():.2f} Hz")
print(f"[STATS] Mean rate (inh):  {rates[N_NEURONS-N_INH:].mean():.2f} Hz")

# Plot raster
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

axes[0].scatter(times * DT, neurons, s=1, c='black', alpha=0.3)
axes[0].set_ylabel("Neuron index")
axes[0].set_title(f"Resting dynamics: {N_NEURONS} Izhikevich neurons with distance-dependent connectivity")
axes[0].set_xlim(0, T_MS)
axes[0].set_ylim(0, N_NEURONS)
axes[0].axhline(N_NEURONS - N_INH, color='red', linestyle='--', alpha=0.5, label='inh boundary')
axes[0].legend()

# Population firing rate
window = 50
bin_edges = np.arange(0, T_MS + window, window)
pop_rate, _ = np.histogram(times * DT, bins=bin_edges)
pop_rate = pop_rate / (window / 1000.0) / N_NEURONS

axes[1].plot(bin_edges[:-1], pop_rate, 'k-', linewidth=1)
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Pop rate (Hz/neuron)")
axes[1].set_title("Population firing rate")

plt.tight_layout()
plt.savefig("validate_resting_distance.png", dpi=150)
print("[SAVED] validate_resting_distance.png")