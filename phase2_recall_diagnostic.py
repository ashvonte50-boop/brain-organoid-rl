"""
PHASE 2 DIAGNOSTIC: Why doesn't the cue recruit the non-cued assembly?

Train the network, then inspect recall in detail:
  - per-cell spike counts (not just averages)
  - sub-threshold voltage trace of a representative non-cued cell
  - the actual synaptic input arriving at non-cued cells

If the I_syn pulses are sub-threshold individually and don't accumulate,
the synapse model needs an exponential time constant.
"""

import torch
import numpy as np
from neuron_models.izhikevich_network import IzhikevichNetwork

torch.set_num_threads(4)
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu")

N_NEURONS, N_INH = 300, 60
N_EXC = N_NEURONS - N_INH
PATTERN = np.arange(0, 20)
CUE = np.arange(0, 5)
NON_CUED = np.arange(5, 20)

A_PLUS, A_MINUS, W_MAX = 0.010, 0.005, 2.0
N_PRES = 40
STIM = 15.0
STIM_MS, REST_MS = 50, 300
DT = 0.5
CUE_STRENGTH = 15.0
RECALL_MS = 200

stim_steps = int(STIM_MS / DT)
rest_steps = int(REST_MS / DT)
recall_steps = int(RECALL_MS / DT)


net = IzhikevichNetwork(
    n_neurons=N_NEURONS, n_inh=N_INH,
    g_exc=5.0, g_inh=-10.0, noise_std=4.0, dt=DT, device=DEVICE
).to(DEVICE)
net.init_stdp(A_plus=A_PLUS, A_minus=A_MINUS,
              tau_plus=20.0, tau_minus=20.0, w_max=W_MAX)

# Train
for _ in range(N_PRES):
    jitter = np.random.randint(0, max(1, stim_steps // 2), size=len(PATTERN))
    for t in range(stim_steps):
        stim = torch.randn(N_NEURONS, device=DEVICE) * 0.5
        for idx, n in enumerate(PATTERN):
            if t >= jitter[idx]:
                stim[n] += STIM
        net.forward(stim)
        net.stdp_step()
    for _ in range(rest_steps):
        net.forward(torch.randn(N_NEURONS, device=DEVICE) * 0.3)


# Inspect connectivity: cue->non-cued synapses
print("=" * 60)
print(" CONNECTIVITY: cue -> non-cued")
print("=" * 60)
W = net.W.data
cue_to_nc = W[NON_CUED][:, CUE]   # rows = post non-cued, cols = pre cue
n_per_nc = (cue_to_nc > 0).sum(dim=1).cpu().numpy()
print(f"  Synapses per non-cued cell from cue cells:")
for i, n in enumerate(NON_CUED):
    incoming = cue_to_nc[i]
    nz = incoming[incoming > 0].cpu().numpy()
    print(f"    cell {n:3d}: {n_per_nc[i]} synapses, weights = {nz}")


# Recall with detailed logging
print("\n" + "=" * 60)
print(" RECALL DIAGNOSTIC")
print("=" * 60)

net.noise_std = 1.5
net.reset_state()
net.pre_trace.zero_()
net.post_trace.zero_()

spikes_log = np.zeros((recall_steps, N_NEURONS), dtype=np.float32)
v_log = np.zeros((recall_steps, N_NEURONS), dtype=np.float32)
isyn_log = np.zeros((recall_steps, N_NEURONS), dtype=np.float32)

for t in range(recall_steps):
    stim = torch.zeros(N_NEURONS, device=DEVICE)
    stim[CUE] = CUE_STRENGTH

    # Replicate forward to capture I_syn before noise
    W_eff = net.W.data
    I_syn = W_eff @ net.spikes
    isyn_log[t] = I_syn.cpu().numpy()

    net.forward(stim)
    spikes_log[t] = net.spikes.cpu().numpy()
    v_log[t] = net.v.cpu().numpy()


# --- per-cell stats ---
print(f"\n  Cue cells (0..4) spike counts over {recall_steps} steps:")
for c in CUE:
    print(f"    cell {c}: {int(spikes_log[:, c].sum())} spikes")
print(f"\n  Non-cued (5..19) spike counts:")
for c in NON_CUED:
    print(f"    cell {c}: {int(spikes_log[:, c].sum())} spikes  "
          f"v_max={v_log[:, c].max():.1f}  "
          f"I_syn_max={isyn_log[:, c].max():.2f}  "
          f"I_syn_mean={isyn_log[:, c].mean():.3f}")

print(f"\n  Background (cells 100-109):")
for c in range(100, 110):
    print(f"    cell {c}: {int(spikes_log[:, c].sum())} spikes  "
          f"v_max={v_log[:, c].max():.1f}  "
          f"I_syn_max={isyn_log[:, c].max():.2f}")


# --- inspect single non-cued cell time-course ---
focus = None
# pick non-cued cell with highest # synapses from cue
best_i = int(np.argmax(n_per_nc))
focus = int(NON_CUED[best_i])
print(f"\n  TIME COURSE for cell {focus} "
      f"({n_per_nc[best_i]} synapses from cue):")
print(f"    {'t':>5} {'v':>8} {'I_syn':>8} {'spike':>6} {'cue_spikes':>10}")
for t in range(0, min(80, recall_steps)):
    cue_active = int(spikes_log[t, CUE].sum())
    spk = int(spikes_log[t, focus])
    print(f"    {t:>5} {v_log[t, focus]:>8.2f} {isyn_log[t, focus]:>8.2f} "
          f"{spk:>6d} {cue_active:>10d}")
