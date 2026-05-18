import torch
import numpy as np
import matplotlib.pyplot as plt
from neuron_models.izhikevich_network import IzhikevichNetwork

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# ============================================================
# NETWORK PARAMETERS
# ============================================================

N_NEURONS = 1000
N_INH = 200

G_EXC = 5.0
G_INH = -10.0

NOISE_STD = 4.0
DT = 0.5

# ============================================================
# BUILD NETWORK
# ============================================================

print("[INFO] Building network...")

net = IzhikevichNetwork(
    n_neurons=N_NEURONS,
    n_inh=N_INH,
    g_exc=G_EXC,
    g_inh=G_INH,
    noise_std=NOISE_STD,
    dt=DT,
    device=DEVICE
).to(DEVICE)

# ============================================================
# STDP PARAMETERS
# ============================================================

net.init_stdp(
    A_plus=0.0025,
    A_minus=0.0027,
    tau_plus=20.0,
    tau_minus=20.0,
    w_max=2.0
)

# ============================================================
# SAVE BASELINE
# ============================================================

W_baseline = net.W.data[:net.n_exc, :net.n_exc].cpu().clone()

# ============================================================
# SPARSE PATTERNS
# ============================================================

PATTERN_SIZE = 10

pattern_A = np.arange(0, 10)
pattern_B = np.arange(20, 30)
pattern_C = np.arange(40, 50)

patterns = {
    "A": pattern_A,
    "B": pattern_B,
    "C": pattern_C
}

# ============================================================
# TRAINING SETTINGS
# ============================================================

TRAIN_PATTERN = pattern_A

N_PRESENTATIONS = 100

STIM_STRENGTH = 8.0
STIM_DURATION = 20       # ms
INTER_PRESENTATION = 200 # ms

stim_steps = int(STIM_DURATION / DT)
rest_steps = int(INTER_PRESENTATION / DT)

# ============================================================
# RESET NETWORK STATE
# ============================================================

net.reset_state()

if net.stdp_enabled:
    net.pre_trace.zero_()
    net.post_trace.zero_()

# ============================================================
# SPIKE RECORDING
# ============================================================

spike_history = []

print(f"[INFO] Training sparse assembly pattern...")

# ============================================================
# TRAINING LOOP
# ============================================================

for presentation in range(N_PRESENTATIONS):

    # --------------------------------------------------------
    # TEMPORALLY JITTERED SPARSE STIMULATION
    # --------------------------------------------------------

    jitter_times = np.random.randint(0, stim_steps // 2, size=len(TRAIN_PATTERN))

    for t in range(stim_steps):

        stim = torch.zeros(N_NEURONS, device=DEVICE)

        # weak background noise
        stim += torch.randn(N_NEURONS, device=DEVICE) * 0.5

        # sparse temporally-jittered assembly activation
        for idx, neuron in enumerate(TRAIN_PATTERN):

            if t >= jitter_times[idx]:
                stim[neuron] += STIM_STRENGTH

        spikes = net.forward_stdp(stim)

        spike_history.append(spikes.detach().cpu().numpy())

    # --------------------------------------------------------
    # REST PERIOD
    # --------------------------------------------------------

    for t in range(rest_steps):

        stim = torch.randn(N_NEURONS, device=DEVICE) * 0.3

        spikes = net.forward_stdp(stim)

        spike_history.append(spikes.detach().cpu().numpy())

    if (presentation + 1) % 10 == 0:
        print(f"  Completed {presentation + 1}/{N_PRESENTATIONS}")

# ============================================================
# TRAINED WEIGHTS
# ============================================================

W_trained = net.W.data[:net.n_exc, :net.n_exc].cpu()

delta_W = W_trained - W_baseline

# ============================================================
# WEIGHT ANALYSIS
# ============================================================

print("\n[INFO] Analyzing learned structure...")

within_A = delta_W[np.ix_(pattern_A, pattern_A)].flatten()

A_to_B = delta_W[np.ix_(pattern_A, pattern_B)].flatten()

A_to_C = delta_W[np.ix_(pattern_A, pattern_C)].flatten()

random_control = delta_W[100:110, 200:210].flatten()

print(f"\n[LEARNING RESULTS]")
print(f"Within A Δw:     {within_A.mean():.4f} ± {within_A.std():.4f}")
print(f"A → B Δw:        {A_to_B.mean():.4f} ± {A_to_B.std():.4f}")
print(f"A → C Δw:        {A_to_C.mean():.4f} ± {A_to_C.std():.4f}")
print(f"Random Δw:       {random_control.mean():.4f} ± {random_control.std():.4f}")

ratio = within_A.mean() / (A_to_B.mean() + 1e-6)

print(f"\nSelective learning ratio: {ratio:.2f}x")

if within_A.mean() > A_to_B.mean():
    print("[PASS] Selective assembly potentiation detected")
else:
    print("[FAIL] No selective assembly learning")

# ============================================================
# PARTIAL CUE RECALL TEST
# ============================================================

print("\n[INFO] Testing partial cue recall...")

net.reset_state()

partial_cue = pattern_A[:3]

recall_spikes = []

TEST_STEPS = 100

for t in range(TEST_STEPS):

    stim = torch.randn(N_NEURONS, device=DEVICE) * 0.3

    if t < 20:
        for neuron in partial_cue:
            stim[neuron] += STIM_STRENGTH

    spikes = net.forward_stdp(stim)

    recall_spikes.append(spikes.detach().cpu().numpy())

recall_spikes = np.array(recall_spikes)

assembly_activity = recall_spikes[:, pattern_A].sum()
background_activity = recall_spikes[:, 100:110].sum()

print(f"Assembly spikes:  {assembly_activity}")
print(f"Background spikes:{background_activity}")

if assembly_activity > background_activity * 2:
    print("[PASS] Partial cue reactivated trained assembly")
else:
    print("[FAIL] No clear assembly recall")

# ============================================================
# RASTER PREP
# ============================================================

spike_history = np.array(spike_history)

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ------------------------------------------------------------
# Weight distribution
# ------------------------------------------------------------

axes[0, 0].hist(delta_W.flatten().numpy(), bins=100)
axes[0, 0].axvline(0, color='red', linestyle='--')
axes[0, 0].set_title("STDP Weight Changes")
axes[0, 0].set_xlabel("Δw")

# ------------------------------------------------------------
# Within vs between
# ------------------------------------------------------------

means = [
    within_A.mean(),
    A_to_B.mean(),
    A_to_C.mean(),
    random_control.mean()
]

labels = [
    "Within A",
    "A→B",
    "A→C",
    "Random"
]

axes[0, 1].bar(labels, means)
axes[0, 1].set_title("Selective Learning")

# ------------------------------------------------------------
# Weight matrix
# ------------------------------------------------------------

im = axes[0, 2].imshow(
    W_trained.numpy(),
    aspect='auto',
    cmap='viridis',
    vmin=0,
    vmax=2
)

axes[0, 2].set_title("Trained Exc→Exc Weights")

plt.colorbar(im, ax=axes[0, 2])

# ------------------------------------------------------------
# Delta weights
# ------------------------------------------------------------

im2 = axes[1, 0].imshow(
    delta_W.numpy(),
    aspect='auto',
    cmap='bwr',
    vmin=-0.2,
    vmax=0.2
)

axes[1, 0].set_title("Weight Changes (ΔW)")

plt.colorbar(im2, ax=axes[1, 0])

# ------------------------------------------------------------
# Raster plot
# ------------------------------------------------------------

spike_times, neuron_ids = np.where(spike_history > 0)

axes[1, 1].scatter(
    spike_times,
    neuron_ids,
    s=1
)

axes[1, 1].set_title("Network Raster")
axes[1, 1].set_xlabel("Time step")
axes[1, 1].set_ylabel("Neuron")

# ------------------------------------------------------------
# Recall test
# ------------------------------------------------------------

axes[1, 2].plot(recall_spikes[:, pattern_A].sum(axis=1), label='Pattern A')
axes[1, 2].plot(recall_spikes[:, 100:110].sum(axis=1), label='Background')

axes[1, 2].legend()
axes[1, 2].set_title("Partial Cue Recall")

plt.tight_layout()

plt.savefig("stdp_selective_learning.png", dpi=150)

print("\n[SAVED] stdp_selective_learning.png")
print("[DONE] Experiment complete.")