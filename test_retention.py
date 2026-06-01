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
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
# INITIALIZE STDP
# ============================================================

print("[INFO] Initializing STDP...")

net.init_stdp(
    A_plus=0.0025,
    A_minus=0.0027,
    tau_plus=20.0,
    tau_minus=20.0,
    w_max=2.0
)


# ============================================================
# INITIALIZE SLOW CONSOLIDATION
# ============================================================

print("[INFO] Initializing slow consolidation...")

net.init_slow_weights(
    gamma=0.4,
    tau_slow=5000.0,
    tau_fast=3000.0
)


# ============================================================
# BASELINE WEIGHTS
# ============================================================

W_baseline = (
    net.W.data[:net.n_exc, :net.n_exc]
    .detach()
    .cpu()
    .clone()
)


# ============================================================
# PATTERN DEFINITIONS
# ============================================================

PATTERN_A = np.arange(0, 10)

N_PRESENTATIONS = 50

STIM_STRENGTH = 8.0

STIM_DURATION_MS = 20
INTER_PRESENTATION_MS = 200

stim_steps = int(STIM_DURATION_MS / DT)
rest_steps = int(INTER_PRESENTATION_MS / DT)


# ============================================================
# TRAINING
# ============================================================

print("[INFO] Training pattern A with slow stabilization...")

net.reset_state()

if net.stdp_enabled:
    net.pre_trace.zero_()
    net.post_trace.zero_()

for presentation in range(N_PRESENTATIONS):

    # Temporal jitter for realistic asynchronous firing
    jitter_times = np.random.randint(
        0,
        max(1, stim_steps // 2),
        size=len(PATTERN_A)
    )

    # --------------------------------------------------------
    # STIMULATION PHASE
    # --------------------------------------------------------

    for t in range(stim_steps):

        stim = (
            torch.randn(
                N_NEURONS,
                device=DEVICE
            ) * 0.5
        )

        for idx, neuron in enumerate(PATTERN_A):

            if t >= jitter_times[idx]:
                stim[neuron] += STIM_STRENGTH

        net.forward_stdp(stim)

    # --------------------------------------------------------
    # REST PHASE BETWEEN PRESENTATIONS
    # --------------------------------------------------------

    for t in range(rest_steps):

        background = (
            torch.randn(
                N_NEURONS,
                device=DEVICE
            ) * 0.3
        )

        net.forward_stdp(background)

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    if (presentation + 1) % 10 == 0:
        print(f"  Completed {presentation + 1}/{N_PRESENTATIONS}")


# ============================================================
# WEIGHT ANALYSIS AFTER TRAINING
# ============================================================

print("\n[INFO] Measuring learned structure...")

W_trained = (
    net.W.data[:net.n_exc, :net.n_exc]
    .detach()
    .cpu()
    .clone()
)

delta_W = W_trained - W_baseline

within_A = delta_W[
    :10,
    :10
].flatten().numpy()

random_control = delta_W[
    100:110,
    200:210
].flatten().numpy()

print(f"Within-pattern Δw: {within_A.mean():.4f} ± {within_A.std():.4f}")
print(f"Random-control Δw: {random_control.mean():.4f} ± {random_control.std():.4f}")


# ============================================================
# IMMEDIATE RECALL TEST
# ============================================================

print("\n[INFO] Testing immediate recall...")

net.reset_state()

if net.stdp_enabled:
    net.pre_trace.zero_()
    net.post_trace.zero_()

TEST_DURATION_MS = 100

test_steps = int(TEST_DURATION_MS / DT)

spikes_immediate = []

for t in range(test_steps):

    stim = torch.zeros(
        N_NEURONS,
        device=DEVICE
    )

    # Partial cue: first 3 neurons only
    stim[0:3] = 8.0

    net.forward_stdp(stim)

    spikes_immediate.append(
        net.spikes.detach().cpu().numpy()
    )

spikes_immediate = np.array(spikes_immediate)

assembly_immediate = spikes_immediate[:, PATTERN_A].sum()

background_immediate = spikes_immediate[
    :,
    100:200
].sum()

print(f"Assembly spikes (immediate):  {assembly_immediate:.1f}")
print(f"Background spikes (immediate): {background_immediate:.1f}")


# ============================================================
# LONG REST / DECAY PHASE
# ============================================================

print("\n[INFO] Simulating long rest period...")

REST_DURATION_STEPS = 20000

for t in range(REST_DURATION_STEPS):

    background = (
        torch.randn(
            N_NEURONS,
            device=DEVICE
        ) * 0.2
    )

    net.forward_stdp(background)

    if (t + 1) % 5000 == 0:
        print(f"  Rest step {t + 1}/{REST_DURATION_STEPS}")


# ============================================================
# POST-REST RECALL TEST
# ============================================================

print("\n[INFO] Testing recall after rest...")

net.reset_state()

if net.stdp_enabled:
    net.pre_trace.zero_()
    net.post_trace.zero_()

spikes_postrest = []

for t in range(test_steps):

    stim = torch.zeros(
        N_NEURONS,
        device=DEVICE
    )

    stim[0:3] = 8.0

    net.forward_stdp(stim)

    spikes_postrest.append(
        net.spikes.detach().cpu().numpy()
    )

spikes_postrest = np.array(spikes_postrest)

assembly_postrest = spikes_postrest[:, PATTERN_A].sum()

background_postrest = spikes_postrest[
    :,
    100:200
].sum()

print(f"Assembly spikes (post-rest):  {assembly_postrest:.1f}")
print(f"Background spikes (post-rest): {background_postrest:.1f}")


# ============================================================
# RETENTION METRIC
# ============================================================

retention_ratio = (
    assembly_postrest
    / (assembly_immediate + 1e-8)
)

print(f"\nRetention ratio: {retention_ratio:.3f}")

if retention_ratio > 0.70:
    print("[PASS] Strong long-term retention")
elif retention_ratio > 0.40:
    print("[PARTIAL] Partial retention")
else:
    print("[FAIL] Memory decayed substantially")


# ============================================================
# FINAL WEIGHT ANALYSIS
# ============================================================

W_final = (
    net.W.data[:net.n_exc, :net.n_exc]
    .detach()
    .cpu()
    .clone()
)

delta_final = W_final - W_baseline

within_final = delta_final[
    :10,
    :10
].flatten().numpy()

print(f"\nFinal within-pattern Δw: {within_final.mean():.4f}")


# ============================================================
# PLOTTING
# ============================================================

print("\n[INFO] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


# ------------------------------------------------------------
# WEIGHT CHANGE DISTRIBUTION
# ------------------------------------------------------------

axes[0, 0].hist(
    delta_final.flatten().numpy(),
    bins=100
)

axes[0, 0].axvline(
    0,
    linestyle='--'
)

axes[0, 0].set_title("Weight Change Distribution")

axes[0, 0].set_xlabel("Δw")
axes[0, 0].set_ylabel("Count")


# ------------------------------------------------------------
# TRAINED WEIGHT MATRIX
# ------------------------------------------------------------

im1 = axes[0, 1].imshow(
    W_final.numpy(),
    aspect='auto',
    vmin=0,
    vmax=2
)

axes[0, 1].set_title("Final Excitatory Weights")

plt.colorbar(im1, ax=axes[0, 1])


# ------------------------------------------------------------
# IMMEDIATE RECALL SPIKES
# ------------------------------------------------------------

axes[1, 0].imshow(
    spikes_immediate.T,
    aspect='auto',
    interpolation='nearest'
)

axes[1, 0].set_title("Immediate Recall")

axes[1, 0].set_xlabel("Time Step")
axes[1, 0].set_ylabel("Neuron")


# ------------------------------------------------------------
# POST-REST RECALL SPIKES
# ------------------------------------------------------------

axes[1, 1].imshow(
    spikes_postrest.T,
    aspect='auto',
    interpolation='nearest'
)

axes[1, 1].set_title("Post-Rest Recall")

axes[1, 1].set_xlabel("Time Step")
axes[1, 1].set_ylabel("Neuron")


plt.tight_layout()

plt.savefig(
    "stdp_retention_test.png",
    dpi=150
)

print("[SAVED] stdp_retention_test.png")


# ============================================================
# DONE
# ============================================================

print("\n[DONE] Retention experiment complete.")