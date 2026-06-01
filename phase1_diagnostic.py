"""
PHASE 1 DIAGNOSTIC

Strip everything down to: build network -> train STDP only -> measure weights.
No slow weights, no decay, no replay. Just prove STDP creates assembly structure.

This script answers ONE question:
  "After training, are within-assembly E->E weights larger than outside-assembly?"

If the answer is NO -> STDP is broken and nothing else matters.
If the answer is YES -> the previous failures were in the *measurement*, not the learning.
"""

import torch
import numpy as np
from neuron_models.izhikevich_network import IzhikevichNetwork

torch.set_num_threads(4)
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- config ---
N_NEURONS = 300
N_INH = 60
N_EXC = N_NEURONS - N_INH

PATTERN_SIZE = 20
PATTERN = np.arange(0, PATTERN_SIZE)

# Assembly-learning regime: A_plus > A_minus so correlated firing
# produces NET potentiation.  (A_minus > A_plus is for irregular-
# Poisson stability — it would erase any assembly trying to form.)
A_PLUS = 0.005
A_MINUS = 0.0025
W_MAX = 1.0

N_PRESENTATIONS = 40
STIM_STRENGTH = 12.0
STIM_DURATION_MS = 50
INTERVAL_MS = 300
DT = 0.5

stim_steps = int(STIM_DURATION_MS / DT)
rest_steps = int(INTERVAL_MS / DT)


def main():
    net = IzhikevichNetwork(
        n_neurons=N_NEURONS, n_inh=N_INH,
        g_exc=5.0, g_inh=-40.0,
        noise_std=4.0, dt=DT, device=DEVICE
    ).to(DEVICE)
    net.init_stdp(A_plus=A_PLUS, A_minus=A_MINUS,
                  tau_plus=20.0, tau_minus=20.0, w_max=W_MAX)

    # --- baseline weight stats ---
    W0 = net.W.data[:N_EXC, :N_EXC].clone()
    conn_mask = (W0 > 0).float()

    asm_mask = torch.zeros_like(W0)
    asm_mask[PATTERN[0]:PATTERN[-1]+1, PATTERN[0]:PATTERN[-1]+1] = 1.0
    asm_conn = (asm_mask * conn_mask).bool()
    out_conn = ((1 - asm_mask) * conn_mask).bool()

    n_asm = int(asm_conn.sum().item())
    n_out = int(out_conn.sum().item())

    def mean_in(W):
        return float(W[asm_conn].mean()) if n_asm > 0 else 0.0
    def mean_out(W):
        return float(W[out_conn].mean()) if n_out > 0 else 0.0

    print("="*60)
    print(" BASELINE (before training)")
    print("="*60)
    print(f"  # E->E synapses in assembly:  {n_asm}")
    print(f"  # E->E synapses outside:      {n_out}")
    print(f"  Mean W in-assembly:  {mean_in(W0):.4f}")
    print(f"  Mean W outside:      {mean_out(W0):.4f}")

    # --- training ---
    print("\n" + "="*60)
    print(" TRAINING (STDP only, no decay, no slow)")
    print("="*60)
    print(f"  Pattern: {PATTERN_SIZE} cells, {N_PRESENTATIONS} presentations")
    print(f"  STIM_STRENGTH={STIM_STRENGTH}, A_PLUS={A_PLUS}, W_MAX={W_MAX}")

    spike_counts = []
    for p in range(N_PRESENTATIONS):
        jitter = np.random.randint(0, max(1, stim_steps // 2), size=PATTERN_SIZE)
        stim_spikes = 0
        for t in range(stim_steps):
            stim = torch.randn(N_NEURONS, device=DEVICE) * 0.5
            for idx, n in enumerate(PATTERN):
                if t >= jitter[idx]:
                    stim[n] += STIM_STRENGTH
            net.forward(stim)
            net.stdp_step()
            stim_spikes += int(net.spikes.sum().item())
        # quiet rest
        for t in range(rest_steps):
            net.forward(torch.randn(N_NEURONS, device=DEVICE) * 0.3)
        spike_counts.append(stim_spikes)
        if p in (0, 9, 19, 29, 39):
            Wn = net.W.data[:N_EXC, :N_EXC]
            print(f"  pres {p+1:3d}: stim-spikes={stim_spikes:4d}  "
                  f"<W in>={mean_in(Wn):.3f}  <W out>={mean_out(Wn):.3f}  "
                  f"<pre>={float(net.pre_trace.mean()):.3f}")

    # --- post-training stats ---
    W1 = net.W.data[:N_EXC, :N_EXC]
    print("\n" + "="*60)
    print(" POST-TRAINING")
    print("="*60)
    print(f"  Mean W in-assembly:  {mean_in(W1):.4f}  "
          f"(d = {mean_in(W1) - mean_in(W0):+.4f})")
    print(f"  Mean W outside:      {mean_out(W1):.4f}  "
          f"(d = {mean_out(W1) - mean_out(W0):+.4f})")
    print(f"  Total stim spikes across all presentations: {sum(spike_counts)}")
    print(f"  Mean pre_trace: {float(net.pre_trace.mean()):.4f}")
    print(f"  Max pre_trace:  {float(net.pre_trace.max()):.4f}")

    # count changed synapses
    diff = (W1 - W0).abs()
    n_changed = int((diff > 1e-4).sum().item())
    print(f"  # synapses changed (|dw|>1e-4): {n_changed}")

    # max-out test: how many in-assembly synapses saturated at w_max?
    sat = ((W1 > W_MAX * 0.9) & asm_conn).sum().item()
    print(f"  In-assembly synapses near w_max: {int(sat)} / {n_asm}")

    # --- verdict ---
    print("\n" + "="*60)
    print(" VERDICT")
    print("="*60)
    in_gain = mean_in(W1) - mean_in(W0)
    out_gain = mean_out(W1) - mean_out(W0)
    if in_gain > out_gain + 0.05:
        print(f"  [PASS] STDP selectively potentiated assembly "
              f"(gain {in_gain:+.3f} vs outside {out_gain:+.3f})")
        return True
    else:
        print(f"  [FAIL] No selective potentiation "
              f"(gain {in_gain:+.3f} vs outside {out_gain:+.3f})")
        return False


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
