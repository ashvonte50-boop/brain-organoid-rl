import torch
import numpy as np
import matplotlib.pyplot as plt
from norse.torch.functional.lif import lif_step, LIFParameters, LIFState

print("Norse LIF defaults:")
p = LIFParameters()
print(f"  tau_mem_inv = {p.tau_mem_inv.item():.4f}  => tau_mem = {1000/p.tau_mem_inv.item():.1f} ms")
print(f"  v_th        = {p.v_th.item():.4f}")
print(f"  v_reset     = {p.v_reset.item():.4f}")

DT = 1.0
T_MS = 500
N_STEPS = int(T_MS / DT)

currents = np.linspace(0.0, 2.0, 21)
rates = []

# 1x1 weights: single input neuron, no recurrence
w_in  = torch.ones(1, 1)
w_rec = torch.zeros(1, 1)

for I_const in currents:
    p = LIFParameters()
    s = LIFState(
        z=torch.zeros(1),
        v=torch.zeros(1),
        i=torch.zeros(1),
    )

    spike_count = 0
    for t in range(N_STEPS):
        x = torch.tensor([I_const], dtype=torch.float32)
        z_out, s = lif_step(x, s, w_in, w_rec, p, dt=DT * 1e-3)
        spike_count += z_out.item()
    
    rate = spike_count / (T_MS / 1000.0)
    rates.append(rate)
    if I_const <= 0.5:
        print(f"I={I_const:.2f} => {rate:.1f} Hz")

plt.figure(figsize=(8, 4))
plt.plot(currents, rates, 'ko-', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Input current (a.u.)")
plt.ylabel("Firing rate (Hz)")
plt.title("Single LIF neuron f-I curve")
plt.tight_layout()
plt.savefig("diagnostic_fI_curve.png", dpi=150)
print("\nSaved: diagnostic_fI_curve.png")

# Diagnosis
print("\n--- DIAGNOSIS ---")
if rates[0] > 5:
    print("FAIL: Neuron fires spontaneously with ZERO input.")
    print("  FIX: Check if v_reset is too high or there is a bias.")
elif max(rates) < 10 and currents[-1] >= 1.5:
    print("FAIL: Neuron barely fires even with strong input.")
    print("  FIX: Lower v_th or increase tau_mem.")
elif rates[-1] > 0 and abs(rates[-1] - rates[0]) < 5:
    print("FAIL: Rate barely changes across input range.")
    print("  FIX: Neuron is saturated. Reduce input scaling or adjust thresholds.")
else:
    print("PASS: f-I curve looks graded.")
