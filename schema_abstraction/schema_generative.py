"""Functional 'Hippocampal-Cortical' Generative Layer.

Replaces the failed VAE with a simple online autoencoder that learns
from REM replay data.

Architecture:
  Input:  cortical state vector (net.v or spike raster, size _ccf.N_EXC)
  Hidden: 64 ReLU units
  Output: _ccf.N_EXC linear units (reconstructed cortical state)

Training data: during each REM phase, every 10th step, capture the
cortical state vector and the corresponding memory label.

Tests:
  1. Reconstruction: after training, feed noisy Schema Core cue,
     measure MSE to true full memory pattern.
  2. Schema Completion: feed noisy Schema Core cue, measure whether
     output activates the correct unique set.
  3. Novel Memory Generalization: after Memory E is encoded, feed its
     noisy core cue; autoencoder should reconstruct it despite never
     seeing it during training.
"""

import numpy as np
import torch

from compare_catastrophic_forgetting import DEVICE
import compare_catastrophic_forgetting as _ccf

# DEV_MODE speed-up
try:
    from compare_catastrophic_forgetting import DEV_MODE as _DEV
except ImportError:
    _DEV = False

# ── Config ──────────────────────────────────────────────────────────────
N_HIDDEN = 64                # hidden layer size
AE_LEARNING_RATE = 0.001     # learning rate
AE_EPOCHS = 5                # training epochs per REM phase
AE_CAPTURE_INTERVAL = 10     # capture every Nth step during REM
_AE_PROBE_STEPS = 5 if _DEV else 20


class CorticalAutoencoder:
    """Online autoencoder that learns cortical state patterns during REM.

    Architecture:
      Input (_ccf.N_EXC) → Dense(64, ReLU) → Dense(_ccf.N_EXC, linear)

    The autoencoder is trained online during REM phases and tested on
    schema completion and novel memory generalization.
    """

    def __init__(self, n_input=None, n_hidden=N_HIDDEN, lr=AE_LEARNING_RATE):
        if n_input is None:
            n_input = _ccf.N_EXC
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = lr

        # Simple two-layer autoencoder: input → hidden → output
        self.W_enc = np.random.randn(n_hidden, n_input).astype(np.float32) * 0.01
        self.b_enc = np.zeros(n_hidden, dtype=np.float32)
        self.W_dec = np.random.randn(n_input, n_hidden).astype(np.float32) * 0.01
        self.b_dec = np.zeros(n_input, dtype=np.float32)

        # Training history
        self.mse_history = []
        self.training_data = []  # list of (input_vector, memory_label)
        self.n_rem_phases = 0
        self.reconstruction_scores = {}  # memory_label -> mse
        self.schema_completion_scores = {}
        self.novel_generalization_score = 0.0

    def encode(self, x):
        """Input vector x → hidden representation."""
        h = self.W_enc @ x + self.b_enc
        h = np.maximum(0, h)  # ReLU
        return h

    def decode(self, h):
        """Hidden representation → reconstruction."""
        return self.W_dec @ h + self.b_dec

    def forward(self, x):
        """Full forward pass: input → reconstruction."""
        return self.decode(self.encode(x))

    def train_on_batch(self, x):
        """Single training step on one input vector."""
        h = self.encode(x)
        recon = self.decode(h)
        error = x - recon
        mse = float(np.mean(error ** 2))

        # Gradients (manual backprop)
        dL_drec = -2.0 * error / max(len(x), 1)

        # Decoder gradients
        dW_dec = np.outer(dL_drec, h)
        db_dec = dL_drec

        # Encoder gradients (through ReLU)
        dh = self.W_dec.T @ dL_drec
        dh[h <= 0] = 0  # ReLU backward
        dW_enc = np.outer(dh, x)
        db_enc = dh

        # Update
        self.W_dec -= self.lr * dW_dec
        self.b_dec -= self.lr * db_dec
        self.W_enc -= self.lr * dW_enc
        self.b_enc -= self.lr * db_enc

        self.mse_history.append(mse)
        return mse

    def capture_cortical_state(self, net, memory_label=None):
        """Capture a cortical state vector from the network.

        Reads the spike raster (net.spikes) as the state representation.
        """
        with torch.no_grad():
            state = net.spikes[:_ccf.N_EXC].float().cpu().numpy().copy()
        # Ensure nonzero
        if state.sum() < 0.01:
            state = np.random.randn(_ccf.N_EXC).astype(np.float32) * 0.01
        if memory_label is not None:
            self.training_data.append((state, memory_label))
        return state

    def train_rem_epoch(self, net, assemblies, memory_labels_true=None):
        """Train the autoencoder on captured REM replay data.

        For each stored training example:
          1. Encode the cortical state.
          2. Decode to reconstruct.
          3. Minimize MSE.

        Runs AE_EPOCHS full passes over the training data.
        """
        if not self.training_data:
            return 0.0

        epoch_mses = []
        for epoch in range(AE_EPOCHS):
            mse_sum = 0.0
            n = 0
            for state, label in self.training_data:
                mse = self.train_on_batch(state)
                mse_sum += mse
                n += 1
            epoch_mses.append(mse_sum / max(1, n))

        self.n_rem_phases += 1
        return float(np.mean(epoch_mses)) if epoch_mses else 0.0

    def test_reconstruction(self, net, core_mask, assemblies, asm_idx=0):
        """Test 1: Feed a noisy Schema Core cue, measure MSE to true pattern.

        Returns the MSE between autoencoder output and the true memory pattern.
        """
        if core_mask is None or asm_idx >= len(assemblies):
            return 0.0

        core_exc = core_mask[core_mask < _ccf.N_EXC]
        asm = assemblies[asm_idx]
        asm_exc = asm[asm < _ccf.N_EXC]

        # Build noisy core cue
        n_cue = max(1, min(10, len(core_exc)))
        cue = np.random.choice(core_exc, n_cue, replace=False)
        stim_np = np.zeros(_ccf.N_EXC, dtype=np.float32)
        stim_np[cue] = 1.0
        # Add small noise
        stim_np += np.random.randn(_ccf.N_EXC).astype(np.float32) * 0.1

        # Get autoencoder output
        recon = self.forward(stim_np)

        # True memory pattern
        true_pattern = np.zeros(_ccf.N_EXC, dtype=np.float32)
        true_pattern[asm_exc] = 1.0

        mse = float(np.mean((recon - true_pattern) ** 2))
        self.reconstruction_scores[asm_idx] = mse
        return mse

    def test_schema_completion(self, net, core_mask, assemblies):
        """Test 2: Noisy Schema Core → correct unique set activation.

        For each assembly, feed a noisy core cue.  Measure whether the
        output activates the correct unique set more than incorrect ones.

        Returns a dict mapping assembly index to completion accuracy.
        """
        if core_mask is None:
            return {}

        core_exc = core_mask[core_mask < _ccf.N_EXC]
        n_mem = len(assemblies)
        results = {}

        for aidx in range(min(n_mem, 8)):
            asm = assemblies[aidx]
            asm_exc = asm[asm < _ccf.N_EXC]
            unique_exc = np.setdiff1d(asm_exc, core_exc)
            if len(unique_exc) == 0:
                continue

            # Build noisy core cue
            n_cue = max(1, min(10, len(core_exc)))
            cue = np.random.choice(core_exc, n_cue, replace=False)
            stim_np = np.zeros(_ccf.N_EXC, dtype=np.float32)
            stim_np[cue] = 1.0

            recon = self.forward(stim_np)

            # Measure activation of each unique set
            self_unique = float(np.mean(recon[unique_exc]))
            other_unique_means = []
            for oi in range(n_mem):
                if oi == aidx:
                    continue
                oasm = assemblies[oi]
                ou = np.setdiff1d(oasm[oasm < _ccf.N_EXC], core_exc)
                if len(ou) > 0:
                    other_unique_means.append(float(np.mean(recon[ou])))

            correct = self_unique > max(other_unique_means) if other_unique_means else True
            results[aidx] = {
                "self_unique_activation": float(self_unique),
                "mean_other_activation": float(np.mean(other_unique_means)) if other_unique_means else 0.0,
                "completion_correct": bool(correct),
                "completion_accuracy": float(self_unique / max(max(other_unique_means + [1e-10]), 1e-10)),
            }

        self.schema_completion_scores = results
        return results

    def test_novel_generalization(self, net, core_mask, mem_e):
        """Test 3: Novel Memory E generalization.

        Feed noisy core cue, measure whether autoencoder output
        reconstructs Memory E's unique set despite never seeing it.
        """
        if core_mask is None or mem_e is None:
            return 0.0

        core_exc = core_mask[core_mask < _ccf.N_EXC]
        mem_e_exc = mem_e[mem_e < _ccf.N_EXC]
        unique_e = np.setdiff1d(mem_e_exc, core_exc)

        n_cue = max(1, min(10, len(core_exc)))
        cue = np.random.choice(core_exc, n_cue, replace=False)
        stim_np = np.zeros(_ccf.N_EXC, dtype=np.float32)
        stim_np[cue] = 1.0

        recon = self.forward(stim_np)

        # Compare unique-E activation to random background
        if len(unique_e) > 0:
            unique_e_act = float(np.mean(recon[unique_e]))
        else:
            unique_e_act = 0.0

        bg = np.random.choice(_ccf.N_EXC, 100, replace=False)
        bg_act = float(np.mean(recon[bg]))

        score = unique_e_act / max(bg_act, 1e-10)
        self.novel_generalization_score = score
        return score

    def summary(self):
        return {
            "n_steps": len(self.mse_history),
            "final_mse": float(np.mean(self.mse_history[-20:])) if len(self.mse_history) >= 20 else 0.0,
            "mse_history": self.mse_history,
            "n_rem_phases": self.n_rem_phases,
            "n_training_samples": len(self.training_data),
            "reconstruction_scores": {str(k): v for k, v in self.reconstruction_scores.items()},
            "schema_completion": {str(k): v for k, v in self.schema_completion_scores.items()},
            "novel_generalization_score": self.novel_generalization_score,
        }


def init_generative_layer(net):
    """Attach a CorticalAutoencoder to the net if not already present."""
    if not hasattr(net, "_generative_layer"):
        net._generative_layer = CorticalAutoencoder()


def capture_rem_state(net, assemblies, memory_label, core_mask=None):
    """Called during REM phase to capture cortical state for training."""
    gl = getattr(net, "_generative_layer", None)
    if gl is None:
        return
    gl.capture_cortical_state(net, memory_label)


def train_autoencoder(net):
    """Train the autoencoder on captured REM data."""
    gl = getattr(net, "_generative_layer", None)
    if gl is None:
        return
    gl.train_rem_epoch(net, None)


# ── Hook callbacks ─────────────────────────────────────────────────────

def _gen_baseline_hook(net, assemblies, n_mem, j=-1, **_):
    """Initialise the generative layer."""  # noqa: D401
    init_generative_layer(net)


def _gen_post_replay_hook(net, assemblies, n_mem, j, **_):
    """Train generative layer on captured REM replay states."""
    gl = getattr(net, "_generative_layer", None)
    if gl is None:
        return
    # Capture state for each available assembly
    for aidx in range(min(j + 1, 8)):
        gl.capture_cortical_state(net, memory_label=aidx)
    # Train
    gl.train_rem_epoch(net, assemblies)


def _gen_final_hook(net, assemblies, n_mem, **_):
    """Measure final generative scores and store in hook_extra."""
    gl = getattr(net, "_generative_layer", None)
    if gl is None:
        return

    core_mask = getattr(net, "_schema_core_mask", None)

    # Test reconstruction for each assembly
    for aidx in range(min(n_mem, 8)):
        gl.test_reconstruction(net, core_mask, assemblies, aidx)

    # Test schema completion
    gl.test_schema_completion(net, core_mask, assemblies)

    # Test novel memory generalization (if memory E exists)
    test_mem_e = getattr(net, "_test_memory_e", None)
    if test_mem_e is not None:
        gl.test_novel_generalization(net, core_mask, test_mem_e)

    extra = getattr(net, "_hook_extra", None)
    if extra is not None:
        extra["generative_layer"] = gl.summary()
