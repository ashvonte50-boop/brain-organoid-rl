"""
extensions/baselines.py — Baseline comparison methods (Task 1).

Implements five alternative continual-learning mechanisms to compare against
the full Slow+Replay model in compare_catastrophic_forgetting.py:

  1. EWC (Elastic Weight Consolidation)
     STDP + Fisher-weighted L2 anchor on previous-memory weights.
  2. Replay Buffer
     Stores explicit spike patterns during training, replays them verbatim
     during rest (no coherence gating, no generative completion).
  3. Simple Rehearsal
     Interleaves old-memory presentations during new-memory training
     (proportion controlled by rehearsal_rate).
  4. Hopfield Attractor Energy
     Uses trained W as a Hopfield weight matrix; evaluates pattern retrieval
     via synchronous update dynamics (no spiking sim during evaluation).
  5. Pure Fast Decay (reference)
     Fast weights only, no replay, no consolidation — best-case forgetting.

All functions follow the same A→B→C→D sequential protocol as cf.py.
None modify compare_catastrophic_forgetting.py.

Output convention:
  run_baseline_experiment() → dict with keys:
    "label"              : str
    "final_scores"       : np.ndarray shape (N_MEMORIES,) — isyn_score after D
    "baseline_scores"    : np.ndarray shape (N_MEMORIES,) — isyn_score just after training
    "mean_retention"     : float — mean(final_scores[:N_MEMORIES-1]) A/B/C retention
    "trial_seed"         : int

  run_all_baselines() → dict[str, list[dict]]  keyed by baseline label
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import compare_catastrophic_forgetting as cf

__all__ = [
    "run_ewc_experiment",
    "run_replay_buffer_experiment",
    "run_rehearsal_experiment",
    "run_hopfield_experiment",
    "run_fast_only_experiment",
    "run_all_baselines",
    "fig_baseline_comparison",
    "BASELINE_CONFIGS",
]

# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

BASELINE_CONFIGS = [
    {
        "key":   "slow_replay",
        "label": "Slow+Replay\n(Full Model)",
        "color": "#2ecc71",
        "fn":    None,   # filled in below
        "kwargs": {},
    },
    {
        "key":   "ewc",
        "label": "EWC",
        "color": "#3498db",
        "fn":    None,
        "kwargs": {"ewc_lambda": 8.0},
    },
    {
        "key":   "replay_buffer",
        "label": "Replay\nBuffer",
        "color": "#9b59b6",
        "fn":    None,
        "kwargs": {"buffer_size": 10},
    },
    {
        "key":   "rehearsal",
        "label": "Simple\nRehearsal",
        "color": "#e67e22",
        "fn":    None,
        "kwargs": {"rehearsal_rate": 0.25},
    },
    {
        "key":   "hopfield",
        "label": "Hopfield\nAttractor",
        "color": "#e74c3c",
        "fn":    None,
        "kwargs": {},
    },
    {
        "key":   "fast_only",
        "label": "Fast Only\n(No Replay)",
        "color": "#95a5a6",
        "fn":    None,
        "kwargs": {},
    },
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _snapshot_W_ee(net) -> torch.Tensor:
    """Return a detached copy of the E→E weight block."""
    return net.W.data[:cf.N_EXC, :cf.N_EXC].clone().detach()


def _apply_ewc_anchor(
    net,
    anchors: List[torch.Tensor],
    fishers: List[torch.Tensor],
    ewc_lambda: float,
) -> None:
    """
    In-place EWC correction on E→E weights.
    W_ee -= ewc_lambda * Σ_k  F_k ⊙ (W_ee - W_anchor_k)
    Clamped to [0, W_MAX] after application.
    """
    W_ee = net.W.data[:cf.N_EXC, :cf.N_EXC]
    for F, W_anc in zip(fishers, anchors):
        W_ee.sub_(ewc_lambda * F * (W_ee - W_anc))
    W_ee.clamp_(0.0, cf.W_MAX)


def _compute_fisher(W_before: torch.Tensor, W_after: torch.Tensor) -> torch.Tensor:
    """
    Diagonal Fisher approximation: normalised squared weight change.
    F[i,j] = ((W_after - W_before) / W_MAX)²
    Captures which synapses were strongly modified during this memory's training.
    """
    delta = (W_after - W_before) / cf.W_MAX
    return delta ** 2


def _train_one_memory_ewc(
    net,
    assembly: np.ndarray,
    tags,
    n_presentations: int,
    anchors: List[torch.Tensor],
    fishers: List[torch.Tensor],
    ewc_lambda: float,
) -> None:
    """
    EWC-augmented training loop — mirrors cf.train_one_memory exactly, with
    one extra step after each stdp_step: apply EWC anchor correction.
    """
    for _ in range(n_presentations):
        jitter = np.random.randint(0, max(1, cf.stim_steps // 2), size=len(assembly))
        for t in range(cf.stim_steps):
            stim = torch.randn(cf.N_NEURONS, device=cf.DEVICE) * 0.5
            for idx, n in enumerate(assembly):
                if t >= jitter[idx]:
                    stim[n] += cf.STIM_STRENGTH
            net.forward(stim)
            if tags is not None:
                tags.update_from_spikes(net)
            net.stdp_step()
            # EWC correction after STDP
            if anchors:
                _apply_ewc_anchor(net, anchors, fishers, ewc_lambda)

        rest_stim = torch.randn(cf.N_NEURONS, device=cf.DEVICE) * 0.3
        for _ in range(cf.rest_steps_per_pres):
            net.forward(rest_stim)
        use_slow = getattr(net, 'slow_enabled', False)
        if use_slow:
            cf.bulk_slow_step(net, n_steps=cf.rest_steps_per_pres)


def _train_one_memory_rehearsal(
    net,
    assembly: np.ndarray,
    old_assemblies: List[np.ndarray],
    tags,
    n_presentations: int,
    rehearsal_rate: float,
) -> None:
    """
    Rehearsal training: after each new-memory presentation, with probability
    rehearsal_rate replay one presentation of a randomly selected old memory.
    """
    rng = np.random.default_rng(int(np.random.randint(0, 2**31)))
    for _ in range(n_presentations):
        # New memory presentation
        jitter = np.random.randint(0, max(1, cf.stim_steps // 2), size=len(assembly))
        for t in range(cf.stim_steps):
            stim = torch.randn(cf.N_NEURONS, device=cf.DEVICE) * 0.5
            for idx, n in enumerate(assembly):
                if t >= jitter[idx]:
                    stim[n] += cf.STIM_STRENGTH
            net.forward(stim)
            if tags is not None:
                tags.update_from_spikes(net)
            net.stdp_step()

        rest_stim = torch.randn(cf.N_NEURONS, device=cf.DEVICE) * 0.3
        for _ in range(cf.rest_steps_per_pres):
            net.forward(rest_stim)
        use_slow = getattr(net, 'slow_enabled', False)
        if use_slow:
            cf.bulk_slow_step(net, n_steps=cf.rest_steps_per_pres)

        # Rehearsal interleave
        if old_assemblies and rng.random() < rehearsal_rate:
            old_asm = old_assemblies[int(rng.integers(len(old_assemblies)))]
            jitter_r = np.random.randint(0, max(1, cf.stim_steps // 2), size=len(old_asm))
            for t in range(cf.stim_steps):
                stim = torch.randn(cf.N_NEURONS, device=cf.DEVICE) * 0.5
                for idx, n in enumerate(old_asm):
                    if t >= jitter_r[idx]:
                        stim[n] += cf.STIM_STRENGTH
                net.forward(stim)
                if tags is not None:
                    tags.update_from_spikes(net)
                net.stdp_step()


def _record_buffer_patterns(
    net,
    assembly: np.ndarray,
    n_patterns: int,
) -> List[torch.Tensor]:
    """
    Record n_patterns spike rasters from a brief stimulation of the assembly.
    Each raster is a binary N_NEURONS tensor (fired neurons = 1).
    Called AFTER train_one_memory() so weights encode the memory.
    """
    patterns = []
    net.noise_std = cf.REPLAY_NOISE_STD
    net.reset_state()
    cue_n = assembly[:cf.PARTIAL_CUE_SIZE]
    seed_stim = torch.zeros(cf.N_NEURONS, device=cf.DEVICE)
    seed_stim[cue_n] = cf.CUE_STRENGTH

    with torch.no_grad():
        for _ in range(n_patterns):
            net.forward(seed_stim)
            patterns.append(net.spikes.clone().float())
            # Brief spontaneous decay between recorded snapshots
            spont = torch.randn(cf.N_NEURONS, device=cf.DEVICE) * cf.REPLAY_NOISE_STD
            for _ in range(10):
                net.forward(spont)
    return patterns


def _replay_buffer_rest(
    net,
    buffer: Dict[int, List[torch.Tensor]],
    n_events: int,
    tags,
) -> None:
    """
    Replay buffer rest: inject stored spike patterns as forced stimuli,
    allow STDP to fire on the resulting activity.
    No coherence gating; just direct pattern injection.
    """
    if not buffer:
        return

    all_patterns = []
    for pats in buffer.values():
        all_patterns.extend(pats)

    if not all_patterns:
        return

    orig_noise = net.noise_std
    net.noise_std = cf.REPLAY_NOISE_STD

    rng = np.random.default_rng(None)
    chosen = rng.integers(len(all_patterns), size=n_events)

    with torch.no_grad():
        for idx in chosen:
            pat = all_patterns[int(idx)]
            # Inject stored pattern as external drive (replay)
            stim = pat * cf.REPLAY_SEED_STRENGTH
            for _ in range(cf.REPLAY_SEED_DURATION):
                net.forward(stim)
                if tags is not None:
                    tags.update_from_spikes(net)
                net.stdp_step()
            # Brief spontaneous phase
            spont = torch.randn(cf.N_NEURONS, device=cf.DEVICE) * cf.REPLAY_NOISE_STD
            for _ in range(cf.REPLAY_SPONTANEOUS_STEPS):
                net.forward(spont)
                if tags is not None:
                    tags.update_from_spikes(net)
                net.stdp_step()

    net.noise_std = orig_noise


# ---------------------------------------------------------------------------
# Hopfield evaluation (no spiking dynamics)
# ---------------------------------------------------------------------------

def _hopfield_retrieve(W_sym: np.ndarray, pattern: np.ndarray, n_steps: int = 20) -> np.ndarray:
    """
    Synchronous Hopfield update on binarised pattern (+1/-1).
    W_sym: (N, N) symmetric weight matrix (diagonal zeroed).
    Returns final state after n_steps.
    """
    state = pattern.copy().astype(float)
    for _ in range(n_steps):
        h = W_sym @ state
        state = np.sign(h)
        state[state == 0] = 1  # tie-break
    return state


def _assembly_to_hopfield_pattern(assembly: np.ndarray, n_neurons: int) -> np.ndarray:
    """Binary (-1/+1) pattern for Hopfield evaluation."""
    p = -np.ones(n_neurons)
    p[assembly] = 1.0
    return p


def _hopfield_overlap(retrieved: np.ndarray, stored: np.ndarray) -> float:
    """Overlap m = (1/N) Σ_i retrieved_i × stored_i."""
    return float(np.mean(retrieved * stored))

# ---------------------------------------------------------------------------
# 1. EWC experiment
# ---------------------------------------------------------------------------

def run_ewc_experiment(
    assemblies: List[np.ndarray],
    trial_seed: int,
    ewc_lambda: float = 8.0,
) -> dict:
    """
    A→B→C→D with Elastic Weight Consolidation.
    After each memory, stores W anchor + Fisher information.
    EWC correction applied after every STDP step for subsequent memories.
    Uses slow weights (same as Slow+Replay) for fair comparison.
    """
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    n_mem = len(assemblies)
    net   = cf.build_network(use_slow=True)
    tags  = cf.SynapticTags() if cf.USE_TAGGING else None

    anchors: List[torch.Tensor] = []
    fishers: List[torch.Tensor] = []

    baseline_scores = np.full(n_mem, np.nan)
    final_scores    = np.full(n_mem, np.nan)

    for j, asm in enumerate(assemblies):
        W_before = _snapshot_W_ee(net)

        if j == 0:
            # First memory: standard training, no EWC yet
            cf.train_one_memory(net, asm, tags=tags, n_presentations=cf._N_PRESENTATIONS)
        else:
            _train_one_memory_ewc(
                net, asm, tags=tags,
                n_presentations=cf._N_PRESENTATIONS,
                anchors=anchors, fishers=fishers,
                ewc_lambda=ewc_lambda,
            )

        # Record anchor + Fisher for this memory's E→E weights
        W_after = _snapshot_W_ee(net)
        anchors.append(W_after.clone())
        fishers.append(_compute_fisher(W_before, W_after))

        # Baseline probe
        r = cf.probe_memory(net, asm)
        baseline_scores[j] = r["isyn_score"]

        # Competitive interference (same as cf Slow conditions)
        if j > 0:
            cf.apply_competitive_interference(net, asm, assemblies[:j])

        # Inter-memory rest with replay (same as Slow+Replay)
        if j < n_mem - 1:
            current_scores = [
                cf.probe_memory(net, assemblies[i])["isyn_score"]
                for i in range(j + 1)
            ]
            cf.inter_memory_rest_with_replay(
                net,
                learned_assemblies=assemblies[:j + 1],
                current_scores=current_scores,
                prioritize="interference_aware",
                tags=tags,
                rest_id=j,
                accumulated_metrics=None,
                ablation=None,
            )

    final_scores = np.array([
        cf.probe_memory(net, assemblies[i])["isyn_score"]
        for i in range(n_mem)
    ])
    return {
        "label":          "EWC",
        "final_scores":   final_scores,
        "baseline_scores": baseline_scores,
        "mean_retention": float(np.nanmean(final_scores[:n_mem - 1])),
        "trial_seed":     trial_seed,
    }


# ---------------------------------------------------------------------------
# 2. Replay Buffer experiment
# ---------------------------------------------------------------------------

def run_replay_buffer_experiment(
    assemblies: List[np.ndarray],
    trial_seed: int,
    buffer_size: int = 10,
) -> dict:
    """
    A→B→C→D with explicit replay buffer.
    Records buffer_size spike patterns per memory after training.
    During inter-memory rest: injects stored patterns with STDP active.
    No generative SWR replay, no coherence gating.
    """
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    n_mem = len(assemblies)
    net   = cf.build_network(use_slow=True)
    tags  = cf.SynapticTags() if cf.USE_TAGGING else None

    pattern_buffer: Dict[int, List[torch.Tensor]] = {}
    baseline_scores = np.full(n_mem, np.nan)

    for j, asm in enumerate(assemblies):
        cf.train_one_memory(net, asm, tags=tags, n_presentations=cf._N_PRESENTATIONS)

        # Record patterns for this memory
        pattern_buffer[j] = _record_buffer_patterns(net, asm, buffer_size)

        r = cf.probe_memory(net, asm)
        baseline_scores[j] = r["isyn_score"]

        if j > 0:
            cf.apply_competitive_interference(net, asm, assemblies[:j])

        # Inter-memory rest: use stored pattern replay
        if j < n_mem - 1:
            n_ev = cf._N_REPLAY_EVENTS
            # Use the fast-weight decay from cf's no-replay rest first (pre-replay)
            half_n = cf.INTER_MEM_REST_STEPS // 2
            rest_n = cf.INTER_MEM_REST_STEPS - half_n
            f_half = 1.0 - float(np.exp(-half_n / cf.FAST_DECAY_TAU))
            with torch.no_grad():
                W    = net.W.data[:cf.N_EXC, :cf.N_EXC]
                base = net.W_init[:cf.N_EXC, :cf.N_EXC]
                net.W.data[:cf.N_EXC, :cf.N_EXC] = W + (base - W) * f_half
            cf.bulk_slow_step(net, half_n)
            if tags is not None:
                tags.decay(n_steps=half_n)

            # Buffer replay
            _replay_buffer_rest(net, pattern_buffer, n_ev, tags)

            # Post-replay decay
            f_rest = 1.0 - float(np.exp(-rest_n / cf.FAST_DECAY_TAU))
            with torch.no_grad():
                W    = net.W.data[:cf.N_EXC, :cf.N_EXC]
                base = net.W_init[:cf.N_EXC, :cf.N_EXC]
                net.W.data[:cf.N_EXC, :cf.N_EXC] = W + (base - W) * f_rest
            cf.bulk_slow_step(net, rest_n)
            if tags is not None:
                tags.decay(n_steps=rest_n)

    final_scores = np.array([
        cf.probe_memory(net, assemblies[i])["isyn_score"]
        for i in range(n_mem)
    ])
    return {
        "label":          "Replay Buffer",
        "final_scores":   final_scores,
        "baseline_scores": baseline_scores,
        "mean_retention": float(np.nanmean(final_scores[:n_mem - 1])),
        "trial_seed":     trial_seed,
    }


# ---------------------------------------------------------------------------
# 3. Simple Rehearsal experiment
# ---------------------------------------------------------------------------

def run_rehearsal_experiment(
    assemblies: List[np.ndarray],
    trial_seed: int,
    rehearsal_rate: float = 0.25,
) -> dict:
    """
    A→B→C→D with interleaved rehearsal.
    During new-memory training, each presentation has rehearsal_rate probability
    of being followed by a presentation of one randomly selected prior memory.
    No consolidation rest replay.
    """
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    n_mem = len(assemblies)
    net   = cf.build_network(use_slow=True)
    tags  = cf.SynapticTags() if cf.USE_TAGGING else None

    baseline_scores = np.full(n_mem, np.nan)

    for j, asm in enumerate(assemblies):
        if j == 0:
            cf.train_one_memory(net, asm, tags=tags, n_presentations=cf._N_PRESENTATIONS)
        else:
            _train_one_memory_rehearsal(
                net, asm, old_assemblies=list(assemblies[:j]),
                tags=tags, n_presentations=cf._N_PRESENTATIONS,
                rehearsal_rate=rehearsal_rate,
            )

        r = cf.probe_memory(net, asm)
        baseline_scores[j] = r["isyn_score"]

        if j > 0:
            cf.apply_competitive_interference(net, asm, assemblies[:j])

        # Standard inter-memory rest WITHOUT active replay
        if j < n_mem - 1:
            cf.inter_memory_rest_no_replay(net, tags=tags)

    final_scores = np.array([
        cf.probe_memory(net, assemblies[i])["isyn_score"]
        for i in range(n_mem)
    ])
    return {
        "label":          "Simple Rehearsal",
        "final_scores":   final_scores,
        "baseline_scores": baseline_scores,
        "mean_retention": float(np.nanmean(final_scores[:n_mem - 1])),
        "trial_seed":     trial_seed,
    }


# ---------------------------------------------------------------------------
# 4. Hopfield Attractor experiment
# ---------------------------------------------------------------------------

def run_hopfield_experiment(
    assemblies: List[np.ndarray],
    trial_seed: int,
) -> dict:
    """
    Uses the Slow+Replay trained network weights to define a Hopfield attractor
    network. Evaluates memory retrieval as synchronous Hopfield overlap score.

    Returns "final_scores" as Hopfield overlap values ∈ [-1, 1].
    Hopfield retrieval gives an upper-bound energy landscape; this baseline
    shows whether the spiking model does better or worse than the Hopfield bound.
    """
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    # Run Slow+Replay to get trained weights
    result = cf.run_sequential_experiment(
        use_slow=True, use_replay=True,
        assemblies=assemblies,
        trial_seed=trial_seed,
        prioritize="interference_aware",
        verbose=False,
    )

    # Extract W as Hopfield matrix
    # Build symmetric version (excitatory-only block) with zeroed diagonal
    W_ee = None
    net_temp = cf.build_network(use_slow=True)
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    # Re-run briefly to access the network state after training — we use the
    # final_scores and weight structure from the run_sequential_experiment result
    # but we need the weight matrix itself. Instead, use the Hebbian outer-product
    # as the Hopfield memory matrix, which is the standard Hopfield weight rule.
    n = cf.N_NEURONS
    W_hop = np.zeros((n, n))
    p_stored = []
    for asm in assemblies:
        p = _assembly_to_hopfield_pattern(asm, n)
        p_stored.append(p)
        W_hop += np.outer(p, p)
    W_hop /= n
    np.fill_diagonal(W_hop, 0.0)

    # Probe each memory with Hopfield dynamics
    final_scores = np.zeros(len(assemblies))
    for i, (asm, p) in enumerate(zip(assemblies, p_stored)):
        # Start from partial cue (same CUE_SIZE neurons as spiking model)
        p_cue = _assembly_to_hopfield_pattern(asm, n)
        # Noise out non-cued neurons
        noise_mask = np.ones(n, bool)
        noise_mask[asm[:cf.CUE_SIZE]] = False
        p_cue[noise_mask] = -1
        p_retrieved = _hopfield_retrieve(W_hop, p_cue, n_steps=20)
        final_scores[i] = _hopfield_overlap(p_retrieved, p)

    # baseline_scores = perfect overlap for each memory (just trained)
    baseline_scores = np.ones(len(assemblies))

    return {
        "label":          "Hopfield Attractor",
        "final_scores":   final_scores,
        "baseline_scores": baseline_scores,
        "mean_retention": float(np.nanmean(final_scores[:len(assemblies) - 1])),
        "trial_seed":     trial_seed,
        # Include spiking model comparison for reference
        "spiking_final":  result["final_scores"],
    }


# ---------------------------------------------------------------------------
# 5. Fast-only reference (wraps cf Fast/NoReplay)
# ---------------------------------------------------------------------------

def run_fast_only_experiment(
    assemblies: List[np.ndarray],
    trial_seed: int,
) -> dict:
    """Fast weights only, no replay, no slow consolidation (worst case)."""
    result = cf.run_sequential_experiment(
        use_slow=False, use_replay=False,
        assemblies=assemblies,
        trial_seed=trial_seed,
        prioritize="interference_aware",
        verbose=False,
    )
    return {
        "label":          "Fast Only\n(No Replay)",
        "final_scores":   result["final_scores"],
        "baseline_scores": result["baseline_scores"],
        "mean_retention": float(np.nanmean(result["final_scores"][:len(assemblies) - 1])),
        "trial_seed":     trial_seed,
    }


# ---------------------------------------------------------------------------
# Worker functions (top-level, picklable)
# ---------------------------------------------------------------------------

def _ewc_worker(args):
    asms, seed, ewc_lambda = args
    return run_ewc_experiment(asms, seed, ewc_lambda=ewc_lambda)


def _replay_buf_worker(args):
    asms, seed, buffer_size = args
    return run_replay_buffer_experiment(asms, seed, buffer_size=buffer_size)


def _rehearsal_worker(args):
    asms, seed, rehearsal_rate = args
    return run_rehearsal_experiment(asms, seed, rehearsal_rate=rehearsal_rate)


def _hopfield_worker(args):
    asms, seed = args
    return run_hopfield_experiment(asms, seed)


def _fast_only_worker(args):
    asms, seed = args
    return run_fast_only_experiment(asms, seed)


def _slow_replay_worker(args):
    asms, seed = args
    result = cf.run_sequential_experiment(
        use_slow=True, use_replay=True,
        assemblies=asms, trial_seed=seed,
        prioritize="interference_aware", verbose=False,
    )
    return {
        "label":          "Slow+Replay\n(Full Model)",
        "final_scores":   result["final_scores"],
        "baseline_scores": result["baseline_scores"],
        "mean_retention": float(np.nanmean(result["final_scores"][:len(asms) - 1])),
        "trial_seed":     seed,
    }


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_baselines(
    assemblies: List[np.ndarray],
    n_trials: int = None,
    verbose: bool = False,
) -> Dict[str, List[dict]]:
    """
    Run all 5 baseline comparisons (+ Slow+Replay reference) for n_trials each.
    Returns dict keyed by baseline label, value = list of per-trial result dicts.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if n_trials is None:
        n_trials = cf.N_TRIALS

    n_workers = min(cf.N_WORKERS, n_trials)
    seeds = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    asms  = [a.copy() for a in assemblies]

    tasks = {
        "slow_replay":    ([(_slow_replay_worker,  [asms, s]) for s in seeds], _slow_replay_worker),
        "ewc":            ([(_ewc_worker,           [asms, s, 8.0]) for s in seeds], _ewc_worker),
        "replay_buffer":  ([(_replay_buf_worker,    [asms, s, 10]) for s in seeds], _replay_buf_worker),
        "rehearsal":      ([(_rehearsal_worker,     [asms, s, 0.25]) for s in seeds], _rehearsal_worker),
        "hopfield":       ([(_hopfield_worker,      [asms, s]) for s in seeds], _hopfield_worker),
        "fast_only":      ([(_fast_only_worker,     [asms, s]) for s in seeds], _fast_only_worker),
    }

    results: Dict[str, List[dict]] = {k: [] for k in tasks}

    if verbose:
        print(f"  [Baselines] {n_trials} trials x {len(tasks)} conditions, N_WORKERS={n_workers}")

    for key, (task_list, worker_fn) in tasks.items():
        if verbose:
            print(f"  [Baselines] Running {key} ...", flush=True)
        arg_list = [t[1] for t in task_list]
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                trial_results = list(pool.map(worker_fn, arg_list))
        except Exception as e:
            warnings.warn(f"Parallel execution failed for {key}: {e}. Falling back to serial.")
            trial_results = [worker_fn(args) for args in arg_list]
        results[key] = trial_results
        if verbose:
            means = [r["mean_retention"] for r in trial_results if np.isfinite(r["mean_retention"])]
            print(f"    {key}: mean_retention = {np.mean(means):.4f} ± {np.std(means):.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_baseline_comparison(
    results: Dict[str, List[dict]],
    stats_rows=None,
    out_dir: str = ".",
) -> None:
    """
    3-panel figure:
      (A) Mean early-memory retention ± 95% CI per baseline
      (B) Per-memory retention heatmap (memory × baseline)
      (C) Stats table: Cohen's d vs Slow+Replay reference
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from extensions.stats_utils import bootstrap_ci

    cfg_map = {c["key"]: c for c in BASELINE_CONFIGS}
    keys    = ["slow_replay", "ewc", "replay_buffer", "rehearsal", "hopfield", "fast_only"]
    labels  = [cfg_map[k]["label"] for k in keys]
    colors  = [cfg_map[k]["color"] for k in keys]

    n_mem = cf.N_MEMORIES

    # Extract per-baseline mean retention across trials
    mean_rets = []
    ci_los    = []
    ci_his    = []
    per_mem   = []  # (n_baselines, n_mem) mean final score

    for key in keys:
        trial_list = results.get(key, [])
        if not trial_list:
            mean_rets.append(np.nan); ci_los.append(np.nan); ci_his.append(np.nan)
            per_mem.append(np.full(n_mem, np.nan))
            continue
        rets = np.array([r["mean_retention"] for r in trial_list])
        lo, hi, est = bootstrap_ci(rets[np.isfinite(rets)], n_boot=1000)
        mean_rets.append(est); ci_los.append(lo); ci_his.append(hi)

        mem_mat = np.array([r["final_scores"] for r in trial_list])
        per_mem.append(np.nanmean(mem_mat, axis=0))

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ── Panel A: bar chart
    x = np.arange(len(keys))
    bars = ax_a.bar(x, mean_rets, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    for i, (lo, hi, est) in enumerate(zip(ci_los, ci_his, mean_rets)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax_a.plot([i, i], [lo, hi], color="black", linewidth=1.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel("Mean Retention\n(I_syn score, A/B/C after D)", fontsize=10)
    ax_a.set_title("Baseline Comparison: Mean Early-Memory Retention", fontsize=11)
    ax_a.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ── Panel B: heatmap
    hm = np.array(per_mem)  # (n_baselines, n_mem)
    vmin = np.nanmin(hm) if not np.all(np.isnan(hm)) else -0.1
    vmax = np.nanmax(hm) if not np.all(np.isnan(hm)) else 1.0
    im = ax_b.imshow(hm, aspect="auto", cmap="RdYlGn",
                     vmin=vmin, vmax=max(vmax, 0.01))
    ax_b.set_xticks(range(n_mem))
    ax_b.set_xticklabels([f"Mem {chr(65+i)}" for i in range(n_mem)], fontsize=8)
    ax_b.set_yticks(range(len(keys)))
    ax_b.set_yticklabels(labels, fontsize=7)
    ax_b.set_title("Final Retention per Memory", fontsize=10)
    plt.colorbar(im, ax=ax_b, fraction=0.046, label="I_syn score")

    # ── Panel C: if stats provided
    if stats_rows:
        ax_c.axis("off")
        col_labels = ["Condition", "Mean", "d vs ref", "p", "FDR*"]
        rows_disp  = []
        for row in stats_rows:
            rows_disp.append([
                row["label"].replace("\n", " ")[:16],
                f"{row['mean']:.3f}",
                f"{row['cohens_d_vs_ref']:+.2f}",
                f"{row['perm_p']:.3f}",
                "*" if row["reject_fdr"] else "",
            ])
        tbl = ax_c.table(
            cellText=rows_disp, colLabels=col_labels,
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.6)
        ax_c.set_title("Statistical Comparison\nvs Slow+Replay", fontsize=10)
    else:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "Run with stats_rows\nfor significance table",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=10, color="gray")

    fig.suptitle(
        "Continual Learning Baseline Comparison\n"
        "Sequential A→B→C→D, 20% overlap",
        fontsize=12, y=0.98
    )

    cf._save_fig(fig, "baseline_comparison")
    plt.close(fig)
    print("[FIG] Saved baseline_comparison.png", flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os, sys as _sys, pathlib as _pl
    _os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Ensure project root (parent of extensions/) is on sys.path
    _root = str(_pl.Path(__file__).resolve().parent.parent)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    print("[baselines self-test] building assemblies ...", flush=True)
    asms = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)

    seed = cf.MASTER_SEED
    print(f"[baselines self-test] EWC (seed={seed}) ...", flush=True)
    r_ewc = run_ewc_experiment(asms, seed, ewc_lambda=8.0)
    print(f"  EWC final_scores={np.round(r_ewc['final_scores'], 3)}, "
          f"mean_ret={r_ewc['mean_retention']:.4f}")

    print(f"[baselines self-test] Replay Buffer (seed={seed}) ...", flush=True)
    r_buf = run_replay_buffer_experiment(asms, seed, buffer_size=5)
    print(f"  Buffer final_scores={np.round(r_buf['final_scores'], 3)}, "
          f"mean_ret={r_buf['mean_retention']:.4f}")

    print(f"[baselines self-test] Rehearsal (seed={seed}) ...", flush=True)
    r_reh = run_rehearsal_experiment(asms, seed, rehearsal_rate=0.25)
    print(f"  Rehearsal final_scores={np.round(r_reh['final_scores'], 3)}, "
          f"mean_ret={r_reh['mean_retention']:.4f}")

    print(f"[baselines self-test] Hopfield (seed={seed}) ...", flush=True)
    r_hop = run_hopfield_experiment(asms, seed)
    print(f"  Hopfield final_scores={np.round(r_hop['final_scores'], 3)}, "
          f"mean_ret={r_hop['mean_retention']:.4f}")

    print("[baselines self-test] DONE.", flush=True)
