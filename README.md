# Brain Organoid RL — Adaptive Stimulation in Cortical Spiking Networks

A computational neuroscience research framework for studying adaptive stimulation strategies
and memory retention in large-scale spiking neural networks (SNNs) inspired by cortical organoids.

Built with [PyTorch](https://pytorch.org/) and [Norse](https://norse.github.io/norse/) for
biologically plausible gradient-based learning in spiking networks.

---

## Research Goals

- Model cortical organoid dynamics using biologically constrained LIF/AdEx neuron ensembles
- Study emergent memory retention under spike-timing-dependent plasticity (STDP)
- Develop reinforcement learning agents that deliver adaptive closed-loop stimulation
- Investigate how network topology and initial connectivity shape long-term memory consolidation

---

## Project Structure

```
brain-organoid-rl/
├── main.py                  # Experiment entry point
├── configs/                 # YAML configuration files
├── neuron_models/           # LIF, AdEx, Izhikevich neuron definitions
├── synapses/                # Synapse models and connectivity patterns
├── plasticity/              # STDP, homeostatic, and neuromodulatory rules
├── tasks/                   # Cognitive/memory task environments
├── rl/                      # Reinforcement learning agents and policies
├── experiments/             # Experiment runners and sweep utilities
├── visualization/           # Spike rasters, weight plots, connectivity graphs
├── utils/                   # Metrics, seeding, checkpointing helpers
├── notebooks/               # Exploratory analysis notebooks
├── logs/                    # TensorBoard / WandB run logs
└── checkpoints/             # Saved model states
```

---

## Quick Start

```bash
# 1. Create and activate environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a baseline experiment
python main.py --config configs/default_config.yaml

# 4. Launch TensorBoard
tensorboard --logdir logs/
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Tensor computation and autograd |
| `norse` | Spiking neuron primitives (LIF, LIFEx, etc.) |
| `numpy` / `scipy` | Numerical analysis |
| `gymnasium` | Task / environment interface |
| `tensorboard` | Experiment monitoring |
| `wandb` | Experiment tracking and hyperparameter sweeps |

---

## Citation

If you build on this framework in your research, please cite relevant Norse and PyTorch papers
alongside your own work. See `notebooks/references.md` for a curated reading list.
