"""
experiments — experiment orchestration and hyperparameter sweep utilities.

Planned modules:
    experiment_runner — Assembles network, task, and agent; executes the training loop
    sweep             — Grid and random search over config parameters
    analysis          — Post-hoc analysis of completed runs (load logs, compare metrics)
"""

from .experiment_runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
