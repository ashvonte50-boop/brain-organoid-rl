"""
tasks — cognitive and memory task environments.

Each task exposes a gymnasium-compatible interface so RL agents can interact
with it via step() / reset() / render().

Planned modules:
    memory_task      — Working memory: pattern presentation and recall probe
    pattern_completion — Associative recall under partial cues (Hopfield-like)
    sequence_learning  — Temporal sequence encoding and replay tasks
    stimulation_task   — Closed-loop adaptive stimulation environment
"""

from .memory_task import WorkingMemoryTask

__all__ = ["WorkingMemoryTask"]
