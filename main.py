"""
Entry point for brain-organoid-rl experiments.

Parses a YAML config, constructs the network and task environment,
then hands off to the appropriate experiment runner.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from experiments.experiment_runner import ExperimentRunner
from utils.seed import set_global_seed
from utils.logging_setup import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain Organoid RL — adaptive stimulation in cortical SNNs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML experiment configuration file",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override random seed from config"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Override compute device from config",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Override run name for logging"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(config: dict, override: str | None) -> torch.device:
    device_str = override or config.get("training", {}).get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable — falling back to CPU.")
        device_str = "cpu"
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    configure_logging(debug=args.debug)
    log = logging.getLogger(__name__)

    config = load_config(args.config)

    # Allow CLI overrides
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    if args.run_name is not None:
        config.setdefault("logging", {})["run_name"] = args.run_name

    seed = config.get("training", {}).get("seed", 42)
    set_global_seed(seed)

    device = resolve_device(config, args.device)
    log.info("Device: %s | Seed: %d", device, seed)
    log.info("Starting experiment: %s", config.get("experiment", {}).get("name", "unnamed"))

    runner = ExperimentRunner(config=config, device=device)
    runner.run()


if __name__ == "__main__":
    main()
