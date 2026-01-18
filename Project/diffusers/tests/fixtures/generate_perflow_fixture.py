"""Generate deterministic fixtures for PeRFlow parity tests.

This script produces a torch .pt file containing timesteps, samples, model outputs,
and add-noise inputs so that parity tests can run deterministically against the
reference PeRFlow implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

# Paths
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent  # .../diffusers/tests -> .../diffusers
FIXTURE_PATH = THIS_DIR / "perflow_fixture.pt"


def build_config() -> Dict[str, object]:
    """Return a baseline scheduler config used for fixture generation."""
    return {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "trained_betas": None,
        "set_alpha_to_one": False,
        "prediction_type": "ddim_eps",
        "t_noise": 1.0,
        "t_clean": 0.0,
        "num_time_windows": 4,
    }


def generate_fixtures(
    num_inference_steps: int = 8,
    sample_shape: Tuple[int, ...] = (2, 4, 8, 8),
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    from diffusers import PeRFlowScheduler

    torch.manual_seed(seed)

    config = build_config()
    scheduler = PeRFlowScheduler(**config)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")

    timesteps = scheduler.timesteps.clone().cpu()
    num_steps = len(timesteps)

    samples = torch.randn((num_steps,) + sample_shape, dtype=torch.float32)
    model_outputs = torch.randn((num_steps,) + sample_shape, dtype=torch.float32)

    add_noise_samples = torch.randn(sample_shape, dtype=torch.float32)
    add_noise_noise = torch.randn(sample_shape, dtype=torch.float32)
    add_noise_timesteps = torch.tensor([1, 250], dtype=torch.int64)

    return {
        "config": config,
        "timesteps": timesteps,
        "samples": samples,
        "model_outputs": model_outputs,
        "add_noise_samples": add_noise_samples,
        "add_noise_noise": add_noise_noise,
        "add_noise_timesteps": add_noise_timesteps,
        "seed": torch.tensor(seed, dtype=torch.int64),
        "num_inference_steps": torch.tensor(num_inference_steps, dtype=torch.int64),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PeRFlow parity fixtures.")
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH, help="Where to write the fixture .pt file.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Number of inference steps for timesteps.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used for generating deterministic random tensors.")
    args = parser.parse_args()

    fixture = generate_fixtures(num_inference_steps=args.num_inference_steps, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(f"Wrote PeRFlow fixtures to {args.output}")


if __name__ == "__main__":
    main()
