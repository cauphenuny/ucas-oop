import sys
from pathlib import Path

import torch

from diffusers import PeRFlowScheduler

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "perflow_fixture.pt"
REPO_ROOT = Path(__file__).resolve().parents[3]
PERFLOW_SRC = REPO_ROOT / "PeRFlow" / "src"


def _load_original_scheduler():
    if str(PERFLOW_SRC) not in sys.path:
        sys.path.insert(0, str(PERFLOW_SRC))
    from scheduler_perflow import PeRFlowScheduler as OriginalPeRFlowScheduler

    return OriginalPeRFlowScheduler


def _instantiate_schedulers(config: dict):
    diffusers_scheduler = PeRFlowScheduler(**config)
    original_scheduler_cls = _load_original_scheduler()
    original_scheduler = original_scheduler_cls(**config)
    return diffusers_scheduler, original_scheduler


def test_perflow_step_parity():
    fixture = torch.load(FIXTURE_PATH, map_location="cpu")
    config = fixture["config"]
    timesteps = fixture["timesteps"].to(dtype=torch.int64)
    samples = fixture["samples"].float()
    model_outputs = fixture["model_outputs"].float()

    diffusers_scheduler, original_scheduler = _instantiate_schedulers(config)

    num_inference_steps = int(fixture["num_inference_steps"].item())
    diffusers_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")
    original_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")

    # Overwrite timesteps to ensure an identical grid for both implementations
    diffusers_scheduler.timesteps = timesteps
    original_scheduler.timesteps = timesteps

    for idx, timestep in enumerate(timesteps):
        sample = samples[idx]
        model_output = model_outputs[idx]

        diff_out = diffusers_scheduler.step(model_output, int(timestep.item()), sample, return_dict=False)[0]
        orig_out = original_scheduler.step(model_output, int(timestep.item()), sample, return_dict=False)[0]

        torch.testing.assert_close(diff_out, orig_out, rtol=1e-6, atol=1e-7)


def test_perflow_add_noise_parity():
    fixture = torch.load(FIXTURE_PATH, map_location="cpu")
    config = fixture["config"]

    diffusers_scheduler, original_scheduler = _instantiate_schedulers(config)

    original_samples = fixture["add_noise_samples"].float()
    noise = fixture["add_noise_noise"].float()
    timesteps = fixture["add_noise_timesteps"].to(dtype=torch.int64)

    diff_noisy = diffusers_scheduler.add_noise(original_samples, noise, timesteps)
    orig_noisy = original_scheduler.add_noise(original_samples, noise, timesteps)

    torch.testing.assert_close(diff_noisy, orig_noisy, rtol=1e-6, atol=1e-7)
