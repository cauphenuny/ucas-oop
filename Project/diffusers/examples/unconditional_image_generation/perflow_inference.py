"""PeRFlow inference and speed benchmark for unconditional generation.

This script loads a baseline DDPMPipeline and a PeRFlow-augmented version
(using delta weights) and measures wall-clock speed for both.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import DDPMPipeline, PeRFlowScheduler
from diffusers.schedulers.utils_perflow import load_delta_weights_into_unet
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PeRFlow accelerated inference for unconditional generation.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path or repo id of the baseline DDPMPipeline.")
    parser.add_argument(
        "--delta_weights",
        type=str,
        required=True,
        help="Path to delta_weights.safetensors or a directory containing it (from finetuning).",
    )
    parser.add_argument("--num_images", type=int, default=4, help="Batch size / number of images to generate.")
    parser.add_argument("--num_steps_base", type=int, default=50, help="Number of inference steps for the baseline scheduler.")
    parser.add_argument("--num_steps_perflow", type=int, default=8, help="Number of inference steps for the PeRFlow scheduler.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cuda, cpu). Defaults to pipeline default.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory to save generated images for visual check.")
    return parser.parse_args()


def _maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_pipeline(pipe: DDPMPipeline, num_images: int, num_steps: int, seed: int) -> tuple[list[Image.Image], float]:
    # Use CPU for inference to avoid MPS device issues on macOS
    original_device = pipe.device
    # Recreate scheduler on CPU to keep tensors aligned
    pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    inference_device = "cpu"
    if torch.cuda.is_available() and str(pipe.device) != "mps":
        inference_device = pipe.device

    pipe = pipe.to(inference_device)
    generator = torch.Generator(device=inference_device).manual_seed(seed)
    _maybe_sync()
    start = time.perf_counter()
    result = pipe(
        generator=generator,
        batch_size=num_images,
        num_inference_steps=num_steps,
        output_type="pil",
    )
    _maybe_sync()
    elapsed = time.perf_counter() - start
    
    # Move back to original device
    pipe = pipe.to(original_device)
    return result.images, elapsed


def main():
    args = parse_args()

    device = args.device
    base_pipe = DDPMPipeline.from_pretrained(args.base_model_path)
    if device is not None:
        base_pipe = base_pipe.to(device)

    perflow_pipe = DDPMPipeline.from_pretrained(args.base_model_path)
    load_delta_weights_into_unet(perflow_pipe, model_path=args.delta_weights, base_path=args.base_model_path)

    # Build a PeRFlow scheduler; map common prediction types to supported ones
    perflow_cfg = dict(perflow_pipe.scheduler.config)
    pred_type = perflow_cfg.get("prediction_type")
    if pred_type in ("epsilon", "sample"):
        perflow_cfg["prediction_type"] = "ddim_eps"
    perflow_pipe.scheduler = PeRFlowScheduler.from_config(perflow_cfg)
    if device is not None:
        perflow_pipe = perflow_pipe.to(device)

    base_images, base_time = _run_pipeline(base_pipe, args.num_images, args.num_steps_base, args.seed)
    perflow_images, perflow_time = _run_pipeline(perflow_pipe, args.num_images, args.num_steps_perflow, args.seed)

    print(f"Baseline steps: {args.num_steps_base}, time: {base_time:.3f}s for {args.num_images} images")
    print(f"PeRFlow  steps: {args.num_steps_perflow}, time: {perflow_time:.3f}s for {args.num_images} images")
    if perflow_time > 0:
        print(f"Speedup: {base_time / perflow_time:.2f}x")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(base_images):
            img.save(out_dir / f"baseline_{idx}.png")
        for idx, img in enumerate(perflow_images):
            img.save(out_dir / f"perflow_{idx}.png")
        print(f"Saved images to {out_dir}")


if __name__ == "__main__":
    main()
