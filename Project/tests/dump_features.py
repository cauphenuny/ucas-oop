"""Utility script for dumping WanVideo pipeline features for unit tests.

This script runs the DiffSynth-Studio WanVideo pipeline (Control-Camera
variant) on the local fixture assets and records intermediate tensors such as
VAE embeddings, patchified inputs, and periodic transformer block activations.
The exported tensors can be used as immutable fixtures inside unit tests.
"""

from __future__ import annotations

import argparse
import json
import types
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import torch
from einops import rearrange
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


DEFAULT_MODEL_ID = "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera"
DEFAULT_TOKENIZER_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CAMERA_ORIGIN = (
    0,
    0.532139961,
    0.946026558,
    0.5,
    0.5,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump WanVideo features for fixtures")
    parser.add_argument("--control-video", type=Path, default=Path("tests/fixtures/control.mp4"), help="Path to the control video (mp4)")
    parser.add_argument("--input-image", type=Path, default=Path("tests/fixtures/doge.jpg"), help="Path to the conditioning image")
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures"), help="Where to write the feature artifacts")
    parser.add_argument("--output-stem", default="wan21_fun_v11_control_camera", help="Filename stem for the exported artifacts")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="DiffSynth model identifier to download")
    parser.add_argument("--device", default="cuda", help="Torch device for inference")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"), help="Computation dtype for the pipeline")
    parser.add_argument("--prompt", default="Unit test prompt", help="Positive text prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative text prompt")
    parser.add_argument("--seed", type=int, default=0, help="Noise seed for reproducibility")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--cfg-merge", action="store_true", help="Enable cfg merge pathway")
    parser.add_argument("--num-steps", type=int, default=4, help="Number of scheduler steps (only the first is logged)")
    parser.add_argument("--denoising-strength", type=float, default=1.0, help="Strength passed to the scheduler")
    parser.add_argument("--sigma-shift", type=float, default=5.0, help="Scheduler sigma shift")
    parser.add_argument("--num-frames", type=int, default=None, help="Optional cap on the number of control frames to use")
    parser.add_argument("--camera-direction", default="Left", help="Camera control direction to activate the adapter")
    parser.add_argument("--camera-speed", type=float, default=0.01, help="Camera control speed parameter")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52), metavar=("H", "W"), help="Tile size for tiled VAE encode/decode")
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26), metavar=("H", "W"), help="Tile stride for tiled VAE encode/decode")
    parser.add_argument("--tiled", dest="tiled", action="store_true", help="Enable tiled VAE encode/decode (default)")
    parser.add_argument("--no-tiled", dest="tiled", action="store_false", help="Disable tiled VAE encode/decode")
    parser.add_argument("--block-interval", type=int, default=6, help="Interval (in blocks) for sampling transformer activations")
    parser.add_argument("--rand-device", default="cpu", help="Device used when sampling initial noise")
    parser.add_argument("--audio-sample-rate", type=int, default=16000, help="Sample rate placeholder for the pipeline")
    parser.set_defaults(tiled=True)
    return parser.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def load_video_frames(path: Path, limit: Optional[int]) -> List[Image.Image]:
    reader = imageio.get_reader(path)
    frames: List[Image.Image] = []
    try:
        for idx, frame in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            frames.append(Image.fromarray(frame).convert("RGB"))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def detach_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.float16).cpu().contiguous()


def tensor_metadata(tensor: torch.Tensor) -> Dict[str, object]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def build_model_configs(model_id: str) -> List[ModelConfig]:
    return [
        ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]


def prepare_inputs(
    prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    control_video: List[Image.Image],
    height: int,
    width: int,
    num_frames: int,
    args: argparse.Namespace,
) -> Tuple[dict, dict, dict]:
    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": args.num_steps,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": args.num_steps,
    }
    inputs_shared = {
        "input_image": input_image,
        "end_image": None,
        "input_video": None,
        "denoising_strength": args.denoising_strength,
        "control_video": control_video,
        "reference_image": None,
        "camera_control_direction": args.camera_direction,
        "camera_control_speed": args.camera_speed,
        "camera_control_origin": DEFAULT_CAMERA_ORIGIN,
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": args.seed,
        "rand_device": args.rand_device,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "cfg_scale": args.cfg_scale,
        "cfg_merge": args.cfg_merge,
        "sigma_shift": args.sigma_shift,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": args.tiled,
        "tile_size": tuple(args.tile_size),
        "tile_stride": tuple(args.tile_stride),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": args.audio_sample_rate,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
    }
    return inputs_shared, inputs_posi, inputs_nega


def capture_units(pipe: WanVideoPipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> Tuple[dict, dict, dict, Dict[str, torch.Tensor]]:
    captured: Dict[str, torch.Tensor] = {}
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
        name = unit.__class__.__name__
        if name == "WanVideoUnit_ImageEmbedderVAE" and inputs_shared.get("y") is not None and "image_vae_embedding" not in captured:
            captured["image_vae_embedding"] = detach_cpu(inputs_shared["y"])
        if name == "WanVideoUnit_FunControl" and inputs_shared.get("y") is not None:
            captured["control_condition_embedding"] = detach_cpu(inputs_shared["y"])
    return inputs_shared, inputs_posi, inputs_nega, captured


class BlockRecorder:
    def __init__(self, dit, interval: int, store: dict):
        self.dit = dit
        self.interval = max(1, interval)
        self.store = store
        self._orig_patchify = None
        self._orig_blocks: List[Tuple[torch.nn.Module, types.MethodType]] = []

    def __enter__(self):
        self._orig_patchify = self.dit.patchify

        def patched_patchify(module_self, x, control_camera_latents_input=None):
            out = self._orig_patchify(x, control_camera_latents_input)
            self.store["patchify_tensor"] = detach_cpu(out)
            return out

        self.dit.patchify = types.MethodType(patched_patchify, self.dit)

        for idx, block in enumerate(self.dit.blocks):
            original_forward = block.forward

            def make_wrapper(orig_fn, block_index):
                def wrapper(module_self, *args, **kwargs):
                    out = orig_fn(*args, **kwargs)
                    if block_index % self.interval == 0:
                        tensor = out[0] if isinstance(out, tuple) else out
                        self.store.setdefault("blocks", OrderedDict())[f"block_{block_index:03d}"] = detach_cpu(tensor)
                    return out

                return wrapper

            wrapper = make_wrapper(original_forward, idx)
            block.forward = types.MethodType(wrapper, block)
            self._orig_blocks.append((block, original_forward))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_patchify is not None:
            self.dit.patchify = self._orig_patchify
        for block, original in self._orig_blocks:
            block.forward = original


def export_artifacts(output_dir: Path, stem: str, tensors: Dict[str, torch.Tensor], block_tensors: OrderedDict, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "image_vae_embedding": tensors.get("image_vae_embedding"),
        "control_condition_embedding": tensors.get("control_condition_embedding"),
        "patchify_tensor": tensors.get("patchify_tensor"),
        "patch_sequence": tensors.get("patch_sequence"),
        "blocks": block_tensors,
    }
    tensor_path = output_dir / f"{stem}.pt"
    torch.save(bundle, tensor_path)
    meta_path = output_dir / f"{stem}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))


def main():
    args = parse_args()
    dtype = str_to_dtype(args.dtype)
    control_frames = load_video_frames(args.control_video, args.num_frames)
    input_image = Image.open(args.input_image).convert("RGB")
    height = input_image.height
    width = input_image.width
    available_frames = len(control_frames)
    requested_frames = args.num_frames or available_frames
    num_frames = min(requested_frames, available_frames)
    control_frames = control_frames[:num_frames]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=args.device,
        model_configs=build_model_configs(args.model_id),
        tokenizer_config=ModelConfig(model_id=DEFAULT_TOKENIZER_MODEL, origin_file_pattern="google/umt5-xxl/"),
    )

    inputs_shared, inputs_posi, inputs_nega = prepare_inputs(
        args.prompt,
        args.negative_prompt,
        input_image,
        control_frames,
        height,
        width,
        num_frames,
        args,
    )

    inputs_shared, inputs_posi, inputs_nega, captured = capture_units(pipe, inputs_shared, inputs_posi, inputs_nega)

    pipe.scheduler.set_timesteps(args.num_steps, denoising_strength=args.denoising_strength, shift=args.sigma_shift)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    timestep = pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

    feature_store: Dict[str, torch.Tensor] = {}
    blocks: OrderedDict = OrderedDict()
    feature_store["blocks"] = blocks

    with torch.inference_mode():
        with BlockRecorder(pipe.dit, args.block_interval, feature_store):
            pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)

    pipe.load_models_to_device([])

    if "patchify_tensor" in feature_store:
        sequence = rearrange(feature_store["patchify_tensor"], "b c f h w -> b (f h w) c").contiguous()
        feature_store["patch_sequence"] = detach_cpu(sequence)

    captured.update({k: v for k, v in feature_store.items() if k != "blocks"})

    metadata = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "block_interval": args.block_interval,
        "tensors": {k: tensor_metadata(v) for k, v in captured.items() if isinstance(v, torch.Tensor)},
        "blocks": {name: tensor_metadata(tensor) for name, tensor in feature_store["blocks"].items()},
    }

    export_artifacts(args.output_dir, args.output_stem, captured, feature_store["blocks"], metadata)


if __name__ == "__main__":
    main()
