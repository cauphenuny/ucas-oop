# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WanFunControl Pipeline for camera-controlled video generation.

This pipeline implements the Wan Fun-Control model that accepts camera parameters
to control video generation through Plücker ray embeddings.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, WanFunControlTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import WanPipelineOutput


logger = logging.get_logger(__name__)


class WanFunControlPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for camera-controlled video generation using Wan Fun-Control.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`UMT5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanFunControlTransformer3DModel`]):
            Conditional Transformer to denoise the input latents with camera control support.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanFunControlTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def encode_control_camera_latents(
        self,
        camera_params: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """
        Encode camera parameters to control latents.

        Args:
            camera_params: Camera parameters tensor
            height: Video height
            width: Video width
            num_frames: Number of frames
            device: Device to place tensors on
            dtype: Data type for tensors

        Returns:
            Control camera latents of shape [batch, control_channels, frames, height, width]
        """
        if camera_params is None:
            return None

        # TODO: Implement proper Plücker embedding generation and encoding
        # For now, return a placeholder
        device = device or self._execution_device
        dtype = dtype or self.vae.dtype

        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        latent_frames = num_frames // self.vae_scale_factor_temporal

        # Placeholder: return zeros with the expected shape
        # Real implementation should generate Plücker embeddings and encode them
        control_latents = torch.zeros(
            (1, 36, latent_frames, latent_height, latent_width),
            device=device,
            dtype=dtype,
        )

        return control_latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        camera_params: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[WanPipelineOutput, tuple]:
        """
        Generate camera-controlled video.

        Args:
            prompt: Text prompt for generation
            camera_params: Optional camera parameters for control
            height: Video height
            width: Video width
            num_frames: Number of frames
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative text prompt
            num_videos_per_prompt: Number of videos per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated latents
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            output_type: Output format ("pil" or "pt")
            return_dict: Whether to return dict
            callback: Callback function for progress
            callback_steps: Frequency of callback calls

        Returns:
            Generated video frames
        """
        # For now, return a simple placeholder
        # Real implementation would follow the pattern from WanPipeline
        # with additional control latents processing

        raise NotImplementedError(
            "WanFunControlPipeline.__call__ not fully implemented yet. "
            "Use this pipeline for feature extraction and testing only."
        )
