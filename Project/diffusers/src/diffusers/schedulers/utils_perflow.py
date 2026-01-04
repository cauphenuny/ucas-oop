# Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved.
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
Utility functions for PeRFlow implementation.
"""

import os
from collections import OrderedDict
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def merge_delta_weights_into_unet(pipe, delta_weights: OrderedDict) -> object:
    """
    Merge delta weights into a UNet model.
    
    This function takes a pipeline and delta weights, and merges the delta weights
    into the UNet component of the pipeline.
    
    Args:
        pipe: The diffusion pipeline containing the UNet model.
        delta_weights (`OrderedDict`):
            The delta weights to merge into the UNet.
            
    Returns:
        The pipeline with updated UNet weights.
    """
    unet_weights = pipe.unet.state_dict()
    assert unet_weights.keys() == delta_weights.keys()
    
    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        unet_weights[key] = unet_weights[key].to(dtype)
    
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe


def load_delta_weights_into_unet(
    pipe,
    model_path: str = "hsyan/piecewise-rectified-flow-v0-1",
    base_path: str = "runwayml/stable-diffusion-v1-5",
) -> object:
    """
    Load delta weights from a model path and merge them into the UNet.
    
    This function loads delta weights (or computes them from merged weights) and
    applies them to the UNet in the pipeline.
    
    Args:
        pipe: The diffusion pipeline to update.
        model_path (`str`, defaults to "hsyan/piecewise-rectified-flow-v0-1"):
            Path to the model containing delta weights or merged weights.
        base_path (`str`, defaults to "runwayml/stable-diffusion-v1-5"):
            Path to the base model for computing delta weights if needed.
            
    Returns:
        The pipeline with updated UNet weights.
    """
    # Load delta_weights
    if os.path.exists(os.path.join(model_path, "delta_weights.safetensors")):
        print("### delta_weights exists, loading...")
        delta_weights = OrderedDict()
        with safe_open(os.path.join(model_path, "delta_weights.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                delta_weights[key] = f.get_tensor(key)
                
    elif os.path.exists(os.path.join(model_path, "diffusion_pytorch_model.safetensors")):
        print("### merged_weights exists, loading...")
        merged_weights = OrderedDict()
        with safe_open(os.path.join(model_path, "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_weights[key] = f.get_tensor(key)
        
        # Import here to avoid circular dependency
        from ..pipelines.stable_diffusion import StableDiffusionPipeline
        base_weights = StableDiffusionPipeline.from_pretrained(
            base_path, torch_dtype=torch.float16, safety_checker=None).unet.state_dict()
        assert base_weights.keys() == merged_weights.keys()
        
        delta_weights = OrderedDict()
        for key in merged_weights.keys():
            delta_weights[key] = merged_weights[key] - base_weights[key].to(device=merged_weights[key].device, dtype=merged_weights[key].dtype)
        
        print("### saving delta_weights...")
        save_file(delta_weights, os.path.join(model_path, "delta_weights.safetensors"))
        
    else:
        raise ValueError(f"{model_path} does not contain delta weights or merged weights")
        
    # Merge delta_weights to the target pipeline
    pipe = merge_delta_weights_into_unet(pipe, delta_weights)
    return pipe


def load_dreambooth_into_pipeline(pipe, sd_dreambooth: str) -> object:
    """
    Load DreamBooth weights into a diffusion pipeline.
    
    This function loads DreamBooth-trained weights from a checkpoint file
    and updates the UNet, VAE, and text encoder components of the pipeline.
    
    Args:
        pipe: The diffusion pipeline to update.
        sd_dreambooth (`str`):
            Path to the DreamBooth safetensors checkpoint file.
            
    Returns:
        The pipeline with updated weights.
    """
    assert sd_dreambooth.endswith(".safetensors")
    
    state_dict = {}
    with safe_open(sd_dreambooth, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    # Import conversion utilities
    from ..pipelines.stable_diffusion.convert_from_ckpt import (
        convert_ldm_unet_checkpoint,
        convert_ldm_vae_checkpoint,
        convert_ldm_clip_checkpoint
    )
    
    unet_config = {}  # unet, line 449 in convert_ldm_unet_checkpoint
    for key in pipe.unet.config.keys():
        if key != 'num_class_embeds':
            unet_config[key] = pipe.unet.config[key]
            
    pipe.unet.load_state_dict(convert_ldm_unet_checkpoint(state_dict, unet_config), strict=False)
    pipe.vae.load_state_dict(convert_ldm_vae_checkpoint(state_dict, pipe.vae.config))
    pipe.text_encoder = convert_ldm_clip_checkpoint(state_dict, text_encoder=pipe.text_encoder)
    return pipe
