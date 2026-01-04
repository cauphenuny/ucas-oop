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

from collections import OrderedDict
from typing import Optional

import torch


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
    raise NotImplementedError("merge_delta_weights_into_unet is not implemented yet")


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
    raise NotImplementedError("load_delta_weights_into_unet is not implemented yet")


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
    raise NotImplementedError("load_dreambooth_into_pipeline is not implemented yet")
