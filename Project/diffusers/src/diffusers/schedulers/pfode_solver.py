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
Piecewise Flow ODE Solver for accelerated diffusion sampling.
"""

from typing import Optional

import torch


class PFODESolver:
    """
    Piecewise Flow ODE Solver for Stable Diffusion models.
    
    This solver implements the piecewise rectified flow algorithm for accelerated
    sampling in diffusion models.
    
    Args:
        scheduler: The diffusion scheduler to use.
        t_initial (`float`, defaults to 1.0):
            The initial time value for the flow.
        t_terminal (`float`, defaults to 0.0):
            The terminal time value for the flow.
    """
    
    def __init__(self, scheduler, t_initial: float = 1.0, t_terminal: float = 0.0) -> None:
        """Initialize the PFODESolver."""
        raise NotImplementedError("PFODESolver.__init__ is not implemented yet")
    
    def get_timesteps(self, t_start: torch.FloatTensor, t_end: torch.FloatTensor, num_steps: int) -> torch.LongTensor:
        """
        Generate timesteps for the ODE solver.
        
        Args:
            t_start (`torch.FloatTensor` of shape `(batch_size,)`):
                Starting time for each sample in the batch.
            t_end (`torch.FloatTensor` of shape `(batch_size,)`):
                Ending time for each sample in the batch.
            num_steps (`int`):
                Number of steps to generate.
                
        Returns:
            `torch.LongTensor` of shape `(batch_size, num_steps)`: The timesteps for the ODE solver.
        """
        raise NotImplementedError("PFODESolver.get_timesteps is not implemented yet")
    
    def solve(
        self,
        latents: torch.FloatTensor,
        unet,
        t_start: torch.FloatTensor,
        t_end: torch.FloatTensor,
        prompt_embeds: torch.FloatTensor,
        negative_prompt_embeds: torch.FloatTensor,
        guidance_scale: float = 1.0,
        num_steps: int = 2,
        num_windows: int = 1,
    ) -> torch.FloatTensor:
        """
        Solve the piecewise flow ODE.
        
        Args:
            latents (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                The initial latents to denoise.
            unet:
                The UNet model to use for denoising.
            t_start (`torch.FloatTensor` of shape `(batch_size,)`):
                Starting time for each sample.
            t_end (`torch.FloatTensor` of shape `(batch_size,)`):
                Ending time for each sample.
            prompt_embeds (`torch.FloatTensor`):
                Text embeddings for the prompts.
            negative_prompt_embeds (`torch.FloatTensor`):
                Text embeddings for the negative prompts.
            guidance_scale (`float`, defaults to 1.0):
                Guidance scale for classifier-free guidance.
            num_steps (`int`, defaults to 2):
                Number of steps for the ODE solver.
            num_windows (`int`, defaults to 1):
                Number of time windows to use.
                
        Returns:
            `torch.FloatTensor`: The denoised latents.
        """
        raise NotImplementedError("PFODESolver.solve is not implemented yet")


class PFODESolverSDXL:
    """
    Piecewise Flow ODE Solver for Stable Diffusion XL models.
    
    This solver implements the piecewise rectified flow algorithm for accelerated
    sampling in SDXL diffusion models, with support for additional conditioning.
    
    Args:
        scheduler: The diffusion scheduler to use.
        t_initial (`float`, defaults to 1.0):
            The initial time value for the flow.
        t_terminal (`float`, defaults to 0.0):
            The terminal time value for the flow.
    """
    
    def __init__(self, scheduler, t_initial: float = 1.0, t_terminal: float = 0.0) -> None:
        """Initialize the PFODESolverSDXL."""
        raise NotImplementedError("PFODESolverSDXL.__init__ is not implemented yet")
    
    def get_timesteps(self, t_start: torch.FloatTensor, t_end: torch.FloatTensor, num_steps: int) -> torch.LongTensor:
        """
        Generate timesteps for the ODE solver.
        
        Args:
            t_start (`torch.FloatTensor` of shape `(batch_size,)`):
                Starting time for each sample in the batch.
            t_end (`torch.FloatTensor` of shape `(batch_size,)`):
                Ending time for each sample in the batch.
            num_steps (`int`):
                Number of steps to generate.
                
        Returns:
            `torch.LongTensor` of shape `(batch_size, num_steps)`: The timesteps for the ODE solver.
        """
        raise NotImplementedError("PFODESolverSDXL.get_timesteps is not implemented yet")
    
    def _get_add_time_ids(
        self,
        original_size: tuple,
        crops_coords_top_left: tuple,
        target_size: tuple,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        """
        Get additional time embeddings for SDXL conditioning.
        
        Args:
            original_size (`tuple`):
                The original size of the image (height, width).
            crops_coords_top_left (`tuple`):
                The crop coordinates (top, left).
            target_size (`tuple`):
                The target size of the image (height, width).
            dtype (`torch.dtype`):
                The data type for the embeddings.
                
        Returns:
            `torch.FloatTensor`: The additional time embeddings.
        """
        raise NotImplementedError("PFODESolverSDXL._get_add_time_ids is not implemented yet")
    
    def solve(
        self,
        latents: torch.FloatTensor,
        unet,
        t_start: torch.FloatTensor,
        t_end: torch.FloatTensor,
        prompt_embeds: torch.FloatTensor,
        pooled_prompt_embeds: torch.FloatTensor,
        negative_prompt_embeds: torch.FloatTensor,
        negative_pooled_prompt_embeds: torch.FloatTensor,
        guidance_scale: float = 1.0,
        num_steps: int = 10,
        num_windows: int = 4,
        resolution: int = 1024,
    ) -> torch.FloatTensor:
        """
        Solve the piecewise flow ODE for SDXL.
        
        Args:
            latents (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                The initial latents to denoise.
            unet:
                The UNet model to use for denoising.
            t_start (`torch.FloatTensor` of shape `(batch_size,)`):
                Starting time for each sample.
            t_end (`torch.FloatTensor` of shape `(batch_size,)`):
                Ending time for each sample.
            prompt_embeds (`torch.FloatTensor`):
                Text embeddings for the prompts.
            pooled_prompt_embeds (`torch.FloatTensor`):
                Pooled text embeddings for SDXL.
            negative_prompt_embeds (`torch.FloatTensor`):
                Text embeddings for the negative prompts.
            negative_pooled_prompt_embeds (`torch.FloatTensor`):
                Pooled text embeddings for negative prompts.
            guidance_scale (`float`, defaults to 1.0):
                Guidance scale for classifier-free guidance.
            num_steps (`int`, defaults to 10):
                Number of steps for the ODE solver.
            num_windows (`int`, defaults to 4):
                Number of time windows to use.
            resolution (`int`, defaults to 1024):
                The resolution of the output image.
                
        Returns:
            `torch.FloatTensor`: The denoised latents.
        """
        raise NotImplementedError("PFODESolverSDXL.solve is not implemented yet")
