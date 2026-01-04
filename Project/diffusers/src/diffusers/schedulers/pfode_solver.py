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
        self.t_initial = t_initial
        self.t_terminal = t_terminal
        self.scheduler = scheduler

        train_step_terminal = 0
        train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps  # 0+1000
        
        self.stepsize = (t_terminal - t_initial) / (train_step_terminal - train_step_initial)  # 1/1000
    
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
        # (b,) -> (b,1)
        t_start = t_start[:, None]
        t_end = t_end[:, None]
        assert t_start.dim() == 2
        
        timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
        interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
        timepoints = t_start + interval * timepoints
        
        timesteps = (self.scheduler.num_train_timesteps - 1) + (timepoints - self.t_initial) / self.stepsize
        return timesteps.round().long()
    
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
        assert t_start.dim() == 1
        assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))
        
        do_classifier_free_guidance = True if guidance_scale > 1 else False
        bsz = latents.shape[0]
            
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
        timestep_cond = None
        if unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)
            
        
        timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
        timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)

        # Denoising loop
        with torch.no_grad():
            for i in range(num_steps):
                
                t = torch.cat([timesteps[:, i]]*2) if do_classifier_free_guidance else timesteps[:, i]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


                # STEP: compute the previous noisy sample x_t -> x_t-1
                batch_timesteps = timesteps[:, i].cpu()
                prev_timestep = batch_timesteps - timestep_interval

                alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
                alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
                for ib in range(prev_timestep.shape[0]): 
                    alpha_prod_t_prev[ib] = self.scheduler.alphas_cumprod[prev_timestep[ib]] if prev_timestep[ib] >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                
                alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
                alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)

                # compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (latents - beta_prod_t[:, None, None, None] ** 0.5 * noise_pred) / alpha_prod_t[:, None, None, None] ** 0.5
                    pred_epsilon = noise_pred
                elif self.scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t[:, None, None, None]**0.5) * latents - (beta_prod_t[:, None, None, None]**0.5) * noise_pred
                    pred_epsilon = (alpha_prod_t[:, None, None, None]**0.5) * noise_pred + (beta_prod_t[:, None, None, None]**0.5) * latents
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )
                    
                pred_sample_direction = (1 - alpha_prod_t_prev[:, None, None, None]) ** 0.5 * pred_epsilon
                latents = alpha_prod_t_prev[:, None, None, None] ** 0.5 * pred_original_sample + pred_sample_direction

            
        return latents


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
        self.t_initial = t_initial
        self.t_terminal = t_terminal
        self.scheduler = scheduler

        train_step_terminal = 0
        train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps  # 0+1000
        
        self.stepsize = (t_terminal - t_initial) / (train_step_terminal - train_step_initial)  # 1/1000
    
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
        # (b,) -> (b,1)
        t_start = t_start[:, None]
        t_end = t_end[:, None]
        assert t_start.dim() == 2
        
        timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
        interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
        timepoints = t_start + interval * timepoints
        
        timesteps = (self.scheduler.num_train_timesteps - 1) + (timepoints - self.t_initial) / self.stepsize
        return timesteps.round().long()
    
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
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
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
        assert t_start.dim() == 1
        assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))
        dtype = latents.dtype
        device = latents.device
        bsz = latents.shape[0]
        do_classifier_free_guidance = True if guidance_scale > 1 else False
        
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = torch.cat(
            [self._get_add_time_ids((resolution, resolution), (0, 0), (resolution, resolution), dtype) for _ in range(bsz)]
        ).to(device)
        negative_add_time_ids = add_time_ids
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
            
        timestep_cond = None
        if unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)
            
        
        timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
        timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)

        # Denoising loop
        with torch.no_grad():
            for i in range(num_steps):
                # expand the latents if we are doing classifier free guidance
                t = torch.cat([timesteps[:, i]]*2) if do_classifier_free_guidance else timesteps[:, i]
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


                # STEP: compute the previous noisy sample x_t -> x_t-1
                batch_timesteps = timesteps[:, i].cpu()
                prev_timestep = batch_timesteps - timestep_interval

                alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
                alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
                for ib in range(prev_timestep.shape[0]): 
                    alpha_prod_t_prev[ib] = self.scheduler.alphas_cumprod[prev_timestep[ib]] if prev_timestep[ib] >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                
                alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
                alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)

                # compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (latents - beta_prod_t[:, None, None, None] ** 0.5 * noise_pred) / alpha_prod_t[:, None, None, None] ** 0.5
                    pred_epsilon = noise_pred
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )
                    
                pred_sample_direction = (1 - alpha_prod_t_prev[:, None, None, None]) ** 0.5 * pred_epsilon
                latents = alpha_prod_t_prev[:, None, None, None] ** 0.5 * pred_original_sample + pred_sample_direction

            
        return latents
