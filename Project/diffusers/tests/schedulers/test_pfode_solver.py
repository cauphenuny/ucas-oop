# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import unittest

import torch

from diffusers import PeRFlowScheduler
from diffusers.schedulers.pfode_solver import PFODESolver, PFODESolverSDXL


class DummyUNet:
    """Dummy UNet model for testing."""
    
    class Config:
        time_cond_proj_dim = None
    
    config = Config()
    
    def __call__(self, latents, timestep, encoder_hidden_states, **kwargs):
        """Return dummy noise prediction."""
        return (torch.randn_like(latents),)


class DummyUNetSDXL:
    """Dummy UNet model for SDXL testing."""
    
    class Config:
        time_cond_proj_dim = None
    
    config = Config()
    
    def __call__(self, latents, timestep, encoder_hidden_states, added_cond_kwargs, **kwargs):
        """Return dummy noise prediction."""
        return (torch.randn_like(latents),)


class PFODESolverTest(unittest.TestCase):
    """
    Comprehensive test suite for PFODESolver.
    
    This test class validates all aspects of the PFODESolver including:
    - Initialization
    - Timestep generation
    - ODE solving with various configurations
    - Integration with schedulers
    """

    def get_scheduler(self):
        """Get a default scheduler for testing."""
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "prediction_type": "ddim_eps",
            "num_time_windows": 4,
        }
        return PeRFlowScheduler(**config)

    def test_initialization(self):
        """Test PFODESolver initialization."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler, t_initial=1.0, t_terminal=0.0)
        assert solver is not None

    def test_initialization_custom_times(self):
        """Test PFODESolver initialization with custom time values."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler, t_initial=0.999, t_terminal=0.001)
        assert solver is not None

    def test_get_timesteps_shape(self):
        """Test that get_timesteps returns correct shape."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 4
        num_steps = 10
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        
        timesteps = solver.get_timesteps(t_start, t_end, num_steps)
        assert timesteps.shape == (batch_size, num_steps)

    def test_get_timesteps_dtype(self):
        """Test that get_timesteps returns long tensor."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 2
        num_steps = 5
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        
        timesteps = solver.get_timesteps(t_start, t_end, num_steps)
        assert timesteps.dtype == torch.long

    def test_get_timesteps_values_range(self):
        """Test that timesteps are within valid range."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 2
        num_steps = 5
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        
        timesteps = solver.get_timesteps(t_start, t_end, num_steps)
        assert (timesteps >= 0).all()
        assert (timesteps < scheduler.config.num_train_timesteps).all()

    def test_get_timesteps_different_batch_sizes(self):
        """Test get_timesteps with different batch sizes."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        for batch_size in [1, 2, 4, 8]:
            num_steps = 10
            t_start = torch.ones(batch_size)
            t_end = torch.zeros(batch_size)
            
            timesteps = solver.get_timesteps(t_start, t_end, num_steps)
            assert timesteps.shape == (batch_size, num_steps)

    def test_solve_basic(self):
        """Test basic solve functionality."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 1
        num_channels = 4
        height = 8
        width = 8
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNet()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 768)
        negative_prompt_embeds = torch.randn(batch_size, 77, 768)
        
        result = solver.solve(
            latents,
            unet,
            t_start,
            t_end,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_scale=1.0,
            num_steps=2,
            num_windows=1,
        )
        
        assert result.shape == latents.shape

    def test_solve_with_guidance(self):
        """Test solve with classifier-free guidance."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 2
        num_channels = 4
        height = 8
        width = 8
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNet()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 768)
        negative_prompt_embeds = torch.randn(batch_size, 77, 768)
        
        result = solver.solve(
            latents,
            unet,
            t_start,
            t_end,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_scale=7.5,
            num_steps=5,
            num_windows=2,
        )
        
        assert result.shape == latents.shape

    def test_solve_different_num_steps(self):
        """Test solve with different numbers of steps."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 1
        num_channels = 4
        height = 8
        width = 8
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNet()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 768)
        negative_prompt_embeds = torch.randn(batch_size, 77, 768)
        
        for num_steps in [2, 5, 10]:
            result = solver.solve(
                latents,
                unet,
                t_start,
                t_end,
                prompt_embeds,
                negative_prompt_embeds,
                guidance_scale=1.0,
                num_steps=num_steps,
                num_windows=1,
            )
            assert result.shape == latents.shape

    def test_solve_different_num_windows(self):
        """Test solve with different numbers of windows."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 1
        num_channels = 4
        height = 8
        width = 8
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNet()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 768)
        negative_prompt_embeds = torch.randn(batch_size, 77, 768)
        
        for num_windows in [1, 2, 4]:
            result = solver.solve(
                latents,
                unet,
                t_start,
                t_end,
                prompt_embeds,
                negative_prompt_embeds,
                guidance_scale=1.0,
                num_steps=10,
                num_windows=num_windows,
            )
            assert result.shape == latents.shape

    def test_solve_batched(self):
        """Test solve with batched inputs."""
        scheduler = self.get_scheduler()
        solver = PFODESolver(scheduler)
        
        batch_size = 4
        num_channels = 4
        height = 8
        width = 8
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNet()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 768)
        negative_prompt_embeds = torch.randn(batch_size, 77, 768)
        
        result = solver.solve(
            latents,
            unet,
            t_start,
            t_end,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_scale=1.0,
            num_steps=5,
            num_windows=2,
        )
        
        assert result.shape == latents.shape


class PFODESolverSDXLTest(unittest.TestCase):
    """
    Comprehensive test suite for PFODESolverSDXL.
    
    This test class validates all aspects of the PFODESolverSDXL including:
    - Initialization
    - Timestep generation
    - SDXL-specific conditioning
    - ODE solving with various configurations
    """

    def get_scheduler(self):
        """Get a default scheduler for testing."""
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "prediction_type": "ddim_eps",
            "num_time_windows": 4,
        }
        return PeRFlowScheduler(**config)

    def test_initialization(self):
        """Test PFODESolverSDXL initialization."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler, t_initial=1.0, t_terminal=0.0)
        assert solver is not None

    def test_get_timesteps_shape(self):
        """Test that get_timesteps returns correct shape."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        batch_size = 4
        num_steps = 10
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        
        timesteps = solver.get_timesteps(t_start, t_end, num_steps)
        assert timesteps.shape == (batch_size, num_steps)

    def test_get_add_time_ids(self):
        """Test _get_add_time_ids returns correct embeddings."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        dtype = torch.float32
        
        time_ids = solver._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype)
        assert time_ids is not None
        assert time_ids.dtype == dtype

    def test_get_add_time_ids_shape(self):
        """Test that _get_add_time_ids returns correct shape."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        dtype = torch.float32
        
        time_ids = solver._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype)
        assert time_ids.shape == (1, 6)

    def test_get_add_time_ids_different_sizes(self):
        """Test _get_add_time_ids with different image sizes."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        for size in [512, 768, 1024]:
            original_size = (size, size)
            crops_coords_top_left = (0, 0)
            target_size = (size, size)
            dtype = torch.float32
            
            time_ids = solver._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype)
            assert time_ids.shape == (1, 6)

    def test_solve_basic(self):
        """Test basic solve functionality for SDXL."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        batch_size = 1
        num_channels = 4
        height = 16
        width = 16
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNetSDXL()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 2048)
        pooled_prompt_embeds = torch.randn(batch_size, 1280)
        negative_prompt_embeds = torch.randn(batch_size, 77, 2048)
        negative_pooled_prompt_embeds = torch.randn(batch_size, 1280)
        
        result = solver.solve(
            latents,
            unet,
            t_start,
            t_end,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            guidance_scale=1.0,
            num_steps=2,
            num_windows=1,
            resolution=1024,
        )
        
        assert result.shape == latents.shape

    def test_solve_with_guidance(self):
        """Test solve with classifier-free guidance for SDXL."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        batch_size = 2
        num_channels = 4
        height = 16
        width = 16
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNetSDXL()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 2048)
        pooled_prompt_embeds = torch.randn(batch_size, 1280)
        negative_prompt_embeds = torch.randn(batch_size, 77, 2048)
        negative_pooled_prompt_embeds = torch.randn(batch_size, 1280)
        
        result = solver.solve(
            latents,
            unet,
            t_start,
            t_end,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            guidance_scale=7.5,
            num_steps=5,
            num_windows=2,
            resolution=1024,
        )
        
        assert result.shape == latents.shape

    def test_solve_different_resolutions(self):
        """Test solve with different resolution values."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        batch_size = 1
        num_channels = 4
        height = 16
        width = 16
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNetSDXL()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 2048)
        pooled_prompt_embeds = torch.randn(batch_size, 1280)
        negative_prompt_embeds = torch.randn(batch_size, 77, 2048)
        negative_pooled_prompt_embeds = torch.randn(batch_size, 1280)
        
        for resolution in [512, 768, 1024]:
            result = solver.solve(
                latents,
                unet,
                t_start,
                t_end,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                guidance_scale=1.0,
                num_steps=5,
                num_windows=2,
                resolution=resolution,
            )
            assert result.shape == latents.shape

    def test_solve_batched(self):
        """Test solve with batched inputs for SDXL."""
        scheduler = self.get_scheduler()
        solver = PFODESolverSDXL(scheduler)
        
        batch_size = 4
        num_channels = 4
        height = 16
        width = 16
        
        latents = torch.randn(batch_size, num_channels, height, width)
        unet = DummyUNetSDXL()
        t_start = torch.ones(batch_size)
        t_end = torch.zeros(batch_size)
        prompt_embeds = torch.randn(batch_size, 77, 2048)
        pooled_prompt_embeds = torch.randn(batch_size, 1280)
        negative_prompt_embeds = torch.randn(batch_size, 77, 2048)
        negative_pooled_prompt_embeds = torch.randn(batch_size, 1280)
        
        result = solver.solve(
            latents,
            unet,
            t_start,
            t_end,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            guidance_scale=1.0,
            num_steps=5,
            num_windows=2,
            resolution=1024,
        )
        
        assert result.shape == latents.shape
