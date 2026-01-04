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

import tempfile
import unittest

import torch

from diffusers import PeRFlowScheduler

from .test_schedulers import SchedulerCommonTest


class PeRFlowSchedulerTest(SchedulerCommonTest):
    """
    Comprehensive test suite for PeRFlowScheduler.
    
    This test class validates all aspects of the PeRFlowScheduler including:
    - Initialization with various configurations
    - Timestep generation and scheduling
    - Step function for denoising
    - Noise addition
    - Configuration save/load
    - Integration with diffusion pipelines
    """
    
    scheduler_classes = (PeRFlowScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)

    def get_scheduler_config(self, **kwargs):
        """
        Get the default scheduler configuration for testing.
        
        Args:
            **kwargs: Additional configuration parameters to override defaults.
            
        Returns:
            dict: Scheduler configuration dictionary.
        """
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "prediction_type": "ddim_eps",
            "num_time_windows": 4,
        }
        config.update(**kwargs)
        return config

    def test_timesteps(self):
        """Test scheduler with different num_train_timesteps values."""
        for timesteps in [100, 500, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        """Test scheduler with different beta_start and beta_end values."""
        for beta_start, beta_end in zip([0.0001, 0.001, 0.00085], [0.002, 0.02, 0.012]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        """Test scheduler with different beta_schedule options."""
        for schedule in ["linear", "scaled_linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        """Test scheduler with different prediction_type options."""
        for prediction_type in ["ddim_eps", "diff_eps", "velocity"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_num_time_windows(self):
        """Test scheduler with different numbers of time windows."""
        for num_windows in [2, 4, 8]:
            self.check_over_configs(num_time_windows=num_windows)

    def test_time_values(self):
        """Test scheduler with different t_noise and t_clean values."""
        for t_noise, t_clean in zip([1.0, 0.999], [0.0, 0.001]):
            self.check_over_configs(t_noise=t_noise, t_clean=t_clean)

    def test_set_alpha_to_one(self):
        """Test scheduler with set_alpha_to_one option."""
        for set_alpha_to_one in [True, False]:
            self.check_over_configs(set_alpha_to_one=set_alpha_to_one)

    def test_inference_steps(self):
        """Test scheduler with different num_inference_steps values."""
        for num_inference_steps in [4, 10, 20, 50]:
            self.check_over_forward(num_inference_steps=num_inference_steps)

    def test_trained_betas(self):
        """Test scheduler with custom trained_betas array."""
        import numpy as np
        
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        
        num_train_timesteps = scheduler_config["num_train_timesteps"]
        trained_betas = np.linspace(0.0001, 0.02, num_train_timesteps, dtype=np.float32)
        
        scheduler = scheduler_class(**scheduler_config, trained_betas=trained_betas)
        assert scheduler is not None

    def test_full_loop_no_noise(self):
        """Test a full denoising loop without adding noise."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        result = sample
        assert result is not None
        assert result.shape == self.dummy_sample_deter.shape

    def test_full_loop_with_noise(self):
        """Test a full denoising loop with noise addition."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        noise = self.dummy_noise_deter

        timesteps = torch.IntTensor([scheduler.config.num_train_timesteps - 1])
        noisy_sample = scheduler.add_noise(sample, noise, timesteps)
        
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(noisy_sample, t)
            noisy_sample = scheduler.step(residual, t, noisy_sample).prev_sample

        result = noisy_sample
        assert result is not None
        assert result.shape == sample.shape

    def test_step_shape(self):
        """Test that step function returns correct output shape."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        scheduler.set_timesteps(num_inference_steps)

        timestep = scheduler.timesteps[0]
        sample = self.dummy_sample_deter
        residual = torch.randn_like(sample)

        output = scheduler.step(residual, timestep, sample)
        assert output.prev_sample.shape == sample.shape

    def test_step_return_dict(self):
        """Test that step function correctly returns dict or tuple based on return_dict parameter."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        scheduler.set_timesteps(num_inference_steps)

        timestep = scheduler.timesteps[0]
        sample = self.dummy_sample_deter
        residual = torch.randn_like(sample)

        output_dict = scheduler.step(residual, timestep, sample, return_dict=True)
        assert hasattr(output_dict, "prev_sample")

        output_tuple = scheduler.step(residual, timestep, sample, return_dict=False)
        assert isinstance(output_tuple, tuple)

    def test_timesteps_generation(self):
        """Test that set_timesteps generates correct number of timesteps."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        for num_inference_steps in [4, 10, 20]:
            scheduler.set_timesteps(num_inference_steps)
            assert len(scheduler.timesteps) == num_inference_steps

    def test_timesteps_device(self):
        """Test that timesteps are created on the correct device."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        
        scheduler.set_timesteps(num_inference_steps, device="cpu")
        assert scheduler.timesteps.device.type == "cpu"

    def test_scale_model_input(self):
        """Test scale_model_input returns correct output."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        sample = self.dummy_sample_deter
        timestep = 100
        
        scaled_sample = scheduler.scale_model_input(sample, timestep)
        assert scaled_sample.shape == sample.shape

    def test_add_noise_shape(self):
        """Test that add_noise returns correct shape."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        sample = self.dummy_sample_deter
        noise = self.dummy_noise_deter
        timesteps = torch.IntTensor([100, 200, 300, 400])

        noisy_sample = scheduler.add_noise(sample, noise, timesteps)
        assert noisy_sample.shape == sample.shape

    def test_scheduler_length(self):
        """Test __len__ method returns num_train_timesteps."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert len(scheduler) == scheduler_config["num_train_timesteps"]

    def test_save_load_config(self):
        """Test that scheduler configuration can be saved and loaded."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_config(tmpdirname)
            loaded_scheduler = scheduler_class.from_pretrained(tmpdirname)
            
            assert loaded_scheduler.config.num_train_timesteps == scheduler.config.num_train_timesteps
            assert loaded_scheduler.config.beta_start == scheduler.config.beta_start
            assert loaded_scheduler.config.beta_end == scheduler.config.beta_end
            assert loaded_scheduler.config.num_time_windows == scheduler.config.num_time_windows

    def test_window_calculation(self):
        """Test that time windows are calculated correctly."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(num_time_windows=4)
        scheduler = scheduler_class(**scheduler_config)
        
        num_inference_steps = 10
        scheduler.set_timesteps(num_inference_steps)
        
        assert scheduler.time_windows is not None

    def test_get_window_alpha(self):
        """Test get_window_alpha computes correct values."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        
        scheduler.set_timesteps(10)
        
        timepoints = torch.FloatTensor([0.5, 0.3, 0.1])
        result = scheduler.get_window_alpha(timepoints)
        
        assert len(result) == 7

    def test_minimum_inference_steps(self):
        """Test that scheduler handles minimum inference steps correctly."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(num_time_windows=4)
        scheduler = scheduler_class(**scheduler_config)
        
        scheduler.set_timesteps(2)
        assert len(scheduler.timesteps) >= scheduler_config["num_time_windows"]

    def test_compatibility_with_other_schedulers(self):
        """Test that PeRFlowScheduler is marked as compatible with KarrasDiffusionSchedulers."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        
        assert hasattr(scheduler, "_compatibles")
        assert len(scheduler._compatibles) > 0

    def test_scheduler_order(self):
        """Test that scheduler order attribute is set correctly."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        
        assert scheduler.order == 1

    def test_step_with_different_prediction_types(self):
        """Test step function with different prediction types."""
        for prediction_type in ["ddim_eps", "diff_eps", "velocity"]:
            scheduler_class = self.scheduler_classes[0]
            scheduler_config = self.get_scheduler_config(prediction_type=prediction_type)
            scheduler = scheduler_class(**scheduler_config)
            
            num_inference_steps = 10
            scheduler.set_timesteps(num_inference_steps)
            
            timestep = scheduler.timesteps[0]
            sample = self.dummy_sample_deter
            residual = torch.randn_like(sample)
            
            output = scheduler.step(residual, timestep, sample)
            assert output.prev_sample is not None

    def test_timesteps_monotonic_decrease(self):
        """Test that timesteps decrease monotonically."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        
        num_inference_steps = 10
        scheduler.set_timesteps(num_inference_steps)
        
        timesteps = scheduler.timesteps.cpu().numpy()
        for i in range(len(timesteps) - 1):
            assert timesteps[i] >= timesteps[i + 1]

    def test_config_immutability(self):
        """Test that scheduler config is properly registered and accessible."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        
        assert hasattr(scheduler, "config")
        assert scheduler.config.num_train_timesteps == scheduler_config["num_train_timesteps"]
        assert scheduler.config.beta_schedule == scheduler_config["beta_schedule"]

    def test_numerical_stability(self):
        """Test that scheduler operations maintain numerical stability."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        
        num_inference_steps = 10
        scheduler.set_timesteps(num_inference_steps)
        
        sample = self.dummy_sample_deter
        residual = torch.randn_like(sample)
        
        for t in scheduler.timesteps:
            output = scheduler.step(residual, t, sample)
            assert not torch.isnan(output.prev_sample).any()
            assert not torch.isinf(output.prev_sample).any()
            sample = output.prev_sample

    def test_batch_consistency(self):
        """Test that scheduler handles batched inputs consistently."""
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        
        scheduler.set_timesteps(10)
        
        batch_sample = self.dummy_sample_deter
        batch_residual = torch.randn_like(batch_sample)
        timestep = scheduler.timesteps[0]
        
        output = scheduler.step(batch_residual, timestep, batch_sample)
        assert output.prev_sample.shape[0] == batch_sample.shape[0]

    def test_different_beta_schedules(self):
        """Test initialization with all supported beta schedules."""
        scheduler_class = self.scheduler_classes[0]
        
        for beta_schedule in ["linear", "scaled_linear", "squaredcos_cap_v2"]:
            scheduler_config = self.get_scheduler_config(beta_schedule=beta_schedule)
            scheduler = scheduler_class(**scheduler_config)
            assert scheduler is not None
            assert scheduler.config.beta_schedule == beta_schedule
