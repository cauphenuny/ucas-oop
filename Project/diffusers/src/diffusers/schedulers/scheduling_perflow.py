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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


class TimeWindows:
    """
    Helper class to manage time windows for piecewise rectified flow.
    
    Args:
        t_initial (`float`, defaults to 1.0):
            The initial time value.
        t_terminal (`float`, defaults to 0.0):
            The terminal time value.
        num_windows (`int`, defaults to 4):
            The number of time windows to use.
        precision (`float`, defaults to 1.0/1000):
            The precision for numerical comparisons.
    """
    
    def __init__(self, t_initial: float = 1.0, t_terminal: float = 0.0, num_windows: int = 4, precision: float = 1.0/1000) -> None:
        """Initialize time windows for piecewise rectified flow."""
        raise NotImplementedError("TimeWindows.__init__ is not implemented yet")
    
    def get_window(self, tp: float) -> Tuple[float, float]:
        """
        Get the time window bounds for a given timepoint.
        
        Args:
            tp (`float`):
                The timepoint to query.
                
        Returns:
            `Tuple[float, float]`: A tuple of (window_start, window_end).
        """
        raise NotImplementedError("TimeWindows.get_window is not implemented yet")
    
    def lookup_window(self, timepoint: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Lookup time window bounds for batched timepoints.
        
        Args:
            timepoint (`torch.FloatTensor`):
                The timepoints to query, can be scalar or batched.
                
        Returns:
            `Tuple[torch.FloatTensor, torch.FloatTensor]`: A tuple of (window_starts, window_ends).
        """
        raise NotImplementedError("TimeWindows.lookup_window is not implemented yet")


@dataclass
class PeRFlowSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
    alpha_transform_type: str = "cosine",
) -> torch.FloatTensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function.

    Args:
        num_diffusion_timesteps (`int`):
            The number of betas to produce.
        max_beta (`float`, defaults to 0.999):
            The maximum beta to use; use values lower than 1 to prevent singularities.
        alpha_transform_type (`str`, defaults to `"cosine"`):
            The type of noise schedule for alpha_bar. Choose from `"cosine"` or `"exp"`.

    Returns:
        `torch.FloatTensor`: The betas used by the scheduler to step the model outputs.
    """
    raise NotImplementedError("betas_for_alpha_bar is not implemented yet")


class PeRFlowScheduler(SchedulerMixin, ConfigMixin):
    """
    `PeRFlowScheduler` implements piecewise rectified flow for accelerated diffusion sampling.

    This scheduler extends the denoising procedure with piecewise linear flow approximations
    across multiple time windows, enabling faster sampling with fewer steps.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.00085):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.012):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"scaled_linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `"linear"`, `"scaled_linear"`, or `"squaredcos_cap_v2"`.
        trained_betas (`np.ndarray` or `List[float]`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        set_alpha_to_one (`bool`, defaults to `False`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        prediction_type (`str`, defaults to `"ddim_eps"`):
            Prediction type of the scheduler function. Choose from `"ddim_eps"`, `"diff_eps"`, or `"velocity"`.
        t_noise (`float`, defaults to 1.0):
            The initial noise time value.
        t_clean (`float`, defaults to 0.0):
            The clean sample time value.
        num_time_windows (`int`, defaults to 4):
            The number of time windows for piecewise approximation.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        set_alpha_to_one: bool = False,
        prediction_type: str = "ddim_eps",
        t_noise: float = 1.0,
        t_clean: float = 0.0,
        num_time_windows: int = 4,
    ):
        """Initialize the PeRFlowScheduler."""
        raise NotImplementedError("PeRFlowScheduler.__init__ is not implemented yet")

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        raise NotImplementedError("PeRFlowScheduler.scale_model_input is not implemented yet")

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        raise NotImplementedError("PeRFlowScheduler.set_timesteps is not implemented yet")

    def get_window_alpha(self, timepoints: torch.FloatTensor) -> Tuple[torch.FloatTensor, ...]:
        """
        Compute alpha-related values for the time windows.

        Args:
            timepoints (`torch.FloatTensor`):
                The timepoints to compute alpha values for.

        Returns:
            `Tuple[torch.FloatTensor, ...]`: A tuple of window-related alpha values including:
                - t_win_start: Start time of the window
                - t_win_end: End time of the window
                - t_win_len: Length of the window
                - t_interval: Time interval from window start
                - gamma_s_e: Gamma value from start to end
                - alphas_cumprod_start: Cumulative alpha product at window start
                - alphas_cumprod_end: Cumulative alpha product at window end
        """
        raise NotImplementedError("PeRFlowScheduler.get_window_alpha is not implemented yet")

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[PeRFlowSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_perflow.PeRFlowSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.PeRFlowSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_perflow.PeRFlowSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        raise NotImplementedError("PeRFlowScheduler.step is not implemented yet")

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to the original samples according to the noise schedule.

        Args:
            original_samples (`torch.FloatTensor`):
                The original samples to add noise to.
            noise (`torch.FloatTensor`):
                The noise to add to the samples.
            timesteps (`torch.IntTensor`):
                The timesteps indicating the noise level.

        Returns:
            `torch.FloatTensor`: The noisy samples.
        """
        raise NotImplementedError("PeRFlowScheduler.add_noise is not implemented yet")

    def __len__(self):
        """Return the number of training timesteps."""
        raise NotImplementedError("PeRFlowScheduler.__len__ is not implemented yet")
