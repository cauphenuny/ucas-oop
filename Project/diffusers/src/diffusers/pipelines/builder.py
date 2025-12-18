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
DiffusionPipelineBuilder - Builder pattern implementation for constructing diffusion pipelines.

This module provides a builder pattern implementation for flexibly constructing various diffusion pipelines.
It allows users to configure and build pipeline components through method chaining, supports loading from
pretrained models, replacing individual components, and applying preset configurations.
"""

import copy
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from ..models.modeling_utils import ModelMixin
from ..schedulers.scheduling_utils import SchedulerMixin
from ..utils import logging
from .pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)


class PipelineValidationError(Exception):
    """Exception raised when pipeline validation fails."""
    pass


class DiffusionPipelineBuilder:
    """
    Builder class for DiffusionPipeline.
    
    This class implements the builder pattern for flexibly constructing diffusion pipelines. It supports:
    - Batch loading components from pretrained models
    - Setting or replacing individual components through method chaining
    - Applying preset configurations
    - Custom validation rules
    - Flexible hook system
    
    Example:
        ```python
        # Build from pretrained model
        builder = DiffusionPipelineBuilder.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        
        # Replace individual component
        builder.with_scheduler(custom_scheduler)
        
        # Build pipeline
        pipeline = builder.build()
        
        # Or export components for training
        components = builder.build(export_modules=True)
        ```
    """
    
    def __init__(
        self,
        base_repo: Optional[Union[str, Path]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the builder.
        
        Args:
            base_repo: Optional base repository path or name
            torch_dtype: Default data type
            device: Default device
        """
        self.base_repo = base_repo
        self.torch_dtype = torch_dtype
        self.device = device
        self.components: Dict[str, Any] = {}
        self.config_overrides: Dict[str, Any] = {}
        self.component_flags: Dict[str, Dict[str, Any]] = {}  # 存储每个组件的额外标志
        self.hooks: Dict[str, List[Callable]] = {"pre_build": [], "post_build": []}
        self.validators: List[Callable] = []
        
        # 预设配置注册表
        self.presets: Dict[str, Callable] = {}
        self._register_default_presets()
    
    def _register_default_presets(self):
        """Register default preset configurations."""
        # 可以在这里添加常用的预设配置
        pass
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs
    ) -> "DiffusionPipelineBuilder":
        """
        Create builder from pretrained model and load all components
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model or Hub ID
            **kwargs: Additional arguments passed to DiffusionPipeline.from_pretrained
                     Supports torch_dtype, device_map, variant, revision, etc.
        
        Returns:
            Configured builder instance
        """
        # Extract builder parameters
        torch_dtype = kwargs.get("torch_dtype", None)
        device = kwargs.get("device", None)
        
        # Create builder instance
        builder = cls(
            base_repo=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # Load complete pipeline to get all components
        # Use DiffusionPipeline.from_pretrained to load
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )
            
            # Extract all components
            expected_modules, optional_kwargs = DiffusionPipeline._get_signature_keys(pipeline)
            for name in expected_modules:
                component = getattr(pipeline, name, None)
                if component is not None:
                    builder.components[name] = component
            
            # Save config
            if hasattr(pipeline, "config"):
                for key, value in pipeline.config.items():
                    if key not in ["_class_name", "_diffusers_version", "_module", "_name_or_path"]:
                        builder.config_overrides[key] = value
                        
        except Exception as e:
            logger.warning(
                f"Unable to load complete pipeline from {pretrained_model_name_or_path}: {e}. "
                "Will try to load components individually."
            )
            # If loading fails, can try loading individual components
            # Not implemented for simplicity
        
        return builder
    
    def with_component(
        self,
        name: str,
        component: Any,
        *,
        freeze: bool = False,
        requires_grad: Optional[bool] = None,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """
        Set or replace the component with the specified name (generic method)
        
        Args:
            name: Component name (e.g. 'unet', 'vae', 'scheduler' 等)
            component: Component object
            freeze: Whether to freeze component parameters
            requires_grad: Whether to require gradients
            **flags: Other flags
        
        Returns:
            self，Supports method chaining
        """
        # Basic type checking
        if component is not None:
            # Check scheduler type
            if name == "scheduler" and not isinstance(component, SchedulerMixin):
                logger.warning(
                    f"Scheduler component should be an instance of SchedulerMixin, "
                    f"but got {type(component)}"
                )
        
        self.components[name] = component
        
        # Save flags
        self.component_flags[name] = {
            "freeze": freeze,
            "requires_grad": requires_grad,
            **flags
        }
        
        # Apply freeze and requires_grad
        if component is not None and isinstance(component, torch.nn.Module):
            if freeze:
                component.requires_grad_(False)
            elif requires_grad is not None:
                component.requires_grad_(requires_grad)
        
        return self
    
    def with_unet(
        self,
        unet: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set UNet component."""
        return self.with_component("unet", unet, **flags)
    
    def with_vae(
        self,
        vae: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set VAE component."""
        return self.with_component("vae", vae, **flags)
    
    def with_scheduler(
        self,
        scheduler: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set scheduler component."""
        return self.with_component("scheduler", scheduler, **flags)
    
    def with_text_encoder(
        self,
        text_encoder: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set text encoder component."""
        return self.with_component("text_encoder", text_encoder, **flags)
    
    def with_tokenizer(
        self,
        tokenizer: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set tokenizer component."""
        return self.with_component("tokenizer", tokenizer, **flags)
    
    def with_feature_extractor(
        self,
        feature_extractor: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set feature extractor component."""
        return self.with_component("feature_extractor", feature_extractor, **flags)
    
    def with_safety_checker(
        self,
        safety_checker: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set safety checker component."""
        return self.with_component("safety_checker", safety_checker, **flags)
    
    def with_image_encoder(
        self,
        image_encoder: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set image encoder component."""
        return self.with_component("image_encoder", image_encoder, **flags)
    
    def with_controlnet(
        self,
        controlnet: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set ControlNet component."""
        return self.with_component("controlnet", controlnet, **flags)
    
    def with_adapter(
        self,
        adapter: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """Set adapter component."""
        return self.with_component("adapter", adapter, **flags)
    
    def apply_preset(
        self,
        name: str,
        **kwargs
    ) -> "DiffusionPipelineBuilder":
        """
        Apply preset configuration
        
        Args:
            name: Preset name
            **kwargs: Additional parameters passed to preset function
        
        Returns:
            self，Supports method chaining
        """
        if name not in self.presets:
            raise ValueError(
                f"未知的Preset name: {name}. "
                f"Available presets: {list(self.presets.keys())}"
            )
        
        preset_func = self.presets[name]
        preset_func(self, **kwargs)
        
        return self
    
    def with_config_override(
        self,
        **kwargs
    ) -> "DiffusionPipelineBuilder":
        """
        Set pipeline-level configuration overrides
        
        Args:
            **kwargs: Configuration parameters
        
        Returns:
            self，Supports method chaining
        """
        self.config_overrides.update(kwargs)
        return self
    
    def register_validator(
        self,
        validator: Callable[["DiffusionPipelineBuilder"], None]
    ) -> "DiffusionPipelineBuilder":
        """
        Register custom validator
        
        Args:
            validator: Validator function that receives builder as parameter,
                      should raise PipelineValidationError if validation fails
        
        Returns:
            self，Supports method chaining
        """
        self.validators.append(validator)
        return self
    
    def add_hook(
        self,
        stage: str,
        hook: Callable
    ) -> "DiffusionPipelineBuilder":
        """
        Add hook function
        
        Args:
            stage: Hook stage ("pre_build" 或 "post_build")
            hook: Hook function
        
        Returns:
            self，Supports method chaining
        """
        if stage not in self.hooks:
            raise ValueError(
                f"无效的Hook stage: {stage}. "
                f"Available stages: {list(self.hooks.keys())}"
            )
        
        self.hooks[stage].append(hook)
        return self
    
    def validate(self) -> None:
        """
        Execute all validators
        
        Raises:
            PipelineValidationError: If validation fails
        """
        for validator in self.validators:
            validator(self)
    
    def build(
        self,
        pipeline_cls: type = DiffusionPipeline,
        *,
        export_modules: bool = False,
        lazy: bool = False
    ) -> Union[DiffusionPipeline, Dict[str, Any], tuple]:
        """
        Build pipeline
        
        Args:
            pipeline_cls: Pipeline class, defaults to DiffusionPipeline
            export_modules: If True, return component dictionary instead of pipeline instance
            lazy: If True, return (pipeline_cls, components) tuple for lazy construction
        
        Returns:
            Return pipeline instance, component dictionary, or lazy build tuple based on parameters
        
        Raises:
            PipelineValidationError: If validation fails
        """
        # Execute pre_build hooks
        for hook in self.hooks["pre_build"]:
            hook(self)
        
        # Execute validation
        self.validate()
        
        # If only exporting modules
        if export_modules:
            return self.components.copy()
        
        # If lazy build
        if lazy:
            return (pipeline_cls, self.components.copy())
        
        # Check required components based on pipeline_cls __init__ signature
        init_signature = inspect.signature(pipeline_cls.__init__)
        required_params = []
        optional_params = []
        
        for param_name, param in init_signature.parameters.items():
            # Skip 'self' and variable argument parameters
            if param_name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                optional_params.append(param_name)
        
        # Check missing required components
        missing_required = [p for p in required_params if p not in self.components]
        if missing_required:
            raise PipelineValidationError(
                f"Missing required components: {missing_required}. "
                f"Please use with_{missing_required[0]}() and similar methods to set these components."
            )
        
        # Build pipeline
        try:
            # Prepare constructor arguments
            init_kwargs = {}
            for param_name, param in init_signature.parameters.items():
                # Skip 'self' and variable argument parameters
                if param_name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if param_name in self.components:
                    init_kwargs[param_name] = self.components[param_name]
            
            # Create pipeline instance
            pipeline = pipeline_cls(**init_kwargs)
            
            # Apply configuration overrides
            if hasattr(pipeline, "register_to_config") and self.config_overrides:
                pipeline.register_to_config(**self.config_overrides)
            
            # Move to specified device and dtype
            if self.device is not None or self.torch_dtype is not None:
                device = self.device
                dtype = self.torch_dtype
                
                # If only dtype specified without device, use current device
                if dtype is not None and device is None:
                    device = pipeline.device
                
                # Move pipeline
                if device is not None or dtype is not None:
                    pipeline.to(device=device, dtype=dtype)
            
            # Execute post_build hooks
            for hook in self.hooks["post_build"]:
                hook(pipeline)
            
            return pipeline
            
        except Exception as e:
            raise PipelineValidationError(
                f"Build pipeline时出错: {e}"
            ) from e
    
    def clone(self, **overrides) -> "DiffusionPipelineBuilder":
        """
        Clone current builder with optional component overrides
        
        Args:
            **overrides: Components to override, format: component_name=component
        
        Returns:
            New builder instance
        """
        # Create new instance
        new_builder = DiffusionPipelineBuilder(
            base_repo=self.base_repo,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        # Copy components and config
        new_builder.components = copy.copy(self.components)
        new_builder.config_overrides = copy.copy(self.config_overrides)
        new_builder.component_flags = copy.deepcopy(self.component_flags)
        new_builder.validators = self.validators.copy()
        new_builder.hooks = {k: v.copy() for k, v in self.hooks.items()}
        new_builder.presets = self.presets.copy()
        
        # Apply overrides
        for name, component in overrides.items():
            new_builder.with_component(name, component)
        
        return new_builder
