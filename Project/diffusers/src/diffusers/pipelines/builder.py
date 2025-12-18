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
DiffusionPipelineBuilder - 建造者模式实现用于构建扩散管道

这个模块提供了一个建造者模式的实现，用于灵活地构建各种扩散管道（Diffusion Pipeline）。
它允许用户通过链式调用来配置和构建管道组件，支持从预训练模型加载、单独替换组件、
以及应用预设配置等功能。
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
    """管道验证错误异常类"""
    pass


class DiffusionPipelineBuilder:
    """
    DiffusionPipeline 的建造者类
    
    这个类实现了建造者模式，用于灵活构建扩散管道。支持：
    - 从预训练模型批量加载组件
    - 通过链式调用单独设置或替换组件
    - 应用预设配置
    - 自定义验证规则
    - 灵活的钩子系统
    
    示例:
        ```python
        # 从预训练模型构建
        builder = DiffusionPipelineBuilder.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        
        # 替换单个组件
        builder.with_scheduler(custom_scheduler)
        
        # 构建管道
        pipeline = builder.build()
        
        # 或者导出组件用于训练
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
        初始化建造者
        
        Args:
            base_repo: 可选的基础仓库路径或名称
            torch_dtype: 默认的数据类型
            device: 默认的设备
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
        """注册默认预设配置"""
        # 可以在这里添加常用的预设配置
        pass
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs
    ) -> "DiffusionPipelineBuilder":
        """
        从预训练模型创建建造者并加载所有组件
        
        Args:
            pretrained_model_name_or_path: 预训练模型的路径或Hub ID
            **kwargs: 传递给 DiffusionPipeline.from_pretrained 的其他参数
                     支持 torch_dtype, device_map, variant, revision 等
        
        Returns:
            配置好的建造者实例
        """
        # 提取建造者参数
        torch_dtype = kwargs.get("torch_dtype", None)
        device = kwargs.get("device", None)
        
        # 创建建造者实例
        builder = cls(
            base_repo=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # 加载完整的管道以获取所有组件
        # 这里我们使用 DiffusionPipeline.from_pretrained 来加载
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )
            
            # 提取所有组件
            expected_modules, optional_kwargs = DiffusionPipeline._get_signature_keys(pipeline)
            for name in expected_modules:
                component = getattr(pipeline, name, None)
                if component is not None:
                    builder.components[name] = component
            
            # 保存配置
            if hasattr(pipeline, "config"):
                for key, value in pipeline.config.items():
                    if key not in ["_class_name", "_diffusers_version", "_module", "_name_or_path"]:
                        builder.config_overrides[key] = value
                        
        except Exception as e:
            logger.warning(
                f"无法从 {pretrained_model_name_or_path} 加载完整管道: {e}. "
                "将尝试单独加载组件。"
            )
            # 如果加载失败，可以尝试单独加载各个组件
            # 这里为简化暂不实现
        
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
        设置或替换指定名称的组件（通用方法）
        
        Args:
            name: 组件名称 (如 'unet', 'vae', 'scheduler' 等)
            component: 组件对象
            freeze: 是否冻结组件参数
            requires_grad: 是否需要梯度
            **flags: 其他标志
        
        Returns:
            self，支持链式调用
        """
        # 基本类型检查
        if component is not None:
            if isinstance(component, torch.nn.Module):
                if not isinstance(component, (ModelMixin, torch.nn.Module)):
                    logger.warning(
                        f"组件 {name} 不是 ModelMixin 或 torch.nn.Module 的实例，"
                        "可能无法正常工作。"
                    )
            elif name == "scheduler" and not isinstance(component, SchedulerMixin):
                logger.warning(
                    f"调度器组件应该是 SchedulerMixin 的实例，"
                    f"但得到了 {type(component)}"
                )
        
        self.components[name] = component
        
        # 保存标志
        self.component_flags[name] = {
            "freeze": freeze,
            "requires_grad": requires_grad,
            **flags
        }
        
        # 应用freeze和requires_grad
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
        """设置UNet组件"""
        return self.with_component("unet", unet, **flags)
    
    def with_vae(
        self,
        vae: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置VAE组件"""
        return self.with_component("vae", vae, **flags)
    
    def with_scheduler(
        self,
        scheduler: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置调度器组件"""
        return self.with_component("scheduler", scheduler, **flags)
    
    def with_text_encoder(
        self,
        text_encoder: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置文本编码器组件"""
        return self.with_component("text_encoder", text_encoder, **flags)
    
    def with_tokenizer(
        self,
        tokenizer: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置分词器组件"""
        return self.with_component("tokenizer", tokenizer, **flags)
    
    def with_feature_extractor(
        self,
        feature_extractor: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置特征提取器组件"""
        return self.with_component("feature_extractor", feature_extractor, **flags)
    
    def with_safety_checker(
        self,
        safety_checker: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置安全检查器组件"""
        return self.with_component("safety_checker", safety_checker, **flags)
    
    def with_image_encoder(
        self,
        image_encoder: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置图像编码器组件"""
        return self.with_component("image_encoder", image_encoder, **flags)
    
    def with_controlnet(
        self,
        controlnet: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置ControlNet组件"""
        return self.with_component("controlnet", controlnet, **flags)
    
    def with_adapter(
        self,
        adapter: Any,
        **flags
    ) -> "DiffusionPipelineBuilder":
        """设置适配器组件"""
        return self.with_component("adapter", adapter, **flags)
    
    def apply_preset(
        self,
        name: str,
        **kwargs
    ) -> "DiffusionPipelineBuilder":
        """
        应用预设配置
        
        Args:
            name: 预设名称
            **kwargs: 传递给预设函数的额外参数
        
        Returns:
            self，支持链式调用
        """
        if name not in self.presets:
            raise ValueError(
                f"未知的预设名称: {name}. "
                f"可用的预设: {list(self.presets.keys())}"
            )
        
        preset_func = self.presets[name]
        preset_func(self, **kwargs)
        
        return self
    
    def with_config_override(
        self,
        **kwargs
    ) -> "DiffusionPipelineBuilder":
        """
        设置管道级别的配置覆盖
        
        Args:
            **kwargs: 配置参数
        
        Returns:
            self，支持链式调用
        """
        self.config_overrides.update(kwargs)
        return self
    
    def register_validator(
        self,
        validator: Callable[["DiffusionPipelineBuilder"], None]
    ) -> "DiffusionPipelineBuilder":
        """
        注册自定义验证器
        
        Args:
            validator: 验证函数，接收builder作为参数，
                      如果验证失败应抛出 PipelineValidationError
        
        Returns:
            self，支持链式调用
        """
        self.validators.append(validator)
        return self
    
    def add_hook(
        self,
        stage: str,
        hook: Callable
    ) -> "DiffusionPipelineBuilder":
        """
        添加钩子函数
        
        Args:
            stage: 钩子阶段 ("pre_build" 或 "post_build")
            hook: 钩子函数
        
        Returns:
            self，支持链式调用
        """
        if stage not in self.hooks:
            raise ValueError(
                f"无效的钩子阶段: {stage}. "
                f"可用的阶段: {list(self.hooks.keys())}"
            )
        
        self.hooks[stage].append(hook)
        return self
    
    def validate(self) -> None:
        """
        执行所有验证器
        
        Raises:
            PipelineValidationError: 如果验证失败
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
        构建管道
        
        Args:
            pipeline_cls: 管道类，默认为 DiffusionPipeline
            export_modules: 如果为True，返回组件字典而不是管道实例
            lazy: 如果为True，返回 (pipeline_cls, components) 元组供延迟构建
        
        Returns:
            根据参数返回管道实例、组件字典或延迟构建元组
        
        Raises:
            PipelineValidationError: 如果验证失败
        """
        # 执行 pre_build 钩子
        for hook in self.hooks["pre_build"]:
            hook(self)
        
        # 执行验证
        self.validate()
        
        # 如果只是导出模块
        if export_modules:
            return self.components.copy()
        
        # 如果是延迟构建
        if lazy:
            return (pipeline_cls, self.components.copy())
        
        # 检查必需的组件（基于pipeline_cls的__init__签名）
        init_signature = inspect.signature(pipeline_cls.__init__)
        required_params = []
        optional_params = []
        
        for param_name, param in init_signature.parameters.items():
            if param_name in ["self", "kwargs"]:
                continue
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                optional_params.append(param_name)
        
        # 检查缺失的必需组件
        missing_required = [p for p in required_params if p not in self.components]
        if missing_required:
            raise PipelineValidationError(
                f"缺失必需的组件: {missing_required}. "
                f"请使用 with_{missing_required[0]}() 等方法设置这些组件。"
            )
        
        # 构建管道
        try:
            # 准备构造参数
            init_kwargs = {}
            for param_name in list(init_signature.parameters.keys())[1:]:  # 跳过self
                if param_name in self.components:
                    init_kwargs[param_name] = self.components[param_name]
            
            # 创建管道实例
            pipeline = pipeline_cls(**init_kwargs)
            
            # 应用配置覆盖
            if hasattr(pipeline, "register_to_config") and self.config_overrides:
                pipeline.register_to_config(**self.config_overrides)
            
            # 移动到指定设备和数据类型
            if self.device is not None or self.torch_dtype is not None:
                device = self.device
                dtype = self.torch_dtype
                
                # 如果只指定了dtype但没有指定device，使用当前设备
                if dtype is not None and device is None:
                    device = pipeline.device
                
                # 移动管道
                if device is not None or dtype is not None:
                    pipeline.to(device=device, dtype=dtype)
            
            # 执行 post_build 钩子
            for hook in self.hooks["post_build"]:
                hook(pipeline)
            
            return pipeline
            
        except Exception as e:
            raise PipelineValidationError(
                f"构建管道时出错: {e}"
            ) from e
    
    def clone(self, **overrides) -> "DiffusionPipelineBuilder":
        """
        克隆当前建造者并可选地覆盖某些组件
        
        Args:
            **overrides: 要覆盖的组件，格式为 component_name=component
        
        Returns:
            新的建造者实例
        """
        # 创建新实例
        new_builder = DiffusionPipelineBuilder(
            base_repo=self.base_repo,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        # 复制组件和配置
        new_builder.components = copy.copy(self.components)
        new_builder.config_overrides = copy.copy(self.config_overrides)
        new_builder.component_flags = copy.deepcopy(self.component_flags)
        new_builder.validators = self.validators.copy()
        new_builder.hooks = {k: v.copy() for k, v in self.hooks.items()}
        new_builder.presets = self.presets.copy()
        
        # 应用覆盖
        for name, component in overrides.items():
            new_builder.with_component(name, component)
        
        return new_builder
