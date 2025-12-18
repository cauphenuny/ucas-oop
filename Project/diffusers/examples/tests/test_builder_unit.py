#!/usr/bin/env python3
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
简单的 Builder 单元测试（不需要网络）

这个脚本测试 DiffusionPipelineBuilder 的基本功能，
不需要从 Hub 下载模型。
"""

import sys
import torch
from diffusers.pipelines import DiffusionPipelineBuilder, PipelineValidationError


def test_builder_instantiation():
    """测试 Builder 实例化"""
    print("\n测试 1: Builder 实例化")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder(
            torch_dtype=torch.float32,
            device="cpu"
        )
        print(f"✓ Builder 创建成功")
        print(f"  - torch_dtype: {builder.torch_dtype}")
        print(f"  - device: {builder.device}")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_component_setting():
    """测试组件设置"""
    print("\n测试 2: 组件设置")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder()
        
        # 创建一个简单的 mock 组件
        mock_component = torch.nn.Linear(10, 10)
        
        # 使用 with_component 方法
        builder.with_component("test_component", mock_component)
        
        if "test_component" in builder.components:
            print(f"✓ 组件设置成功")
            print(f"  - 组件名称: test_component")
            print(f"  - 组件类型: {type(mock_component).__name__}")
            return True
        else:
            print(f"❌ 组件未正确设置")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_freeze_functionality():
    """测试冻结功能"""
    print("\n测试 3: 组件冻结功能")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder()
        
        # 创建一个简单的 mock 组件
        mock_component = torch.nn.Linear(10, 10)
        
        # 确保初始状态需要梯度
        mock_component.requires_grad_(True)
        initial_requires_grad = any(p.requires_grad for p in mock_component.parameters())
        print(f"  初始状态 requires_grad: {initial_requires_grad}")
        
        # 使用 freeze 标志设置组件
        builder.with_component("frozen_component", mock_component, freeze=True)
        
        # 检查是否冻结
        is_frozen = all(not p.requires_grad for p in mock_component.parameters())
        
        if is_frozen:
            print(f"✓ 组件成功冻结")
            print(f"  - 所有参数 requires_grad=False")
            return True
        else:
            print(f"❌ 组件未成功冻结")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chain_methods():
    """测试链式调用"""
    print("\n测试 4: 链式调用")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder()
        
        # 创建 mock 组件
        unet = torch.nn.Linear(10, 10)
        vae = torch.nn.Linear(20, 20)
        
        # 链式调用
        result = builder.with_unet(unet).with_vae(vae)
        
        # 验证返回的是同一个 builder
        if result is builder:
            print(f"✓ 链式调用成功")
            print(f"  - with_unet 返回 builder")
            print(f"  - with_vae 返回 builder")
            print(f"  - 已设置组件: {list(builder.components.keys())}")
            return True
        else:
            print(f"❌ 链式调用失败")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_override():
    """测试配置覆盖"""
    print("\n测试 5: 配置覆盖")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder()
        
        # 设置配置
        builder.with_config_override(
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        if "guidance_scale" in builder.config_overrides and "num_inference_steps" in builder.config_overrides:
            print(f"✓ 配置覆盖成功")
            print(f"  - guidance_scale: {builder.config_overrides['guidance_scale']}")
            print(f"  - num_inference_steps: {builder.config_overrides['num_inference_steps']}")
            return True
        else:
            print(f"❌ 配置未正确设置")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_export_modules():
    """测试导出模块功能"""
    print("\n测试 6: 导出模块")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder()
        
        # 添加一些组件
        builder.with_component("unet", torch.nn.Linear(10, 10))
        builder.with_component("vae", torch.nn.Linear(20, 20))
        
        # 导出模块
        components = builder.build(export_modules=True)
        
        if isinstance(components, dict) and "unet" in components and "vae" in components:
            print(f"✓ 模块导出成功")
            print(f"  - 导出类型: {type(components)}")
            print(f"  - 导出组件: {list(components.keys())}")
            return True
        else:
            print(f"❌ 模块导出失败")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clone():
    """测试克隆功能"""
    print("\n测试 7: 克隆 Builder")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder(torch_dtype=torch.float32)
        builder.with_component("test", torch.nn.Linear(10, 10))
        builder.with_config_override(key="value")
        
        # 克隆
        cloned = builder.clone()
        
        # 验证克隆
        if (cloned is not builder and 
            cloned.torch_dtype == builder.torch_dtype and
            "test" in cloned.components and
            "key" in cloned.config_overrides):
            print(f"✓ Builder 克隆成功")
            print(f"  - 新 builder 实例: {cloned is not builder}")
            print(f"  - 组件已复制: {'test' in cloned.components}")
            print(f"  - 配置已复制: {'key' in cloned.config_overrides}")
            return True
        else:
            print(f"❌ Builder 克隆失败")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validator():
    """测试验证器功能"""
    print("\n测试 8: 验证器")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder()
        
        # 添加一个简单的验证器
        validation_called = [False]  # 使用列表来避免闭包问题
        
        def custom_validator(b):
            validation_called[0] = True
            if "required_component" not in b.components:
                raise PipelineValidationError("缺失 required_component")
        
        builder.register_validator(custom_validator)
        
        # 尝试验证（应该失败）
        try:
            builder.validate()
            print(f"❌ 验证器未正确触发异常")
            return False
        except PipelineValidationError as e:
            if validation_called[0]:
                print(f"✓ 验证器成功触发")
                print(f"  - 错误信息: {e}")
                return True
            else:
                print(f"❌ 验证器未被调用")
                return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hooks():
    """测试钩子功能"""
    print("\n测试 9: 钩子系统")
    print("-" * 50)
    
    try:
        builder = DiffusionPipelineBuilder()
        
        # 添加钩子
        pre_build_called = [False]
        
        def pre_build_hook(b):
            pre_build_called[0] = True
        
        builder.add_hook("pre_build", pre_build_hook)
        
        # 添加必要组件以避免构建失败
        builder.with_component("dummy", torch.nn.Linear(1, 1))
        
        # 尝试导出模块（会触发钩子）
        try:
            builder.build(export_modules=True)
            
            if pre_build_called[0]:
                print(f"✓ 钩子系统正常工作")
                print(f"  - pre_build 钩子被调用")
                return True
            else:
                print(f"❌ 钩子未被调用")
                return False
        except Exception as e:
            # 即使构建失败，钩子也应该被调用
            if pre_build_called[0]:
                print(f"✓ 钩子在构建前被调用（即使构建失败）")
                return True
            else:
                print(f"❌ 钩子未被调用")
                return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("DiffusionPipelineBuilder 单元测试")
    print("=" * 60)
    
    # 运行测试
    results = []
    
    results.append(("Builder 实例化", test_builder_instantiation()))
    results.append(("组件设置", test_component_setting()))
    results.append(("组件冻结", test_freeze_functionality()))
    results.append(("链式调用", test_chain_methods()))
    results.append(("配置覆盖", test_config_override()))
    results.append(("导出模块", test_export_modules()))
    results.append(("克隆功能", test_clone()))
    results.append(("验证器", test_validator()))
    results.append(("钩子系统", test_hooks()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    # 返回退出码
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
