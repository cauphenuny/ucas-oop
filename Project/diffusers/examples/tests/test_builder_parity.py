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
测试 DiffusionPipelineBuilder 的功能等价性

这个脚本验证使用 Builder 构建的 pipeline 与传统方式构建的 pipeline
在功能上是等价的，确保 Builder 模式不会引入任何回归。

测试内容：
1. 组件一致性：确保 builder 构建的 pipeline 具有相同的组件
2. 配置一致性：确保配置参数相同
3. 类型一致性：确保组件类型相同
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.pipelines import DiffusionPipelineBuilder


def compare_pipelines(pipeline1, pipeline2, component_names):
    """
    比较两个 pipeline 的组件是否相同
    
    Args:
        pipeline1: 第一个 pipeline
        pipeline2: 第二个 pipeline
        component_names: 要比较的组件名称列表
    
    Returns:
        是否相同
    """
    differences = []

    for name in component_names:
        comp1 = getattr(pipeline1, name, None)
        comp2 = getattr(pipeline2, name, None)

        # 检查是否都存在或都不存在
        if (comp1 is None) != (comp2 is None):
            differences.append(f"组件 {name}: 一个为 None，另一个不是")
            continue

        # 如果都为 None，跳过
        if comp1 is None and comp2 is None:
            continue

        assert comp1 is not None and comp2 is not None

        # 检查类型
        if type(comp1) != type(comp2):
            differences.append(
                f"组件 {name}: 类型不同 - {type(comp1).__name__} vs {type(comp2).__name__}"
            )
            continue

        # 对于 torch.nn.Module，检查是否是同一个对象或有相同的状态
        if isinstance(comp1, torch.nn.Module):
            # 检查配置（如果有）
            if hasattr(comp1, "config") and hasattr(comp2, "config"):
                if comp1.config != comp2.config:
                    differences.append(f"组件 {name}: 配置不同")

        print(f"✓ 组件 {name}: 类型匹配 ({type(comp1).__name__})")

    return differences


def test_basic_builder():
    """测试基本的 builder 功能"""
    print("\n" + "="*60)
    print("测试 1: 基本 Builder 功能")
    print("="*60)
    
    # 使用一个简单的模型路径进行测试
    # 注意：在实际CI环境中，这里应该使用一个小型的测试模型
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    
    try:
        print(f"\n正在从 {model_id} 加载传统 pipeline...")
        traditional_pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        print("✓ 传统 pipeline 加载成功")
        
        print(f"\n正在使用 Builder 从 {model_id} 加载...")
        builder = DiffusionPipelineBuilder.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        builder_pipeline = builder.build(pipeline_cls=StableDiffusionPipeline)
        print("✓ Builder pipeline 构建成功")
        
        # 比较组件
        print("\n比较 pipeline 组件:")
        component_names = ["unet", "vae", "text_encoder", "tokenizer", "scheduler"]
        differences = compare_pipelines(traditional_pipeline, builder_pipeline, component_names)
        
        if differences:
            print("\n❌ 发现差异:")
            for diff in differences:
                print(f"  - {diff}")
            return False
        else:
            print("\n✓ 所有组件匹配！")
            return True
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_override():
    """测试组件覆盖功能"""
    print("\n" + "="*60)
    print("测试 2: 组件覆盖功能")
    print("="*60)
    
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    
    try:
        print(f"\n正在使用 Builder 从 {model_id} 加载...")
        builder = DiffusionPipelineBuilder.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        
        # 保存原始 scheduler
        original_scheduler = builder.components.get("scheduler")
        print(f"原始 scheduler: {type(original_scheduler).__name__}")
        
        # 测试替换 scheduler
        from diffusers import DDIMScheduler
        new_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        builder.with_scheduler(new_scheduler)
        print(f"新 scheduler: {type(builder.components['scheduler']).__name__}")
        
        # 构建 pipeline
        pipeline = builder.build(pipeline_cls=StableDiffusionPipeline)
        
        # 验证 scheduler 已更新
        if isinstance(pipeline.scheduler, DDIMScheduler):
            print("✓ Scheduler 成功替换为 DDIMScheduler")
            return True
        else:
            print(f"❌ Scheduler 替换失败: 期望 DDIMScheduler, 得到 {type(pipeline.scheduler).__name__}")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export_modules():
    """测试导出组件功能（用于训练脚本）"""
    print("\n" + "="*60)
    print("测试 3: 导出组件功能")
    print("="*60)
    
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    
    try:
        print(f"\n正在使用 Builder 从 {model_id} 加载...")
        builder = DiffusionPipelineBuilder.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        
        # 导出组件
        components = builder.build(export_modules=True)
        print(f"\n导出的组件: {list(components.keys())}")
        
        # 验证导出的是字典
        if not isinstance(components, dict):
            print(f"❌ 导出结果应该是字典，得到 {type(components)}")
            return False
        
        # 验证包含必要的组件
        required_components = ["unet", "vae", "scheduler"]
        missing = [c for c in required_components if c not in components]
        if missing:
            print(f"❌ 缺失必要组件: {missing}")
            return False
        
        print("✓ 组件导出功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_freeze_components():
    """测试组件冻结功能"""
    print("\n" + "="*60)
    print("测试 4: 组件冻结功能")
    print("="*60)
    
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    
    try:
        print(f"\n正在使用 Builder 从 {model_id} 加载...")
        builder = DiffusionPipelineBuilder.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        
        # 获取 VAE 并冻结
        vae = builder.components["vae"]
        builder.with_vae(vae, freeze=True)
        
        # 检查是否冻结
        frozen = all(not p.requires_grad for p in vae.parameters())
        
        if frozen:
            print("✓ VAE 参数成功冻结")
            return True
        else:
            print("❌ VAE 参数冻结失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 DiffusionPipelineBuilder 功能")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只运行快速测试"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("DiffusionPipelineBuilder 功能测试")
    print("="*60)
    
    # 运行测试
    results = []
    
    results.append(("基本功能", test_basic_builder()))
    results.append(("组件覆盖", test_component_override()))
    results.append(("导出组件", test_export_modules()))
    
    if not args.quick:
        results.append(("组件冻结", test_freeze_components()))
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
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
