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
Builder 模式应用示例：统一训练脚本的 Pipeline 构造流程

这个脚本演示了如何使用 DiffusionPipelineBuilder 来统一
train_custom_diffusion.py 和 train_text_to_image.py 中的
pipeline 构造代码，避免重复和不一致。

对比传统方式和 Builder 方式：

传统方式 (train_text_to_image.py):
```python
# 加载调度器
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

# 加载分词器
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)

# 加载文本编码器
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
)

# 加载 VAE
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
)

# 加载 UNet
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)

# 冻结组件
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
```

Builder 方式:
```python
# 一次性加载所有组件并配置
builder = DiffusionPipelineBuilder.from_pretrained(
    args.pretrained_model_name_or_path,
    revision=args.revision,
    variant=args.variant,
    torch_dtype=torch.float16 if args.mixed_precision == "fp16" else None,
)

# 替换或配置特定组件
if args.non_ema_revision:
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    builder.with_unet(unet)

# 设置冻结
builder.with_vae(builder.components["vae"], freeze=True)
builder.with_text_encoder(builder.components["text_encoder"], freeze=True)

# 导出组件用于训练
components = builder.build(export_modules=True)
unet = components["unet"]
vae = components["vae"]
text_encoder = components["text_encoder"]
tokenizer = components["tokenizer"]
noise_scheduler = components["scheduler"]
```
"""

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.pipelines import DiffusionPipelineBuilder


def traditional_construction(model_id, revision=None, variant=None):
    """
    传统方式构造 pipeline 组件
    （模拟 train_text_to_image.py 的做法）
    """
    print("\n传统方式构造:")
    print("-" * 50)
    
    # 逐个加载组件
    print("加载 scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    print("加载 tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision
    )
    
    print("加载 text_encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=revision, variant=variant
    )
    
    print("加载 vae...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", revision=revision, variant=variant
    )
    
    print("加载 unet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", revision=revision
    )
    
    # 冻结组件
    print("冻结 vae 和 text_encoder...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    print("✓ 传统方式构造完成")
    
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": noise_scheduler,
    }


def builder_construction(model_id, revision=None, variant=None):
    """
    使用 Builder 方式构造 pipeline 组件
    （展示新的统一做法）
    """
    print("\nBuilder 方式构造:")
    print("-" * 50)
    
    # 一次性加载所有组件
    print(f"从 {model_id} 加载所有组件...")
    builder = DiffusionPipelineBuilder.from_pretrained(
        model_id,
        revision=revision,
        variant=variant,
        safety_checker=None,  # 训练时通常不需要
    )
    
    # 配置组件（冻结 VAE 和 text_encoder）
    print("配置组件（冻结 vae 和 text_encoder）...")
    vae = builder.components["vae"]
    text_encoder = builder.components["text_encoder"]
    
    builder.with_vae(vae, freeze=True)
    builder.with_text_encoder(text_encoder, freeze=True)
    
    # 导出组件字典用于训练
    print("导出组件...")
    components = builder.build(export_modules=True)
    
    print("✓ Builder 方式构造完成")
    
    return components


def compare_components(traditional, builder_based):
    """比较两种方式构造的组件"""
    print("\n比较结果:")
    print("=" * 50)
    
    all_match = True
    
    for key in traditional.keys():
        trad_comp = traditional[key]
        build_comp = builder_based.get(key)
        
        if build_comp is None:
            print(f"❌ {key}: Builder 方式缺失此组件")
            all_match = False
            continue
        
        # 比较类型
        if type(trad_comp) != type(build_comp):
            print(f"❌ {key}: 类型不匹配 - {type(trad_comp).__name__} vs {type(build_comp).__name__}")
            all_match = False
            continue
        
        # 检查冻结状态
        if isinstance(trad_comp, torch.nn.Module):
            trad_frozen = all(not p.requires_grad for p in trad_comp.parameters())
            build_frozen = all(not p.requires_grad for p in build_comp.parameters())
            
            freeze_status = "冻结" if build_frozen else "未冻结"
            if trad_frozen == build_frozen:
                print(f"✓ {key}: 类型匹配 ({type(trad_comp).__name__}), {freeze_status}")
            else:
                print(f"❌ {key}: 冻结状态不匹配")
                all_match = False
        else:
            print(f"✓ {key}: 类型匹配 ({type(trad_comp).__name__})")
    
    if all_match:
        print("\n✓ 两种方式构造的组件完全一致！")
    else:
        print("\n❌ 发现差异")
    
    return all_match


def demonstrate_builder_advantages():
    """演示 Builder 的优势"""
    print("\n" + "=" * 70)
    print("Builder 模式的优势")
    print("=" * 70)
    
    advantages = [
        "1. 代码复用：避免在不同训练脚本中重复相同的加载逻辑",
        "2. 一致性：确保所有脚本使用相同的组件配置方式",
        "3. 灵活性：轻松替换单个组件而无需重写整个加载流程",
        "4. 可读性：链式调用更清晰地表达意图",
        "5. 可维护性：配置集中管理，易于更新和调试",
        "6. 预设支持：可以定义和应用常用配置预设（如 SDXL、Flux）",
        "7. 验证：内置验证机制确保组件兼容性",
        "8. 扩展性：通过钩子和验证器轻松扩展功能",
    ]
    
    for advantage in advantages:
        print(f"  {advantage}")
    
    print("\n代码对比:")
    print("-" * 70)
    print("传统方式需要:")
    print("  - 6+ 行代码分别加载各个组件")
    print("  - 2+ 行代码单独设置冻结状态")
    print("  - 在每个训练脚本中重复这些代码")
    print("\nBuilder 方式只需:")
    print("  - 1 行加载所有组件")
    print("  - 2 行配置冻结状态")
    print("  - 1 行导出组件")
    print("  - 可在多个脚本间共享相同逻辑")


def main():
    """主函数"""
    print("=" * 70)
    print("Builder 模式示例：统一训练脚本的 Pipeline 构造")
    print("=" * 70)
    
    # 使用测试模型
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    
    try:
        # 传统方式
        traditional_components = traditional_construction(model_id)
        
        # Builder 方式
        builder_components = builder_construction(model_id)
        
        # 比较结果
        compare_components(traditional_components, builder_components)
        
        # 展示优势
        demonstrate_builder_advantages()
        
        print("\n" + "=" * 70)
        print("示例运行成功！")
        print("=" * 70)
        
        # 展示如何在实际训练脚本中使用
        print("\n在实际训练脚本中的使用示例:")
        print("-" * 70)
        print("""
# 在 train_text_to_image.py 中:
builder = DiffusionPipelineBuilder.from_pretrained(
    args.pretrained_model_name_or_path,
    revision=args.revision,
    variant=args.variant,
    safety_checker=None,
)

# 配置训练需求
builder.with_vae(builder.components["vae"], freeze=True)
builder.with_text_encoder(builder.components["text_encoder"], freeze=True)

# 导出用于训练
components = builder.build(export_modules=True)
unet = components["unet"]
vae = components["vae"]
text_encoder = components["text_encoder"]
tokenizer = components["tokenizer"]
noise_scheduler = components["scheduler"]

# 其余训练代码保持不变...
        """)
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
