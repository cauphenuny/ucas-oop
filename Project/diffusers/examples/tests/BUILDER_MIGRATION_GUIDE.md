# Builder 模式在训练脚本中的应用示例

## 概述

本文档展示了如何使用 `DiffusionPipelineBuilder` 来统一 `train_custom_diffusion.py` 和 `train_text_to_image.py` 中的 pipeline 构造流程。

## 问题分析

### 当前问题

在现有的训练脚本中，存在以下问题：

1. **代码重复**：每个训练脚本都重复实现相同的组件加载逻辑
2. **不一致性**：不同脚本可能使用略有不同的参数和配置
3. **维护困难**：需要在多个地方修改相同的逻辑
4. **易出错**：手动管理多个组件容易遗漏配置

### train_custom_diffusion.py 中的构造代码

该脚本在两个地方构造 pipeline：

**位置 1：生成先验图像（第 762-796 行）**
```python
torch_dtype = torch.float32
if args.prior_generation_precision == "fp32":
    torch_dtype = torch.float32
elif args.prior_generation_precision == "fp16":
    torch_dtype = torch.float16
elif args.prior_generation_precision == "bf16":
    torch_dtype = torch.bfloat16

pipeline = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch_dtype,
    safety_checker=None,
    revision=args.revision,
    variant=args.variant,
)
pipeline.set_progress_bar_config(disable=True)
# ... 使用 pipeline 生成图像
del pipeline  # 用完即删
```

**位置 2：训练准备（第 809-836 行）**
```python
# 加载 tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, ...)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, ...)

# 导入正确的 text encoder 类
text_encoder_cls = import_model_class_from_model_name_or_path(...)

# 加载组件
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", ...)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", ...)
unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", ...)
```

### train_text_to_image.py 中的构造代码

**训练准备（第 592-626 行）**
```python
# 加载 scheduler
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

# 加载 tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)

# 加载模型
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
)
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)

# 冻结 vae 和 text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
```

## Builder 方案

### 统一的构造流程

使用 Builder 可以将上述代码统一为：

```python
from diffusers.pipelines import DiffusionPipelineBuilder

# 构建 builder
builder = DiffusionPipelineBuilder.from_pretrained(
    args.pretrained_model_name_or_path,
    revision=args.revision,
    variant=args.variant,
    safety_checker=None,  # 训练时不需要
    torch_dtype=get_torch_dtype(args),  # 辅助函数
)

# 配置训练需求
builder.with_vae(builder.components["vae"], freeze=True)
builder.with_text_encoder(builder.components["text_encoder"], freeze=True)

# 如果需要替换 unet（如使用 non_ema_revision）
if args.non_ema_revision:
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision
    )
    builder.with_unet(unet)

# 导出组件用于训练
components = builder.build(export_modules=True)
unet = components["unet"]
vae = components["vae"]
text_encoder = components["text_encoder"]
tokenizer = components["tokenizer"]
noise_scheduler = components["scheduler"]
```

### 用于推理/验证

在验证阶段需要完整 pipeline 时：

```python
# 使用相同的 builder 构建完整 pipeline
validation_pipeline = builder.build(pipeline_cls=StableDiffusionPipeline)

# 或者为验证创建新的 builder（如果需要不同配置）
validation_builder = DiffusionPipelineBuilder.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    safety_checker=None,
)
validation_pipeline = validation_builder.build()
```

### 用于生成先验图像（train_custom_diffusion.py）

```python
# 构建用于生成先验图像的 pipeline
prior_builder = DiffusionPipelineBuilder.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=get_torch_dtype(args.prior_generation_precision),
    safety_checker=None,
    revision=args.revision,
    variant=args.variant,
)

# 构建 pipeline
pipeline = prior_builder.build()
pipeline.set_progress_bar_config(disable=True)

# 使用 pipeline...
# 完成后 del pipeline
```

## 代码对比

### 代码量对比

| 场景 | 传统方式 | Builder 方式 | 减少行数 |
|------|---------|------------|---------|
| 基本组件加载 | 8-10 行 | 1 行 | 7-9 行 |
| 组件冻结 | 2 行 | 2 行（链式） | 0 行 |
| 生成验证 pipeline | 8 行 | 2 行 | 6 行 |
| **总计** | **18-20 行** | **5 行** | **13-15 行** |

### 代码可读性对比

**传统方式：**
```python
# 需要记住每个组件的加载方式
noise_scheduler = DDPMScheduler.from_pretrained(path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer", revision=rev)
text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", revision=rev, variant=var)
vae = AutoencoderKL.from_pretrained(path, subfolder="vae", revision=rev, variant=var)
unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", revision=rev)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
```

**Builder 方式：**
```python
# 清晰的构建流程
builder = DiffusionPipelineBuilder.from_pretrained(path, revision=rev, variant=var, safety_checker=None)
builder.with_vae(builder.components["vae"], freeze=True)
builder.with_text_encoder(builder.components["text_encoder"], freeze=True)
components = builder.build(export_modules=True)
```

## 优势总结

1. **代码复用**
   - 一次性加载所有组件，无需重复编写加载代码
   - 可以在多个训练脚本间共享 builder 配置

2. **一致性保证**
   - 所有脚本使用相同的组件加载逻辑
   - 参数配置集中管理，减少不一致

3. **易于维护**
   - 修改加载逻辑只需在一处修改
   - 新增组件类型时只需扩展 builder

4. **灵活性**
   - 支持部分组件替换
   - 支持不同的构建模式（完整 pipeline / 组件字典）
   - 支持克隆和实验对比

5. **可测试性**
   - Builder 本身可以单独测试
   - 易于模拟和注入测试组件

## 迁移指南

### 步骤 1：识别组件加载代码

在训练脚本中找到所有 `from_pretrained` 调用，特别是：
- `DDPMScheduler.from_pretrained`
- `CLIPTokenizer.from_pretrained` / `AutoTokenizer.from_pretrained`
- `CLIPTextModel.from_pretrained`
- `AutoencoderKL.from_pretrained`
- `UNet2DConditionModel.from_pretrained`
- `DiffusionPipeline.from_pretrained`

### 步骤 2：创建 Builder

将这些加载调用替换为一个 builder：

```python
from diffusers.pipelines import DiffusionPipelineBuilder

builder = DiffusionPipelineBuilder.from_pretrained(
    args.pretrained_model_name_or_path,
    # 传递所有相关参数
    revision=args.revision,
    variant=args.variant,
    safety_checker=None,
)
```

### 步骤 3：配置组件

应用需要的配置：

```python
# 冻结组件
builder.with_vae(builder.components["vae"], freeze=True)
builder.with_text_encoder(builder.components["text_encoder"], freeze=True)

# 替换组件（如果需要）
if args.use_custom_unet:
    builder.with_unet(load_custom_unet())
```

### 步骤 4：导出组件

根据使用场景导出：

```python
# 训练：导出组件字典
components = builder.build(export_modules=True)
unet = components["unet"]
vae = components["vae"]
# ...

# 推理/验证：构建完整 pipeline
pipeline = builder.build(pipeline_cls=StableDiffusionPipeline)
```

### 步骤 5：测试

运行训练脚本确保：
- 所有组件正确加载
- 冻结状态正确
- 训练/验证流程正常

## 示例：完整迁移

### 迁移前（train_text_to_image.py 片段）

```python
# Load scheduler, tokenizer and models.
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)

with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)

# Freeze vae and text_encoder and set unet to trainable
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()
```

### 迁移后

```python
from diffusers.pipelines import DiffusionPipelineBuilder

# Load all components using builder
builder = DiffusionPipelineBuilder.from_pretrained(
    args.pretrained_model_name_or_path,
    revision=args.revision,
    variant=args.variant,
    safety_checker=None,
)

# Handle special unet revision if needed
if args.non_ema_revision and args.non_ema_revision != args.revision:
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision
    )
    builder.with_unet(unet)

# Configure for training
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    builder.with_vae(builder.components["vae"], freeze=True)
    builder.with_text_encoder(builder.components["text_encoder"], freeze=True)

# Export components
components = builder.build(export_modules=True)
noise_scheduler = components["scheduler"]
tokenizer = components["tokenizer"]
text_encoder = components["text_encoder"]
vae = components["vae"]
unet = components["unet"]

# Set unet to trainable (others are already frozen)
unet.train()
```

## 总结

`DiffusionPipelineBuilder` 提供了一种统一、灵活、易维护的方式来构建扩散管道。通过使用 builder 模式，我们可以：

- 减少代码重复
- 提高代码一致性
- 简化维护工作
- 提升代码可读性
- 增强可测试性

这使得训练脚本更加简洁和健壮。
