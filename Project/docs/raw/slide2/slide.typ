#show emph: text.with(font: ("New Computer Modern", "STKaiti"))
#set text(font: ("Libertinus Serif", "Songti SC"), lang: "zh")
#show emph: text.with(font: ("Libertinus Serif", "STKaiti"))
#import "@preview/theorion:0.4.1"
#import "@preview/tablem:0.3.0": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *


#import "meta.typ": *
#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": *

#show: doc => {
  // import themes.university: *
  // import themes.metropolis: *
  import themes.dewdrop: *
  show: dewdrop-theme.with(
    aspect-ratio: "16-9",
    footer: self => grid(
      columns: (1fr, 1fr, 1fr),
      align: center + horizon,
      self.info.author, self.info.title, self.info.date.display(),
    ),
    navigation: "mini-slides",
    config-info(
      title: meta.slide-title,
      subtitle: meta.subtitle,
      date: meta.date,
      author: meta.author,
    ),
  )
  // show: university-theme.with(
  //   aspect-ratio: "16-9",
  //   footer: self => grid(
  //     columns: (1fr, 1fr, 1fr),
  //     align: center + horizon,
  //     self.info.author,
  //     self.info.title,
  //     self.info.date.display(),
  //     ),
  //   config-info(
  //     title: meta.slide-title,
  //     subtitle: meta.subtitle,
  //   )
  // )
  // show: metropolis-theme.with(
  //   aspect-ratio: "16-9",
  //   footer: self => grid(
  //     columns: (1fr, 1fr, 1fr),
  //     align: center + horizon,
  //     self.info.author,
  //     self.info.title,
  //     self.info.date.display(),
  //     ),
  //   config-info(
  //     title: meta.slide-title,
  //     subtitle: meta.subtitle,
  //     author: meta.author,
  //     date: meta.date,
  //     institution: meta.institution,
  //     logo: none,
  //   ),
  // )
  show: text.with(size: 0.90em)
  show: codly-init.with()
  show raw.where(block: true): text.with(size: 0.8em)

  set heading(numbering: numbly("{1:一}、", default: "1.1  "))

  title-slide()
  doc
  focus-slide[
    Thanks!
  ]
}

= 缺陷诊断

---

== 利用大模型发现问题

#grid(
  columns: (1fr, 4fr),
  align: horizon,
  [
    与 Deep-Wiki 多轮对话，\
    整理成重构文档
  ],
  [
    #figure(image("deepwiki.png", width: 60%), caption: "对话截图")
  ],
)

---

文档节选：

#let md-doc = read("attn-demo.md")

#[
  #show: text.with(size: 0.7em)
  #raw(md-doc, lang: "markdown")
]

---

= 重构采用的设计模式介绍

---

== 建造者模式 (Builder Pattern)

#grid(
  columns: (1fr, 1em, 1fr),
  align: horizon,
  [
    #theorion.note-box(title: "建造者模式")[
      将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示。
    ]

    === 引入

    假设有这样一个复杂对象， 在对其进行构造时需要构造众多成员变量和嵌套对象。 这些初始化代码通常深藏于一个包含众多参数的构造函数中，且散落在客户端代码的多个位置。
  ],
  [
  ],
  [
    #figure(image("image.png"), caption: "一个有复杂构造函数的 House 类")
  ],
)

---

#grid(
  columns: (1fr, 1em, 1fr),
  align: horizon,
  [
    *建造者模式的解决方案*

    将对象构造代码从产品类中抽取出来， 并将其放在一个名为 _建造者_ 的独立对象中。

    将对象构造过程划分为一组步骤， 比如 `build­Walls` 创建墙壁和 `build­Door` 创建房门等。 每次创建对象时， 都需要通过建造者对象执行一系列步骤。 重点在于无需调用所有步骤， 而只需调用创建特定对象配置所需的那些步骤即可。

  ],
  [],
  [
    #figure(image("image-1.png"), caption: "建造者：HouseBuilder")
  ],
)

---

=== 在代码中的应用

统一 `DiffusionPipeline` 各组件的构建过程，解决训练脚本中的代码重复和不一致问题。

核心类：`DiffusionPipelineBuilder`，提供链式配置和组件管理

`DiffusionPipelineBuilder` 提供一些方法：

- `from_pretrained()`, `add_component()`, `with_vae()`, `with_text_encoder()` 等用于灵活配置和构建不同的扩散管道。

- `build()` 方法根据配置组装并返回最终的 `DiffusionPipeline` 实例或者组件 `dict`。

---

== 策略模式 (Strategy Pattern)

=== 模式介绍

#grid(
  columns: (1fr, 1em, 1fr),
  align: horizon,
  [
    #theorion.note-box(title: "策略模式")[
      定义一系列算法， 将每个算法封装起来， 并使它们可以互换。 策略模式让算法独立于使用它的客户而变化。
    ]

    - 完成一项任务，往往可以有多种不同的方式，每一种方式称为一个策略，我们可以根据环境或者条件的不同选择不同的策略来完成该项任务。

  ],
  [],
  [
    #figure(image("image-2.png"), caption: "一些路径规划策略")
  ],
)
---

策略模式建议找出负责用许多不同方式完成特定任务的类， 然后将其中的算法抽取到一组被称为策略的独立类中。

名为上下文的原始类必须包含一个成员变量来存储对于每种策略的引用。 上下文并不执行任务， 而是将工作委派给已连接的策略对象。

上下文不负责选择符合任务需要的算法——客户端会将所需策略传递给上下文。 实际上， 上下文并不十分了解策略， 它会通过同样的通用接口与所有策略进行交互， 而该接口只需暴露一个方法来触发所选策略中封装的算法即可。

因此， 上下文可独立于具体策略。 这样你就可在不修改上下文代码或其他策略的情况下添加新算法或修改已有算法了。

---

=== 重构中的应用

==== 问题背景

Diffusers 库支持多种 attention 后端（如 FlashAttention、xFormers、PyTorch 原生等），用于优化不同硬件上的性能。但原始实现存在一些问题：

- 扩展困难：新增后端需修改多处代码（如枚举、注册、检查函数）。
- 维护复杂：函数式实现难以测试和调试。
- 类型不安全：缺乏抽象接口，易出错。

---

目前原有的实现是基于注册表模式管理后端

```python
@_AttentionBackendRegistry.register(AttentionBackendName.FLASH)
def _flash_attention(query, key, value, **kwargs):
    return flash_attn_func(q=query, k=key, v=value, **kwargs)
```

这个 `_AttentionBackendRegistry.register` 装饰器会在全局的注册表中将后端名称映射到对应的函数。

---

引入抽象策略接口，将函数式实现转换为类结构

- 抽象策略接口：`AttentionStrategy` 基类
- 具体策略类：`FlashAttentionStrategy`、`XFormersAttentionStrategy` 等，封装各自的实现细节
- 工厂模式：`AttentionStrategyFactory` 根据名称实例化对应策略类
- 约束检查：共同的检查移到基类

```python
class AttentionStrategy(ABC):
    @abstractmethod
    def compute_attention(self, query, key, value, **kwargs):
        pass

class FlashAttentionStrategy(AttentionStrategy):
    def compute_attention(self, query, key, value, **kwargs):
        return flash_attn_func(q=query, k=key, v=value, **kwargs)
```

---

= 重构过程以及效果

---

== 重构过程

=== 构造单元测试

测试驱动开发 (TDD) 思想，先编写测试用例，再进行重构

减少大模型重构过程中可能发生的错误

```python
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
            print(f"配置未正确设置")
            return False
    except Exception as e:
        print(f"测试失败: {e}")
        return False
```

---

#grid(
  columns: (2fr, 4fr),
  [
    === 大模型辅助重构

    整理设计文档，结合代码库当作上下文
  ],
  [
    #figure(image("image-3.png", width: 80%), caption: "Coding Agent")
  ],
)

---

== 效果展示

=== Builder

#figure(image("image-7.png", width: 30%), caption: "Builder 类图")

---

#grid(
  columns: (1fr, 1fr),
  [
    传统方式 (train_text_to_image.py)

    ```python
    # 需要 8+ 行重复代码
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # 手动冻结组件
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ```
  ],
  [
    Builder 方式

    ```python
    # 只需 4 行代码
    builder = DiffusionPipelineBuilder.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )

    # 链式配置和冻结
    builder.with_vae(builder.components["vae"], freeze=True)
    builder.with_text_encoder(builder.components["text_encoder"], freeze=True)

    # 构造时传递参数
    builder.with_scheduler(DDIMScheduler, num_train_timesteps=1000)

    pipe = builder.build()
    ```
  ],
)

=== Attention Strategy

#figure(image("image-8.png", width: 70%), caption: "Attention Strategy 类图")

#figure(image("image-10.png", width: 70%), caption: "执行过程")

---

使用示例

```python
# 内部使用策略模式
# 自动选择合适的 attention 后端
from diffusers.models.attention_dispatch import dispatch_attention_fn

# 根据硬件和配置自动选择策略
output = dispatch_attention_fn(
    backend="FLASH",  # 或 "XFORMERS", "NATIVE"
    query=query, key=key, value=value
)
```

---

= 扩展：PeRFlow 实现

---

== PeRFlow 简介

#grid(
  columns: (1fr, 1em, 1fr),
  align: horizon,
  [
    *PeRFlow (Piecewise Rectified Flow)*

    - 分段线性流加速调度器
    - 减少扩散模型采样步数
    - 在保持质量的同时提升生成速度
    
    === 核心思想
    
    将时间域划分为 K 个窗口（默认 4 个），在每个窗口内使用线性流近似，从而加速去噪过程。
  ],
  [],
  [
    #theorion.note-box(title: "主要优势")[
      - 更少的采样步数（5-10步 vs 50步）
      - 保持生成质量
      - 兼容现有 Diffusion Pipeline
      - 支持 Stable Diffusion 和 SDXL
    ]
  ],
)

---

== 框架设计

=== 总体架构

框架包含三个核心组件：

1. *PeRFlowScheduler*: 主调度器类，实现分段线性流
2. *PFODESolver*: ODE求解器，用于Stable Diffusion模型
3. *PFODESolverSDXL*: SDXL专用ODE求解器

所有组件继承自 `SchedulerMixin` 和 `ConfigMixin`，确保与 diffusers 库的兼容性。

---

=== 时间窗口管理

```python
class TimeWindows:
    """管理分段时间窗口"""
    def __init__(self, t_initial=1, t_terminal=0, num_windows=4):
        # 将时间域划分为 K 个窗口
        # 例如：[1.0, 0.75], [0.75, 0.5], [0.5, 0.25], [0.25, 0]
        time_windows = [1.*i/num_windows for i in range(1, num_windows+1)][::-1]
        self.window_starts = time_windows
        self.window_ends = time_windows[1:] + [t_terminal]
    
    def get_window(self, tp: float) -> Tuple[float, float]:
        """获取时间点所在的窗口"""
        # 返回 (window_start, window_end)
        pass
    
    def lookup_window(self, timepoint: torch.FloatTensor):
        """批量查找时间窗口"""
        # 支持批处理
        pass
```

---

== PeRFlowScheduler 实现

=== 核心方法

```python
class PeRFlowScheduler(SchedulerMixin, ConfigMixin):
    def __init__(self, num_train_timesteps=1000, num_windows=4, 
                 beta_schedule="scaled_linear", ...):
        """初始化调度器"""
        # 设置时间窗口
        self.time_windows = TimeWindows(num_windows=num_windows)
        
        # 计算 beta 调度
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas)
        else:
            self.betas = betas_for_alpha_bar(num_train_timesteps, ...)
        
        # 计算 alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
```

---

```python
    def set_timesteps(self, num_inference_steps: int, device=None):
        """生成推理时间步"""
        # 在各窗口间分配时间步
        # 确保覆盖所有时间窗口
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, 
            num_inference_steps
        )
        self.timesteps = self.timesteps.round().long().to(device)
    
    def step(self, model_output, timestep, sample, **kwargs):
        """执行单步去噪"""
        # 1. 获取当前时间窗口
        window_start, window_end = self.time_windows.get_window(timestep)
        
        # 2. 计算窗口的 alpha 值
        alpha = self.get_window_alpha(window_start, window_end)
        
        # 3. 根据预测类型计算前一样本
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        # ... 其他预测类型
        
        return PeRFlowSchedulerOutput(prev_sample=prev_sample)
```

---

== ODE 求解器

=== Stable Diffusion 求解器

```python
class PFODESolver:
    """SD模型的ODE求解器"""
    def __init__(self, scheduler, t_initial=1.0, t_terminal=0.0):
        self.scheduler = scheduler
        self.t_initial = t_initial
        self.t_terminal = t_terminal
    
    def solve(self, unet, latents, prompt_embeds, 
              guidance_scale=7.5, num_inference_steps=10):
        """求解分段流ODE"""
        # 1. 准备时间步
        timesteps = self.get_timesteps(num_inference_steps)
        
        # 2. 迭代去噪
        for i, t in enumerate(timesteps):
            # Classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # 预测噪声
            noise_pred = unet(latent_model_input, t, prompt_embeds).sample
            
            # 应用 guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            # 使用调度器步进
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
```

---

=== SDXL 求解器

```python
class PFODESolverSDXL(PFODESolver):
    """SDXL模型的ODE求解器，支持额外的条件输入"""
    
    def _get_add_time_ids(self, original_size, crops_coords_top_left, 
                          target_size, dtype):
        """生成SDXL所需的额外时间嵌入"""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def solve(self, unet, latents, prompt_embeds, pooled_prompt_embeds,
              add_time_ids, guidance_scale=5.0, num_inference_steps=10):
        """求解SDXL的ODE，包含pooled embeddings和time_ids"""
        timesteps = self.get_timesteps(num_inference_steps)
        
        for i, t in enumerate(timesteps):
            # SDXL需要额外的条件输入
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
            
            # 预测和去噪（与SD类似，但传入额外参数）
            noise_pred = unet(
                latent_model_input, t, prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            # ... 后续步骤与SD求解器类似
        
        return latents
```

---

== 工具函数

=== 权重管理

```python
# 从 utils_perflow.py

def merge_delta_weights_into_unet(unet, delta_weights):
    """合并增量权重到UNet模型"""
    state_dict = unet.state_dict()
    for key, delta in delta_weights.items():
        if key in state_dict:
            state_dict[key] = state_dict[key] + delta
    unet.load_state_dict(state_dict)
    return unet

def load_delta_weights_into_unet(unet, checkpoint_path):
    """从文件加载并合并增量权重"""
    # 支持 .safetensors 和 .bin 格式
    delta_weights = load_file(checkpoint_path)  # or torch.load()
    return merge_delta_weights_into_unet(unet, delta_weights)

def load_dreambooth_into_pipeline(pipeline, checkpoint_path):
    """加载DreamBooth检查点到pipeline"""
    # 加载并设置到pipeline的UNet
    unet = load_delta_weights_into_unet(pipeline.unet, checkpoint_path)
    pipeline.unet = unet
    return pipeline
```

---

== 使用示例

=== 基本用法

```python
from diffusers import StableDiffusionPipeline, PeRFlowScheduler

# 1. 加载模型和调度器
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# 2. 替换为PeRFlow调度器
scheduler = PeRFlowScheduler.from_pretrained(
    model_id, 
    subfolder="scheduler",
    num_windows=4
)
pipe.scheduler = scheduler

# 3. 生成图像（只需5-10步）
image = pipe(
    "a photo of an astronaut riding a horse on mars",
    num_inference_steps=10,  # 比原来的50步快5倍
    guidance_scale=7.5
).images[0]
```

---

=== 使用ODE求解器

```python
from diffusers.schedulers.pfode_solver import PFODESolver

# 创建求解器
solver = PFODESolver(
    scheduler=scheduler,
    t_initial=1.0,
    t_terminal=0.0
)

# 准备输入
latents = torch.randn((1, 4, 64, 64))
prompt_embeds = pipe.encode_prompt("a beautiful landscape")

# 求解ODE
denoised_latents = solver.solve(
    unet=pipe.unet,
    latents=latents,
    prompt_embeds=prompt_embeds,
    guidance_scale=7.5,
    num_inference_steps=10
)

# 解码为图像
image = pipe.vae.decode(denoised_latents / pipe.vae.config.scaling_factor).sample
```

---

== 测试覆盖

=== 完整的测试体系

框架包含 **69 个测试用例**，覆盖所有关键功能：

*PeRFlowScheduler 测试* (30个测试)
- 初始化配置测试
- 时间步生成和分布
- 各种预测类型 (epsilon, velocity, v_prediction)
- 噪声添加和移除
- 配置保存/加载
- 数值稳定性
- 批处理一致性

*ODE求解器测试* (20个测试)
- PFODESolver: 10个测试
- PFODESolverSDXL: 10个测试
- 包括不同分辨率、批处理、guidance测试

*工具函数测试* (19个测试)
- 权重合并和加载
- 数值精度保持
- 文件格式兼容性

---

== 实现成果

=== 框架统计

#grid(
  columns: (1fr, 1fr),
  align: horizon,
  [
    *源代码*
    - 总行数: 564
    - 创建文件: 3
    - 类: 4个
    - 方法: 18个
    - 函数: 3个
  ],
  [
    *测试代码*
    - 总行数: 1,251
    - 测试文件: 3
    - 测试方法: 69个
    - TODO注释: 0
  ],
)

*文档*
- 实现计划文档
- 修改总结文档
- 完整的API文档

---

=== 设计亮点

1. *分段近似*: 时间域分为 K 个窗口（默认4个），线性流近似
2. *三种预测类型*: 支持 ddim_eps, diff_eps, velocity
3. *窗口感知调度*: 时间步在窗口间分布，非均匀分布
4. *SDXL支持*: 独立的求解器类，支持pooled embeddings和time_ids
5. *增量权重*: 通过增量权重合并支持微调模型
6. *测试驱动*: 69个测试用例定义准确的预期行为

---

== 集成要点

=== 与 Diffusers 兼容

```python
# 已集成到 diffusers 包导出
from diffusers import PeRFlowScheduler

# 兼容标准调度器API
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    PeRFlowScheduler,  # 新增
)

# 支持 from_pretrained
scheduler = PeRFlowScheduler.from_pretrained(
    "model_id",
    subfolder="scheduler"
)

# 兼容所有标准Pipeline
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("...")
pipe.scheduler = scheduler  # 直接替换
```

---

=== 性能对比

#grid(
  columns: (1fr, 1fr),
  align: horizon,
  [
    *传统调度器*
    - DDIM: 50步
    - DPM++: 25步
    - Euler: 30步
    
    生成时间: ~5-10秒
  ],
  [
    *PeRFlow调度器*
    - PeRFlow: 5-10步
    
    生成时间: ~1-2秒
    
    *加速比: 5-10倍* 🚀
  ],
)

质量保持: 通过分段线性近似，在大幅减少步数的同时保持生成质量

---

---
