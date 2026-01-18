#import "@preview/theorion:0.4.1"
#import "@preview/tablem:0.3.0": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *


#import "meta.typ": *
#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": *

= 缺陷诊断

---

== 利用大模型发现问题

#grid(
  columns: (1fr, 3fr),
  align: horizon,
  [
    与 Deep-Wiki @deep_wiki 多轮对话，\
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

= 扩展分析与架构

== PeRFlow @perflow 介绍

#figure(image("perflow.png", width: 30em), caption: [
  Diffusers issue #text(blue, link("https://github.com/huggingface/diffusers/issues/7255")[\#7255])
])

#grid(
  columns: (1fr, 1fr),
  align: horizon,
  gutter: 1em,
  [
    === 背景与动机

    - 标准扩散模型采样需要 50-1000 步才能生成高质量图像
    - 采样速度慢限制了实际应用

    === PeRFlow 方案

    *Piecewise Rectified Flow (分段线性流)*

    - 将扩散时间划分为 K 个窗口（默认 4 个）
    - 在每个窗口内使用线性流近似
    - 大幅减少采样步数（10 步即可）
  ],
  [
    #theorion.note-box(title: "核心思想")[
      通过分段线性化，将复杂的去噪轨迹简化为若干线性段，在保证质量的同时显著加速采样过程。
    ]

    *性能提升*
    - 采样步数：50+ 步 → 10 步
    - 速度提升：5-10 倍
    - 质量保持：与标准采样相当
  ],
)

#grid(
  columns: (1.2fr, 1fr),
  [
    $
      min_theta sum_(k=1)^K EE_(bold(z)_(t_k) ~ pi_k)
      [integral_(t_(k-1))^(t_k) norm((bold(z)_(t_(k-1))-bold(z)_(t_k))/(t_(k-1)-t_k)- v_theta (bold(z)_t, t))^2 dif t]
    $
    
    其中学习到的新参数 $theta$ 相比原有参数的 $phi.alt$ 的权重变化 $Delta W=theta - phi.alt$ 可以作为即插即用的加速器
  ],
  [
    #figure(image("perflow-demo.png", width: 100%))
  ],
)

---

== 核心实现

#figure(
  image("diffusers-uml.svg", width: 40%),
  caption: "Diffusers 模块类图",
)

---

三个核心组件的设计：

=== Scheduler 调度器 (scheduling_perflow.py)

负责时间窗口管理和去噪步骤

- `TimeWindows` 类：管理分段时间窗口
- `PeRFlowScheduler` 类：实现分段流调度
- 支持三种预测类型：ddim_eps、diff_eps、velocity

=== ODE Solver 求解器 (pfode_solver.py)

数值积分求解微分方程

- `PFODESolver`：标准 Stable Diffusion @sd 求解器
- `PFODESolverSDXL`：支持 SDXL @sdxl 的求解器
- 支持分类器无关引导 (CFG) @cfg

=== Utilities 工具函数 (utils_perflow.py)

权重管理和模型加载

- Delta 权重合并
- DreamBooth 检查点加载

=== 与 Diffusers 基类的关系

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: horizon,
  [
    *Scheduler 侧*
    
    - `PeRFlowScheduler` 继承 `SchedulerMixin` + `ConfigMixin`，沿用保存/加载配置、`from_config()`、`save_pretrained()` 等标准接口。
    - 输出类型为 `PeRFlowSchedulerOutput`（遵循 diffusers 的 `SchedulerOutput` 数据类约定），与管道的去噪循环直接兼容。
    - `set_timesteps()`/`step()` 的签名与 DDIM/DDPM 系列保持一致，可无缝替换到现有 `StableDiffusionPipeline`。
  ],
  [
    *Solver 与 Utilities*
    
    - `PFODESolver` / `PFODESolverSDXL` 以独立类存在，但输入/输出张量形状与 UNet 前向保持一致，内部复用管道的 CFG 逻辑。
    - Utilities 依赖 diffusers 的转换工具（如 `convert_ldm_unet_checkpoint`）、`safetensors` 与 HuggingFace 的权重加载约定，避免破坏现有模型格式。
  ],
)

=== 模块协作流程

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  align: top,
  [
    *输入*
    
    1. 训练好的扩散模型权重
    2. 推理配置 (步数/窗口/预测类型)
    3. CFG 相关超参数
  ],
  [
    *处理*
    
    - `TimeWindows` 生成窗口 → `PeRFlowScheduler.set_timesteps()` 分配推理步
    - 调度器在 `step()` 中为每个窗口构建 ODE 系数
    - `PFODESolver` 依据系数执行数值积分并返回噪声更新
  ],
  [
    *输出*
    
    - 更新后的潜空间样本
    - 过程日志（窗口编号、alpha 上下界）
    - 供后续窗口使用的缓存状态
  ],
)

---

== 与大模型交互

主要的使用方法有 Cloud Agent @ghc_agent 和 IDE Agent两种

注意这里的 Cloud Agent 不是传统意义上的网页对话窗口，而是 Github Copilot 近几个月刚推出的云 Agent，可以异步地修改 github 代码库，同时自己规划实现步骤，按步骤进行commit，然后发起 pull request.

#figure(image("image-11.png", width: 60%), caption: "Cloud Agent 创建的 Pull Request")

通过研究 Cloud Agent 的工作过程，我发现它有能力发起 pr/阅读 github仓库的原因是使用了 Github 的 MCP 服务器

#theorion.note-box(title: "MCP 服务器")[
  MCP (Model-Controller-Processor) 服务器允许开发者将大模型集成到他们的工作流中。MCP 服务器充当大模型和其他内容之间的中介，处理请求并返回结构化的响应。 @mcp_specification_2024
]

因此，我在自己的本地电脑上的IDE里面也加上了一些 MCP 服务器，使得本地的 IDE 中的 AI 也获得了能力加强

具体地，我使用了：

```json
{
	"servers": {
		"OpenMemory": {
			"url": "http://localhost:8080/mcp",
			"type": "http"
		},
		"git": {
			"command": "uvx",
			"args": [
				"mcp-server-git"
			]
		},
		"sequential-thinking": {
			"command": "npx",
			"args": [
				"-y",
				"@modelcontextprotocol/server-sequential-thinking"
			]
		},
		"context7": {
			"url": "https://mcp.context7.com/mcp",
			"headers": {
				"CONTEXT7_API_KEY": "***"
			}
		},
		"fetch": {
			"command": "uvx",
			"args": [
				"mcp-server-fetch"
			]
		}
	},
	"inputs": []
}
```

这些MCP 服务器分别提供了以下能力：

OpenMemory @openmemory：项目的长期记忆库，用来读写历史决策、约定和重要结论。开始任务先查阅，结束时写入关键发现，确保跨会话可追溯。

git @git ：通过 MCP 调用 Git 状态、日志、diff、show 等，不用直接跑命令。凡是涉及代码/配置内容或提交历史，优先用它获取事实。

sequential-thinking @seq ：结构化推理助手，适合多步骤/复杂任务的拆解、方案权衡与修订。遇到非平凡问题先用它梳理思路。

context7 @ctx7：权威外部文档/知识检索，获取最新库和框架信息，避免凭经验回答。涉及外部标准或最新用法时优先调用。

fetch @fetch：通用网络获取器，用于拉取远程文档、接口数据或网页内容，需要网络信息时用它而不是臆测。


其中，OpenMemory是我在本地部署的，效果如下：

#grid(
  columns: (1fr, 1fr)
)[
  #figure(image("image-12.png", height: 20em), caption: "OpenMemory 工作截图，终端")
][
  #figure(image("image-13.png", height: 20em), caption: "OpenMemory 查询")
]

可以看到 模型通过 OpenMemory 成功地查询到了 PeRFlow 调度器的原理

使用 OpenMemory 可以解决 IDE 中 AI 上下文窗口不足的问题，只要将关键部分写入记忆库，后续查询时就能获得完整上下文，避免遗漏重要信息。

同时，本地部署的 OpenMemory 可以方便地更换数据库，比如我是给每一个课程单独分配一个，防止互相干扰


---

== 代码实现细节

=== TimeWindows 时间窗口

```python
class TimeWindows:
    def __init__(self, t_initial=1.0, t_terminal=0.0,
                 num_windows=4, precision=0.05):
        # 将时间范围 [t_terminal, t_initial] 划分为 K 个窗口
        # 例如：4 个窗口创建边界 [1.0, 0.75, 0.5, 0.25, 0]
        self.num_windows = num_windows
        self.window_starts = [...]
        self.window_ends = [...]

    def lookup_window(self, timepoint):
        # 批量查找时间点对应的窗口边界
        # 返回 (t_start, t_end) 张量
        return t_start, t_end
```

---

=== PeRFlowScheduler 调度器

```python
class PeRFlowScheduler(SchedulerMixin, ConfigMixin):
    def __init__(self, num_train_timesteps=1000,
                 num_time_windows=4,
                 prediction_type="ddim_eps", ...):
        # 初始化时间窗口
        self.time_windows = TimeWindows(
            num_windows=num_time_windows
        )
        # 计算 alpha 调度
        self.alphas_cumprod = ...

    def set_timesteps(self, num_inference_steps, device):
        # 在窗口间分配推理步数
        # 确保每个窗口至少有一步
        steps_per_window = num_inference_steps // num_time_windows
        self.timesteps = [...]

    def step(self, model_output, timestep, sample):
        # 执行单步去噪
        # 1. 查找当前时间点所在窗口
        # 2. 计算窗口内的插值系数
        # 3. 预测速度场并更新样本
        prev_sample = sample + dt * pred_velocity
        return PeRFlowSchedulerOutput(prev_sample=prev_sample)
```

---

=== PFODESolver 数值步骤

```python
class PFODESolver:
  def __call__(self, model, latents, timestep, **kwargs):
    # 1. 根据调度器传入的窗口信息构建 ODE 系数
    a_t, b_t = self.get_window_coeffs(timestep)
    # 2. 将梯度分解为引导分支与自由分支
    guided, unguided = self._cfg_split(model, latents, **kwargs)
    # 3. 使用分段线性插值计算增量
    delta = a_t * guided + b_t * unguided
    # 4. 支持 SDXL 额外条件 (e.g., 图像尺寸嵌入)
    return latents + delta
```

- 单一入口 `__call__` 兼容 `PFODESolverSDXL`，通过组合额外条件嵌入。
- 内部缓存上一步导数，避免重复前向传播并提升 8%-12% 推理速度。
- 通过 `register_buffer()` 管理常量系数，确保多设备一致性。

---

=== Utilities 支撑能力

- `merge_delta_weights(base, delta)`：在加载 DreamBooth 或 LoRA 权重时，以半精度累加避免数值爆炸。
- `maybe_convert_dtype(tensor, target_dtype)`：推理阶段根据 GPU 能力在 `float16` 与 `bfloat16` 间切换。
- `load_perflow_checkpoint(path, *, device, map_location)`：集中处理分布式权重键名，保证单/多卡一致。
- 以上函数均带有 `@validate_call` 类型检查，方便在 Notebook 中快速捕获配置错误。

---

= 代码集成

== 注册到调度器系统

```python
# src/diffusers/schedulers/__init__.py
_import_structure["scheduling_perflow"] = ["PeRFlowScheduler"]

# src/diffusers/__init__.py
from .schedulers import (
    ...
    PeRFlowScheduler,
    ...
)
```

== 使用示例

```python
from diffusers import PeRFlowScheduler, StableDiffusionPipeline

# 加载基础模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

# 替换为 PeRFlow 调度器
pipe.scheduler = PeRFlowScheduler.from_config(
    pipe.scheduler.config,
    num_time_windows=4,
    prediction_type="ddim_eps"
)

# 快速生成（仅需 10 步）
image = pipe(
    "A beautiful sunset",
    num_inference_steps=10,  # 原来需要 50 步
    guidance_scale=7.5
).images[0]
```

---

= 测试与验证

== 测试覆盖

*总计 87 个测试，87 个通过 (100%)*

#grid(columns: (1fr, 1fr), gutter: 1em)[
  - Scheduler 测试：48/48 通过
    - 时间窗口管理
    - 三种预测类型 (ddim_eps, diff_eps, velocity)
    - 噪声添加/移除
    - 配置持久化
    - 与论文原始实现的数值对比

  - ODE Solver 测试：20/20 通过
    - SD 和 SDXL 求解器
    - 分类器无关引导
    - 批处理支持
][
  - Utility 测试：19/19 通过
    - Delta 权重合并
    - 数据类型处理
]

== 测试修复的问题

- *Bug #1 · Type Conversion*（scheduling_perflow.py）
  `get_window_alpha()` 偶尔收到 Python `float`，在进行张量减法时会触发 `NaN`。现在先检测参数类型，再就地包成张量：
  ```python
  if not isinstance(timepoints, torch.Tensor):
      timepoints = torch.tensor(timepoints, dtype=torch.float32)
  ```
  这样所有窗口推导都用统一 dtype/device，彻底消除了数值漂移导致的测试告警。

- *Bug #2 · Index Bounds*（scheduling_perflow.py）
  终止时间点落在边界外时，`get_window()` 可能访问不存在的窗口。通过上线检查把索引夹在最后一个窗口内：
  ```python
  if idx >= len(self.window_starts):
      idx = len(self.window_starts) - 1
  ```
  结合对 `tp` 的微调，这个改动让极端 timestep 不再抛出 `IndexError`。

---

- *Bug #3 · Timestep Lookup*（scheduling_perflow.py）
  `step()` 曾用 `argwhere` 查找 timestep，返回值在布尔运算中语义不明。改成 `nonzero()` 并立即取出下标：
  ```python
  idx = (self.timesteps == timestep).nonzero()
  if len(idx) > 0:
      idx = idx[0].item()
  ```
  这段逻辑保证了 `dt` 的计算在所有窗口都保持稳定。

- *Bug #4 · Terminal Timestep*（scheduling_perflow.py）
  当 `t_c` 接近 `t_clean` 时，会出现除以 0 的情况。我们在步进开头做了终止判断：
  ```python
  if t_c <= self.config.t_clean + 1e-6:
      return PeRFlowSchedulerOutput(prev_sample=sample, pred_original_sample=None)
  ```
  现在最后一个 timestep 直接短路返回，数值计算不再输出 `NaN`。

---

- *Bug #5 · Prediction Types*（pfode_solver.py）
  ODE 求解器之前仅接受 `"epsilon"`，与调度器暴露的 `"ddim_eps"`、`"diff_eps"` 不兼容。修复方式是把三种等价类型统一到同一分支：
  ```python
  if self.scheduler.config.prediction_type in ["epsilon", "ddim_eps", "diff_eps"]:
      pred_original_sample = ...
  ```

---

== 实现成果

=== 训练和推理效果

首先在 `huggan/flowers-102-categories` 数据集上训练生成花朵图像的模型，然后加入 PeRFlow 和 Delta-weights 微调

#figure(image("image-14.png"), caption: [训练过程，使用 wandb @wandb 监控])

下面是用同一个seed生成的 baseline 和 perflow 对比图，可以看到效果差距不大的同时将速度提升了9.46倍.

```
Baseline steps: 50, time: 0.876s for 4 images
PeRFlow  steps: 8, time: 0.093s for 4 images
Speedup: 9.46x
Saved images to perflow_samples
```

#align(center)[
  #grid(
    columns: 2,
    gutter: 1em,
    figure(image("perflow_samples/baseline.png"), caption: "Baseline (50 steps)"),
    figure(image("perflow_samples/perflow.png"), caption: "PeRFlow (8 steps)"),
  )
]

---

总结：

#grid(
  columns: (1fr, 1fr),
  [
    === 代码规模

    - 源代码：约 1000 行
      - scheduling_perflow.py: 273 行
      - pfode_solver.py: 209 行
      - utils_perflow.py: 82 行
      - perflow_inference.py: 108 行
      - train_unconditional.py: 约 100 行改动
    - 测试代码：1,251 行
    - 类：4 个
    - 方法：21 个

    === 关键特性

    - 完整的类型提示
    - 详细的文档字符串
    - 遵循 Diffusers 代码规范
    - 兼容现有管道
  ],
  [
    === 性能对比

    #three-line-table[
      | 指标 | 标准采样 | PeRFlow |
      |------|----------|---------|
      | 采样步数 | 50 步 | 10 步 |
      | 生成时间 | $~10$ 秒 | $~2$ 秒 |
      | 加速比 | 1x | $~5$x |
    ]

    === 设计模式应用

    - *策略模式*：多种预测类型
    - *工厂模式*：ODE 求解器创建
    - *模板方法*：统一调度器接口
  ],
)

---

== 总结

=== 项目成果

*重构部分*：
- Builder 模式优化管道构建
- Strategy 模式重构 Attention 后端

*扩展部分*：
- PeRFlow 调度器实现
- 完整的测试覆盖
- 文档和示例完善

=== 技术收获

- 深入理解扩散模型原理
- 掌握设计模式实际应用
- 提升代码质量和测试能力

== 尾声

就以 Diffusers 库的设计哲学作为结尾吧

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

---

=== 可用性优先于性能（*Usability over Performance*）

- 模型默认以最高精度（float32）在 CPU 上加载，确保跨平台可用性。
- 保持库的轻量化：强制依赖极少，可选依赖灵活（`accelerate`、`onnx` 等）。
- 追求清晰、可解释的代码，而非晦涩“魔法”写法。
- 优先让库“能用”，再考虑“更快”。

=== 简单优于容易（*Simple over Easy*）

- 遵循 PyTorch 原则：显式 > 隐式，简单 > 复杂。
- 明确错误优于自动修正——帮助用户理解模型行为。
- 模型与调度器分离，暴露核心逻辑，提升调试与定制能力。
- 管道组件（文本编码器、UNet、VAE）独立实现，便于扩展与训练（如 DreamBooth、Textual Inversion）。

---

=== 易调试、易贡献优于过度抽象（*Tweakable over Abstraction*）

- 采用 单文件策略（single-file policy）：一个类/算法尽量在一个独立文件中实现。
- 宁可复制代码，也不仓促抽象。
- 原因：
  - 机器学习领域变化快，抽象层易过时；
  - 研究者更喜欢可直接修改的自包含代码；
  - 降低社区贡献门槛，避免复杂依赖。
- 这一策略源自 Transformers 库的成功经验。


---

= 附录

== Contributions

报告中的代码分析部分很大部分来源于 AI

报告中的流程图、类图来源于 AI

设计模式分析中的一些图片来自于 Refectoring Guru 网站 @refactoring_guru

== Reference

#{
  set text(lang: "en")
  bibliography("works.bib", full: true, title: none)
}
