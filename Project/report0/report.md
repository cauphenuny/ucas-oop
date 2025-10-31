https://github.com/huggingface/diffusers/issues/7255

https://github.com/huggingface/diffusers/issues/12235

# Diffusers 的主要组成部分

Diffusers 库由三个核心组件构成,它们协同工作以创建完整的扩散模型系统:<cite />

## 1. Pipelines (管道)

Pipelines 是高级的端到端类,用于快速运行预训练扩散模型的推理 [1](#0-0) 。所有 pipeline 都继承自 `DiffusionPipeline` 基类 [2](#0-1) 。

Pipeline 将多个独立训练的模型和调度器组件编排在一起 [3](#0-2) 。例如,Stable Diffusion 包含三个独立训练的模型:自动编码器(Autoencoder)、条件 UNet 和 CLIP 文本编码器,以及一个调度器组件 [4](#0-3) 。

使用示例: [5](#0-4) 

## 2. Models (模型)

Models 是神经网络组件,用于去噪处理<cite />。所有模型都继承自 `ModelMixin` 和 `ConfigMixin` [6](#0-5) 。

模型对应特定的架构类型 [7](#0-6) 。主要模型类型包括:
- **UNet 架构**: 如 `UNet2DConditionModel`,用于期望 2D 图像输入并受上下文条件约束的所有 UNet 变体 [7](#0-6) 
- **Transformer 模型**: 如 `SD3Transformer2DModel`、`FluxTransformer2DModel`<cite />
- **自动编码器**: 如 `AutoencoderKL`,用于潜在空间编码和解码<cite />

模型位于 `src/diffusers/models` 目录中 [8](#0-7) 。

## 3. Schedulers (调度器)

Schedulers 控制噪声调度和采样过程 [9](#0-8) 。它们负责在推理时引导去噪过程,以及在训练时定义噪声计划 [10](#0-9) 。

所有调度器都继承自 `SchedulerMixin` 和 `ConfigMixin` [11](#0-10) 。常见的调度器包括:
- `DDPMScheduler`: DDPM 调度算法<cite />
- `DDIMScheduler`: DDIM 调度算法<cite />
- `EulerDiscreteScheduler`: Euler 离散方法<cite />
- `FlowMatchEulerDiscreteScheduler`: 流匹配方法<cite />

调度器可以轻松互换,以平衡生成速度和输出质量 [12](#0-11) 。使用示例: [13](#0-12) 

## 组件组合方式

这三个组件通过模块化设计组合在一起<cite />。Pipeline 作为统一接口,将独立训练的模型和高度适应性的调度器组件编排起来<cite />。每个 pipeline 的组件都记录在 `model_index.json` 文件中,可以通过同名属性访问,并可在不同 pipeline 之间共享 [14](#0-13) 。

## Notes

Diffusers 的设计哲学强调"可用性优于性能"、"简洁优于简易"和"可定制性优于抽象" [15](#0-14) 。这种三支柱架构使得用户既可以使用高级 pipeline 进行快速推理,也可以组合单个组件构建自定义的端到端扩散系统 [16](#0-15) 。

Wiki pages you might want to explore:
- [Introduction to Diffusers (huggingface/diffusers)](/wiki/huggingface/diffusers#1)

### Citations

**File:** README.md (L30-30)
```markdown
🤗 Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you're looking for a simple inference solution or training your own diffusion models, 🤗 Diffusers is a modular toolbox that supports both. Our library is designed with a focus on [usability over performance](https://huggingface.co/docs/diffusers/conceptual/philosophy#usability-over-performance), [simple over easy](https://huggingface.co/docs/diffusers/conceptual/philosophy#simple-over-easy), and [customizability over abstractions](https://huggingface.co/docs/diffusers/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).
```

**File:** README.md (L34-34)
```markdown
- State-of-the-art [diffusion pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview) that can be run in inference with just a few lines of code.
```

**File:** README.md (L35-35)
```markdown
- Interchangeable noise [schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview) for different diffusion speeds and output quality.
```

**File:** README.md (L36-36)
```markdown
- Pretrained [models](https://huggingface.co/docs/diffusers/api/models/overview) that can be used as building blocks, and combined with schedulers, for creating your own end-to-end diffusion systems.
```

**File:** README.md (L72-78)
```markdown
```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

**File:** README.md (L83-90)
```markdown
```python
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)
```

**File:** docs/source/en/conceptual/philosophy.md (L67-67)
```markdown
- Pipelines all inherit from [`DiffusionPipeline`].
```

**File:** docs/source/en/conceptual/philosophy.md (L68-68)
```markdown
- Every pipeline consists of different model and scheduler components, that are documented in the [`model_index.json` file](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/model_index.json), are accessible under the same name as attributes of the pipeline and can be shared between pipelines with [`DiffusionPipeline.components`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.components) function.
```

**File:** docs/source/en/conceptual/philosophy.md (L83-83)
```markdown
- Models correspond to **a type of model architecture**. *E.g.* the [`UNet2DConditionModel`] class is used for all UNet variations that expect 2D image inputs and are conditioned on some context.
```

**File:** docs/source/en/conceptual/philosophy.md (L84-84)
```markdown
- All models can be found in [`src/diffusers/models`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models) and every model architecture shall be defined in its file, e.g. [`unets/unet_2d_condition.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py), [`transformers/transformer_2d.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_2d.py), etc...
```

**File:** docs/source/en/conceptual/philosophy.md (L87-87)
```markdown
- Models all inherit from `ModelMixin` and `ConfigMixin`.
```

**File:** src/diffusers/pipelines/README.md (L3-5)
```markdown
Pipelines provide a simple way to run state-of-the-art diffusion models in inference.
Most diffusion systems consist of multiple independently-trained models and highly adaptable scheduler
components - all of which are needed to have a functioning end-to-end diffusion system.
```

**File:** src/diffusers/pipelines/README.md (L7-11)
```markdown
As an example, [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) has three independently trained models:
- [Autoencoder](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/models/vae.py#L392)
- [Conditional Unet](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/models/unet_2d_condition.py#L12)
- [CLIP text encoder](https://huggingface.co/docs/transformers/main/en/model_doc/clip#transformers.CLIPTextModel)
- a scheduler component, [scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py),
```

**File:** docs/source/zh/conceptual/philosophy.md (L89-91)
```markdown
### 调度器(Schedulers)

调度器负责引导推理去噪过程及定义训练噪声计划。它们设计为独立的可加载配置类，严格遵循**单文件政策**。
```

**File:** docs/source/zh/conceptual/philosophy.md (L98-98)
```markdown
- 所有调度器继承`SchedulerMixin`和`ConfigMixin`。
```

**File:** docs/source/zh/index.md (L29-29)
```markdown
- 可交替使用的各种噪声调度器 [noise schedulers](api/schedulers/overview)，用于平衡生成速度和质量。
```


根据您提供的 Hugging Face 模型链接和代码库上下文,您想要为 Wan 添加一个类似 CogVideoX-Fun-Control 的控制分支。<cite/>让我分析一下需要实现哪些功能。

## 核心功能需求

### 1. 创建 WanFunControlPipeline

参考 `CogVideoXFunControlPipeline` 的实现 [1](#8-0) ,需要创建一个新的 `WanFunControlPipeline` 类,继承自 `DiffusionPipeline` 和 `WanLoraLoaderMixin`。

**主要组件:**
- `tokenizer`: T5 分词器
- `text_encoder`: UMT5 文本编码器  
- `transformer`: Wan 3D Transformer 去噪模型
- `vae`: AutoencoderKLWan 视频编解码器
- `scheduler`: FlowMatchEulerDiscreteScheduler 调度器

### 2. 控制视频处理功能

需要实现 `prepare_control_latents()` 方法 [2](#8-1) ,用于:
- 将控制视频编码为潜在表示
- 支持 mask 和 masked_image 输入
- 使用 VAE 编码控制信号

### 3. 推理循环中的控制信号注入

在 `__call__()` 方法的去噪循环中 [3](#8-2) ,需要:
- 将控制视频潜在表示与噪声潜在表示拼接
- 在通道维度上拼接: `torch.cat([latent_model_input, latent_control_input], dim=2)`
- 支持 classifier-free guidance 时的批次处理

### 4. Transformer 模型适配

如果需要特殊的控制机制,可能需要创建类似 `WanVACETransformerBlock` 的结构 [4](#8-3) ,支持:
- 控制隐藏状态的输入投影
- 自注意力和交叉注意力机制
- 控制信号的条件化处理

### 5. 输入验证

实现 `check_inputs()` 方法 [5](#8-4) ,验证:
- `control_video` 和 `control_video_latents` 不能同时提供
- 高度和宽度必须能被 8 整除
- prompt 和 embedding 的一致性检查

### 6. 视频预处理

使用 `VideoProcessor` 预处理控制视频 [6](#8-5) :
- 调整视频尺寸到指定的 height 和 width
- 转换为正确的数据类型和设备

### 7. 模型配置

在转换脚本中添加 Wan2.2-Fun-A14B-Control 的配置 [7](#8-6) ,包括:
- `in_channels`: 可能需要调整以支持额外的控制通道
- 其他 transformer 配置参数

### 8. 文档和示例

参考 CogVideoX 文档 [8](#8-7) ,需要添加:
- API 文档说明
- 使用示例代码
- 控制信号的格式说明

## 实现步骤总结

1. **创建新文件**: `src/diffusers/pipelines/wan/pipeline_wan_fun_control.py`
2. **实现核心方法**: `prepare_control_latents()`, `check_inputs()`, `__call__()`
3. **修改 Transformer**: 如需特殊控制机制,扩展 `WanVACETransformer3DModel`
4. **注册 Pipeline**: 在 `__init__.py` 中导出新的 pipeline 类
5. **添加转换脚本**: 支持从原始权重转换
6. **编写文档**: 添加使用说明和示例

## Notes

Wan2.2-Fun-A14B-Control 模型很可能采用与 WanVACE 类似的架构 [9](#8-8) ,支持多种控制信号(depth, pose, sketch 等) [10](#8-9) 。关键是要正确处理控制视频的编码和在去噪过程中的注入方式。

Wiki pages you might want to explore:
- [Fine-tuning and Training (huggingface/diffusers)](/wiki/huggingface/diffusers#5)

### Citations

**File:** src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py (L154-175)
```python
class CogVideoXFunControlPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for controlled text-to-video generation using CogVideoX Fun.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """
```

**File:** src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py (L362-387)
```python
    def prepare_control_latents(
        self, mask: Optional[torch.Tensor] = None, masked_image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            masks = []
            for i in range(mask.size(0)):
                current_mask = mask[i].unsqueeze(0)
                current_mask = self.vae.encode(current_mask)[0]
                current_mask = current_mask.mode()
                masks.append(current_mask)
            mask = torch.cat(masks, dim=0)
            mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            mask_pixel_values = []
            for i in range(masked_image.size(0)):
                mask_pixel_value = masked_image[i].unsqueeze(0)
                mask_pixel_value = self.vae.encode(mask_pixel_value)[0]
                mask_pixel_value = mask_pixel_value.mode()
                mask_pixel_values.append(mask_pixel_value)
            masked_image_latents = torch.cat(mask_pixel_values, dim=0)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents
```

**File:** src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py (L415-471)
```python
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        control_video=None,
        control_video_latents=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if control_video is not None and control_video_latents is not None:
            raise ValueError(
                "Cannot pass both `control_video` and `control_video_latents`. Please make sure to pass only one of these parameters."
            )
```

**File:** src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py (L748-752)
```python
            control_video = self.video_processor.preprocess_video(control_video, height=height, width=width)
            control_video = control_video.to(device=device, dtype=prompt_embeds.dtype)

        _, control_video_latents = self.prepare_control_latents(None, control_video)
        control_video_latents = control_video_latents.permute(0, 2, 1, 3, 4)
```

**File:** src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py (L775-781)
```python
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_control_input = (
                    torch.cat([control_video_latents] * 2) if do_classifier_free_guidance else control_video_latents
                )
                latent_model_input = torch.cat([latent_model_input, latent_control_input], dim=2)
```

**File:** src/diffusers/models/transformers/transformer_wan_vace.py (L41-134)
```python
class WanVACETransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        apply_input_projection: bool = False,
        apply_output_projection: bool = False,
    ):
        super().__init__()

        # 1. Input projection
        self.proj_in = None
        if apply_input_projection:
            self.proj_in = nn.Linear(dim, dim)

        # 2. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=WanAttnProcessor(),
        )

        # 3. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. Output projection
        self.proj_out = None
        if apply_output_projection:
            self.proj_out = nn.Linear(dim, dim)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(control_hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            control_hidden_states
        )
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        control_hidden_states = (control_hidden_states.float() + attn_output * gate_msa).type_as(control_hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(control_hidden_states.float()).type_as(control_hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        control_hidden_states = control_hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(control_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            control_hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        control_hidden_states = (control_hidden_states.float() + ff_output.float() * c_gate_msa).type_as(
            control_hidden_states
        )

        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states = self.proj_out(control_hidden_states)

        return conditioning_states, control_hidden_states
```

**File:** scripts/convert_wan_to_diffusers.py (L302-320)
```python
    elif model_type == "Wan2.2-T2V-A14B":
        config = {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
```

**File:** docs/source/en/api/pipelines/cogvideox.md (L209-213)
```markdown
## CogVideoXFunControlPipeline

[[autodoc]] CogVideoXFunControlPipeline
  - all
  - __call__
```

**File:** src/diffusers/pipelines/wan/pipeline_wan_vace.py (L141-186)
```python
class WanVACEPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for controllable generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanVACETransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
```

**File:** docs/source/en/api/pipelines/wan.md (L241-250)
```markdown
Wan VACE supports various generation techniques which achieve controllable video generation. Some of the capabilities include:
- Control to Video (Depth, Pose, Sketch, Flow, Grayscale, Scribble, Layout, Boundary Box, etc.). Recommended library for preprocessing videos to obtain control videos: [huggingface/controlnet_aux]()
- Image/Video to Video (first frame, last frame, starting clip, ending clip, random clips)
- Inpainting and Outpainting
- Subject to Video (faces, object, characters, etc.)
- Composition to Video (reference anything, animate anything, swap anything, expand anything, move anything, etc.)

The code snippets available in [this](https://github.com/huggingface/diffusers/pull/11582) pull request demonstrate some examples of how videos can be generated with controllability signals.

The general rule of thumb to keep in mind when preparing inputs for the VACE pipeline is that the input images, or frames of a video that you want to use for conditioning, should have a corresponding mask that is black in color. The black mask signifies that the model will not generate new content for that area, and only use those parts for conditioning the generation process. For parts/frames that should be generated by the model, the mask should be white in color.
```

# Adding a New Scheduler and Loader to Diffusers

Based on the Diffusers library codebase, here's a comprehensive guide on how to add a new scheduler (like PeRFlow) and loader to the library:

## Adding a New Scheduler

### 1. **Review Design Philosophy**

Before implementing, you must read and follow the design philosophy outlined in the library. Schedulers follow the **single-file policy**, meaning all code for a scheduler should be self-contained in one file. [1](#9-0) 

### 2. **Create the Scheduler File**

Create a new file in `src/diffusers/schedulers/` following the naming convention `scheduling_<scheduler_name>.py`. For PeRFlow, this would be `scheduling_perflow.py`.

Your scheduler must:
- Inherit from `SchedulerMixin` and `ConfigMixin`
- Implement a `set_num_inference_steps()` method
- Implement a `step()` function
- Expose timesteps via a `timesteps` attribute [2](#9-1) 

### 3. **Implement Core Methods**

Use an existing scheduler as a template. The `FlowMatchEulerDiscreteScheduler` is a good reference for flow-matching based methods: [3](#9-2) 

Your scheduler class should use the `@register_to_config` decorator for the `__init__` method: [4](#9-3) 

The `set_timesteps` method is crucial for configuring the denoising process: [5](#9-4) 

The `step` method performs the actual denoising step: [6](#9-5) 

### 4. **Register the Scheduler**

Add your scheduler to the import structure in `src/diffusers/schedulers/__init__.py`: [7](#9-6) 

You'll need to add entries in both the `_import_structure` dictionary and the `TYPE_CHECKING` section.

### 5. **Export from Main Package**

The scheduler will be automatically exported through the main `__init__.py` via the lazy loading mechanism, but ensure it's properly registered in the schedulers module.

### 6. **Write Tests**

Create a test file in `tests/schedulers/test_scheduler_<name>.py`. Your test should inherit from `SchedulerCommonTest`: [8](#9-7) 

Implement a `full_loop` method to test the complete denoising process: [9](#9-8) 

Add test methods for various configurations and edge cases: [10](#9-9) 

### 7. **Follow Contribution Guidelines**

Review and follow the contribution guidelines, especially for adding new schedulers: [11](#9-10) 

## Adding a New Loader

Loaders in Diffusers are implemented as **mixin classes** that provide additional loading functionality to pipelines or models.

### 1. **Understand the Loader Structure**

Examine the existing loader structure in `src/diffusers/loaders/`: [12](#9-11) 

### 2. **Create the Loader Mixin**

For a new loader (like a PeRFlow-specific loader), create a new file in `src/diffusers/loaders/`. The `FromSingleFileMixin` is a good example of a loader implementation: [13](#9-12) 

### 3. **Implement Loading Logic**

Your loader mixin should provide methods that:
- Load checkpoints from various sources (files, URLs, Hub)
- Handle configuration and model initialization
- Support different checkpoint formats
- Include proper error handling

The `load_single_file_sub_model` function shows how to handle different component types: [14](#9-13) 

### 4. **Register the Loader**

Add your loader to the imports in `src/diffusers/loaders/__init__.py` and ensure it's exported properly through the `_import_structure` dictionary.

### 5. **Integrate with Pipelines**

To use your loader, pipelines should inherit from your mixin class. This allows them to use the loading methods you've defined.

## File Structure Summary

```
src/diffusers/
├── schedulers/
│   ├── __init__.py                          # Register scheduler here
│   └── scheduling_perflow.py                # New scheduler implementation
├── loaders/
│   ├── __init__.py                          # Register loader here
│   └── perflow_loader.py                    # New loader implementation (if needed)
└── __init__.py                              # Main package exports

tests/
└── schedulers/
    └── test_scheduler_perflow.py            # Scheduler tests
```

## Notes

- **PeRFlow is not currently implemented** in the Diffusers library. You would need to implement it following these guidelines.
- Schedulers must follow the **single-file policy** to maintain code readability and ease of contribution.
- All new components should be linked to the original paper/codebase in the PR, and ideally ping the original authors.
- Testing is mandatory - PRs without high-coverage tests will not be merged.
- The library prioritizes **simple over easy** and **usability over performance** by default.
- Consider whether your scheduler is compatible with existing schedulers and update the `_compatibles` list accordingly.

For flow-matching based methods like PeRFlow, the `FlowMatchEulerDiscreteScheduler` serves as an excellent template to follow for implementation patterns and best practices.

### Citations

**File:** PHILOSOPHY.md (L95-109)
```markdown
### Schedulers

Schedulers are responsible to guide the denoising process for inference as well as to define a noise schedule for training. They are designed as individual classes with loadable configuration files and strongly follow the **single-file policy**.

The following design principles are followed:
- All schedulers are found in [`src/diffusers/schedulers`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers).
- Schedulers are **not** allowed to import from large utils files and shall be kept very self-contained.
- One scheduler Python file corresponds to one scheduler algorithm (as might be defined in a paper).
- If schedulers share similar functionalities, we can make use of the `# Copied from` mechanism.
- Schedulers all inherit from `SchedulerMixin` and `ConfigMixin`.
- Schedulers can be easily swapped out with the [`ConfigMixin.from_config`](https://huggingface.co/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config) method as explained in detail [here](./docs/source/en/using-diffusers/schedulers.md).
- Every scheduler has to have a `set_num_inference_steps`, and a `step` function. `set_num_inference_steps(...)` has to be called before every denoising process, *i.e.* before `step(...)` is called.
- Every scheduler exposes the timesteps to be "looped over" via a `timesteps` attribute, which is an array of timesteps the model will be called upon.
- The `step(...)` function takes a predicted model output and the "current" sample (x_t) and returns the "previous", slightly more denoised sample (x_t-1).
- Given the complexity of diffusion schedulers, the `step` function does not expose all the complexity and can be a bit of a "black box".
```

**File:** src/diffusers/schedulers/scheduling_utils.py (L75-94)
```python
class SchedulerMixin(PushToHubMixin):
    """
    Base class for all schedulers.

    [`SchedulerMixin`] contains common functions shared by all schedulers such as general loading and saving
    functionalities.

    [`ConfigMixin`] takes care of storing the configuration attributes (like `num_train_timesteps`) that are passed to
    the scheduler's `__init__` function, and the attributes can be accessed by `scheduler.config.num_train_timesteps`.

    Class attributes:
        - **_compatibles** (`List[str]`) -- A list of scheduler classes that are compatible with the parent scheduler
          class. Use [`~ConfigMixin.from_config`] to load a different compatible scheduler class (should be overridden
          by parent class).
    """

    config_name = SCHEDULER_CONFIG_NAME
    _compatibles = []
    has_compatibles = True

```

**File:** src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py (L47-89)
```python
class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation. Increasing `base_shift` reduces variation and image is more consistent
            with desired output.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors. Increasing `max_shift` encourages more variation and image may be
            more exaggerated or stylized.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to False):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to None):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to False):
            Whether to use Karras sigmas for step sizes in the noise schedule during sampling.
        use_exponential_sigmas (`bool`, defaults to False):
            Whether to use exponential sigmas for step sizes in the noise schedule during sampling.
        use_beta_sigmas (`bool`, defaults to False):
            Whether to use beta sigmas for step sizes in the noise schedule during sampling.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply. Either "exponential" or "linear".
        stochastic_sampling (`bool`, defaults to False):
            Whether to use stochastic sampling.
    """

    _compatibles = []
    order = 1

```

**File:** src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py (L90-116)
```python
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

```

**File:** src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py (L249-276)
```python
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")
```

**File:** src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py (L373-415)
```python
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
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
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
```

**File:** src/diffusers/schedulers/__init__.py (L40-78)
```python
else:
    _import_structure["deprecated"] = ["KarrasVeScheduler", "ScoreSdeVpScheduler"]
    _import_structure["scheduling_amused"] = ["AmusedScheduler"]
    _import_structure["scheduling_consistency_decoder"] = ["ConsistencyDecoderScheduler"]
    _import_structure["scheduling_consistency_models"] = ["CMStochasticIterativeScheduler"]
    _import_structure["scheduling_ddim"] = ["DDIMScheduler"]
    _import_structure["scheduling_ddim_cogvideox"] = ["CogVideoXDDIMScheduler"]
    _import_structure["scheduling_ddim_inverse"] = ["DDIMInverseScheduler"]
    _import_structure["scheduling_ddim_parallel"] = ["DDIMParallelScheduler"]
    _import_structure["scheduling_ddpm"] = ["DDPMScheduler"]
    _import_structure["scheduling_ddpm_parallel"] = ["DDPMParallelScheduler"]
    _import_structure["scheduling_ddpm_wuerstchen"] = ["DDPMWuerstchenScheduler"]
    _import_structure["scheduling_deis_multistep"] = ["DEISMultistepScheduler"]
    _import_structure["scheduling_dpm_cogvideox"] = ["CogVideoXDPMScheduler"]
    _import_structure["scheduling_dpmsolver_multistep"] = ["DPMSolverMultistepScheduler"]
    _import_structure["scheduling_dpmsolver_multistep_inverse"] = ["DPMSolverMultistepInverseScheduler"]
    _import_structure["scheduling_dpmsolver_singlestep"] = ["DPMSolverSinglestepScheduler"]
    _import_structure["scheduling_edm_dpmsolver_multistep"] = ["EDMDPMSolverMultistepScheduler"]
    _import_structure["scheduling_edm_euler"] = ["EDMEulerScheduler"]
    _import_structure["scheduling_euler_ancestral_discrete"] = ["EulerAncestralDiscreteScheduler"]
    _import_structure["scheduling_euler_discrete"] = ["EulerDiscreteScheduler"]
    _import_structure["scheduling_flow_match_euler_discrete"] = ["FlowMatchEulerDiscreteScheduler"]
    _import_structure["scheduling_flow_match_heun_discrete"] = ["FlowMatchHeunDiscreteScheduler"]
    _import_structure["scheduling_flow_match_lcm"] = ["FlowMatchLCMScheduler"]
    _import_structure["scheduling_heun_discrete"] = ["HeunDiscreteScheduler"]
    _import_structure["scheduling_ipndm"] = ["IPNDMScheduler"]
    _import_structure["scheduling_k_dpm_2_ancestral_discrete"] = ["KDPM2AncestralDiscreteScheduler"]
    _import_structure["scheduling_k_dpm_2_discrete"] = ["KDPM2DiscreteScheduler"]
    _import_structure["scheduling_lcm"] = ["LCMScheduler"]
    _import_structure["scheduling_pndm"] = ["PNDMScheduler"]
    _import_structure["scheduling_repaint"] = ["RePaintScheduler"]
    _import_structure["scheduling_sasolver"] = ["SASolverScheduler"]
    _import_structure["scheduling_scm"] = ["SCMScheduler"]
    _import_structure["scheduling_sde_ve"] = ["ScoreSdeVeScheduler"]
    _import_structure["scheduling_tcd"] = ["TCDScheduler"]
    _import_structure["scheduling_unclip"] = ["UnCLIPScheduler"]
    _import_structure["scheduling_unipc_multistep"] = ["UniPCMultistepScheduler"]
    _import_structure["scheduling_utils"] = ["AysSchedules", "KarrasDiffusionSchedulers", "SchedulerMixin"]
    _import_structure["scheduling_vq_diffusion"] = ["VQDiffusionScheduler"]
```

**File:** tests/schedulers/test_scheduler_euler.py (L9-51)
```python
class EulerDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EulerDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_timestep_type(self):
        timestep_types = ["discrete", "continuous"]
        for timestep_type in timestep_types:
            self.check_over_configs(timestep_type=timestep_type)

    def test_karras_sigmas(self):
        self.check_over_configs(use_karras_sigmas=True, sigma_min=0.02, sigma_max=700.0)

    def test_rescale_betas_zero_snr(self):
        for rescale_betas_zero_snr in [True, False]:
            self.check_over_configs(rescale_betas_zero_snr=rescale_betas_zero_snr)

```

**File:** tests/schedulers/test_scheduler_euler.py (L52-73)
```python
    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = self.num_inference_steps
        scheduler.set_timesteps(num_inference_steps)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample
        return sample
```

**File:** tests/schedulers/test_scheduler_euler.py (L129-146)
```python
    def test_full_loop_no_noise(self):
        sample = self.full_loop()

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 10.0807) < 1e-2
        assert abs(result_mean.item() - 0.0131) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 0.0002) < 1e-2
        assert abs(result_mean.item() - 2.2676e-06) < 1e-3

```

**File:** CONTRIBUTING.md (L288-310)
```markdown
### 9. Adding pipelines, models, schedulers

Pipelines, models, and schedulers are the most important pieces of the Diffusers library.
They provide easy access to state-of-the-art diffusion technologies and thus allow the community to
build powerful generative AI applications.

By adding a new model, pipeline, or scheduler you might enable a new powerful use case for any of the user interfaces relying on Diffusers which can be of immense value for the whole generative AI ecosystem.

Diffusers has a couple of open feature requests for all three components - feel free to gloss over them
if you don't know yet what specific component you would like to add:
- [Model or pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
- [Scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Before adding any of the three components, it is strongly recommended that you give the [Philosophy guide](https://github.com/huggingface/diffusers/blob/main/PHILOSOPHY.md) a read to better understand the design of any of the three components. Please be aware that
we cannot merge model, scheduler, or pipeline additions that strongly diverge from our design philosophy
as it will lead to API inconsistencies. If you fundamentally disagree with a design choice, please
open a [Feedback issue](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=) instead so that it can be discussed whether a certain design
pattern/design choice shall be changed everywhere in the library and whether we shall update our design philosophy. Consistency across the library is very important for us.

Please make sure to add links to the original codebase/paper to the PR and ideally also ping the
original author directly on the PR so that they can follow the progress and potentially help with questions.

If you are unsure or stuck in the PR, don't hesitate to leave a message to ask for a first review or help.
```

**File:** src/diffusers/loaders/__init__.py (L54-92)
```python
_import_structure = {}

if is_torch_available():
    _import_structure["single_file_model"] = ["FromOriginalModelMixin"]
    _import_structure["transformer_flux"] = ["FluxTransformer2DLoadersMixin"]
    _import_structure["transformer_sd3"] = ["SD3Transformer2DLoadersMixin"]
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]
    if is_transformers_available():
        _import_structure["single_file"] = ["FromSingleFileMixin"]
        _import_structure["lora_pipeline"] = [
            "AmusedLoraLoaderMixin",
            "StableDiffusionLoraLoaderMixin",
            "SD3LoraLoaderMixin",
            "AuraFlowLoraLoaderMixin",
            "StableDiffusionXLLoraLoaderMixin",
            "LTXVideoLoraLoaderMixin",
            "LoraLoaderMixin",
            "FluxLoraLoaderMixin",
            "CogVideoXLoraLoaderMixin",
            "CogView4LoraLoaderMixin",
            "Mochi1LoraLoaderMixin",
            "HunyuanVideoLoraLoaderMixin",
            "SanaLoraLoaderMixin",
            "Lumina2LoraLoaderMixin",
            "WanLoraLoaderMixin",
            "HiDreamImageLoraLoaderMixin",
            "SkyReelsV2LoraLoaderMixin",
            "QwenImageLoraLoaderMixin",
        ]
        _import_structure["textual_inversion"] = ["TextualInversionLoaderMixin"]
        _import_structure["ip_adapter"] = [
            "IPAdapterMixin",
            "FluxIPAdapterMixin",
            "SD3IPAdapterMixin",
            "ModularIPAdapterMixin",
        ]

_import_structure["peft"] = ["PeftAdapterMixin"]
```

**File:** src/diffusers/loaders/single_file.py (L52-100)
```python
def load_single_file_sub_model(
    library_name,
    class_name,
    name,
    checkpoint,
    pipelines,
    is_pipeline_module,
    cached_model_config_path,
    original_config=None,
    local_files_only=False,
    torch_dtype=None,
    is_legacy_loading=False,
    disable_mmap=False,
    **kwargs,
):
    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
    else:
        # else we just import it from the library.
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )
    is_tokenizer = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedTokenizer)
        and transformers_version >= version.parse("4.20.0")
    )

    diffusers_module = importlib.import_module(__name__.split(".")[0])
    is_diffusers_single_file_model = issubclass(class_obj, diffusers_module.FromOriginalModelMixin)
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)
    is_diffusers_scheduler = issubclass(class_obj, diffusers_module.SchedulerMixin)

    if is_diffusers_single_file_model:
        load_method = getattr(class_obj, "from_single_file")

        # We cannot provide two different config options to the `from_single_file` method
        # Here we have to ignore loading the config from `cached_model_config_path` if `original_config` is provided
```
