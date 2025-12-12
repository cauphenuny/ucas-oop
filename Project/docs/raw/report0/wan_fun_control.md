# WanFunControl 实现计划（Wan2.1-Fun-Control）

日期：2025-10-31

目标：在现有 `WanPipeline` 基础上实现 `WanFunControlPipeline`，支持以控制视频（control video）和相机外参（camera extrinsics）注入时序控制信号，使生成的视频具有时空一致的控制效果，并与现有 Loader/LoRA 机制保持兼容。

一、为什么要实现
- 支持用户通过 control video（或 control image + mask）控制时空动作、相机视角或其他额外条件。
- 为下游任务（例如视频合成、视角一致渲染、相机路径驱动合成）提供可复用管道。

二、仓库中相关参考实现（已发现）
- `src/diffusers/pipelines/wan/pipeline_wan.py`（WanPipeline）
  - 主要 pipeline 实现，包含 `prepare_latents()`, `encode_prompt()`, `__call__()` 去噪循环、`prepare_latents()`。
- `src/diffusers/loaders/lora_pipeline.py`（包含 `WanLoraLoaderMixin`）
  - pipeline 的 LoRA 加载能力参考。
- `src/diffusers/models/controlnets/*`（ControlNet 相关模型）
  - 可参考 ControlNet 如何把条件（conditioning）映射成模型可消费的形状并加载/保存。
- `src/diffusers/video_processor/VideoProcessor`（已在 pipeline_wan 中使用）

三、要实现的功能清单（高优先级）
1. 新 pipeline 类 `WanFunControlPipeline`（建议位置：`src/diffusers/pipelines/wan/pipeline_wan_funcontrol.py` 或在 `pipeline_wan.py` 新类）
   - 继承：`DiffusionPipeline, WanLoraLoaderMixin`（或其他必要 mixin）
   - 注册 modules：`tokenizer`, `text_encoder`, `transformer`, `vae`, `scheduler`, `control_adapter`（可选）, `control_video_encoder`（可选）

2. 输入校验与预处理
   - 扩展 `check_inputs()` 支持 control inputs：
     - `control_video`: Optional[Tensor] (B, T, C, H, W) 或 视频路径
     - `control_mask` / `masked_image`: Optional[Tensor]
     - `camera_params`: Optional[List[CameraParam]]（长度 T），约定格式（例如 4x4 矩阵或 (tx,ty,tz, rx,ry,rz)）
   - 若 control_video 提供，则允许降级/兜底：当未提供时退回原 WanPipeline 行为。

3. 视频/相机预处理函数
   - `preprocess_control_video(control_video, target_resolution, device, dtype)`：
     - 解码/resize/normalize/tensorize，将帧转换为与 `vae` 输入一致的格式。
     - 支持 mask/masked_image；返回 (frames_tensor, masks_tensor)
   - `encode_camera_params(camera_params)`：
     - 将相机外参序列编码为与 transformer patch/latent 对齐的向量（例如通过 MLP or linear projection）。

4. control latents 的建模
   - `prepare_control_latents(frames_tensor, masks_tensor=None)`：
     - 使用 `control_video_encoder`（若存在）或 `vae.encode()` 把每帧编码为 control latents（形状应与 transformer 的 patch embedding 对齐）。
     - 支持对 mask 的处理（masked_image -> masked latents）。
     - 返回 `control_latents: Tensor` 和 `control_camera_latents: Optional[Tensor]`（由 `encode_camera_params` 生成）。

5. patchify / transformer 输入融合点
   - 修改或扩展 `transformer` 所使用的 `patchify()` 接口（见 slide 示例），以接受 `control_camera_latents_input`：
     - `x = patch_embedding(x)`
     - `if control_camera_latents_input is not None and control_adapter is not None:`
         `y_camera = control_adapter(control_camera_latents_input)`
         `x = [u + v for u, v in zip(x, y_camera)]`
   - 提供多种融合策略（`add`, `concat`, `cross-attn`），通过 pipeline 参数可切换。

6. 去噪循环中注入控制信号
   - 在 `__call__()` 的 denoising loop：在调用 transformer 时，把 `control_latents` 与 `noise_latents`（或 `latent_model_input`）按配置注入（例如 concat channel, add, 或将 control latents 通过 cross-attention keys/values 传入）。
   - 支持按时间步对 control latents 做插值（若 control video 的帧数与生成帧数不完全一致）。

7. 输出与后处理
   - 通过 `vae.decode()` 解码最终 latents 为视频帧，复用 `VideoProcessor.postprocess_video()`。
   - 如果需要，支持将结果写出为视频文件或返回 frames 列表。

8. Loader / 保存
   - pipeline 的 `save_pretrained()` 与 `from_pretrained()` 应保存/加载 `control_adapter` 的权重（若存在）与配置（control strategy、camera param schema 等）。
   - 支持 `pipeline.load_lora_weights()` 依旧能工作（保持 WanLoraLoaderMixin 兼容性）。

四、异常与降级策略（必需）
- 输入错误：格式、帧数、相机参数长度不匹配 -> 抛出 `ValueError`（并给出详细信息）。
- 视频解码/IO 错误 -> 抛出 `IOError` 并返回友好提示。
- control_adapter 缺失但 control_video 提供 -> 若可用 fallback（如把 control_video 编为 latents 并简单 add 到 patch embeddings），否则抛出错误或记录 warning 并忽略 control。
- OOM -> 提供分批处理（chunked encoding）或提示用户降低分辨率/帧数。

五、实现所需仓库修改（清单）
1. 新文件 / 修改：
   - 新建 `src/diffusers/pipelines/wan/pipeline_wan_funcontrol.py`（或在 `pipeline_wan.py` 中添加 `WanFunControlPipeline` 类）
     - 新增方法：`preprocess_control_video`, `prepare_control_latents`, `encode_camera_params`, `patchify` 调用点、以及在 `__call__()` 中的控制注入逻辑。
   - 修改/扩展模型/adapter：
     - 新增 `ControlAdapter`（轻量 MLP/conv 层）实现（如果需要）或复用现有 controlnet 风格的 `control_adapter`。
   - loaders 与保存：无需修改现有 `WanLoraLoaderMixin`，但需要更新 `from_pretrained` 的文档/示例，保证 control_adapter 权重可被保存/加载。

2. 注册与导出：
   - 在 `src/diffusers/pipelines/wan/__init__.py` 导入并导出 `WanFunControlPipeline`。

3. 测试：
   - `tests/pipelines/wan/test_wan_fun_control.py`：
     - 覆盖：`check_inputs()` 行为、`prepare_control_latents()`、denoising loop 中注入（端到端小样本测试）、保存/加载 control_adapter 权重的 roundtrip。

六、接口与数据契约（草案）
- 输入参数（`__call__` 扩展）示例：
```py
output = pipe(
    prompt=..., 
    control_video=Optional[Tensor] or path,
    control_mask=Optional[Tensor],
    camera_params=Optional[List[CameraParam]],
    control_strategy="add"|"concat"|"cross-attn",
    control_strength=0.0..1.0,
    ...
)
```
- control frame alignment：如果 `num_frames` 不等于 `len(control_frames)`，默认做线性插值或 nearest-neighbor 对齐，或抛出异常（由 `control_align` 参数控制）。

七、性能与可扩展性建议
- 预计算 control latents：提供 `precompute_control_latents()` 接口，允许用户在内存或磁盘中缓存 control latents，避免每次推理重复编码。
- 流式/分块处理：当输入视频较长时，支持按时间窗口分块生成并拼接输出，以避免 OOM。
- dtype 与设备：强制/统一 dtype（float16/float32），并提供自动迁移/警告。

八、示例使用（草案）
```py
from diffusers import WanFunControlPipeline, AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae")
pipe = WanFunControlPipeline.from_pretrained(model_id, vae=vae)
pipe.to("cuda")

output = pipe(
    prompt="a person walking around a table",
    control_video=control_video_tensor,  # shape (B, T, C, H, W)
    camera_params=camera_params_list,
    control_strategy="cross-attn",
    control_strength=0.8,
    num_frames=48,
    num_inference_steps=50,
)

# save video
pipe.save_pretrained("./wan_fun_control_model")
```

九、测试计划（更具体）
- 单元测试：
  - `test_check_inputs_invalid_shapes`：控制输入维度/长度错误时抛错。
  - `test_prepare_control_latents_shapes`：确保 control latents 形状符合 transformer 期待。
  - `test_pipeline_forward_with_control`：最小模型与随机权重下的端到端前向（小尺寸、少帧）不崩溃并输出正确形状。
- 集成实验：
  - 与现有 WanPipeline 在相同 seed 下比较，验证 control 信号改变了输出（定性评估）。

十、后续工作与开放问题
- control_adapter 的设计：简单线性投影 vs 更复杂的时序 encoder（Transformer/Conv3D）。需依据质量/性能权衡确定实现复杂度。
- 相机外参编码细节：需确认输入格式（4x4 矩阵或 6-DoF）并实现数值稳定的编码器。
- 若需要把 control adapter 的参数训练出来，需设计训练流程（固定 pipeline 其余部分只训练 control_adapter 的参数或联合训练）。

---

文件路径：`/Users/ycp/Source/Courses/oop/assignments/Project/report0/wan_fun_control.md`

我已经把这份实现计划保存为上面的文件。下一步我可以：
- A) 生成 `pipeline_wan_funcontrol.py` 的类骨架并提交补丁；或
- B) 先实现 `preprocess_control_video` / `prepare_control_latents` 的实用函数并加单元测试骨架；或
- C) 在 `pipeline_wan.py` 中添加示例注入点（小补丁）以便后续扩展。

你想让我先做哪一项？
