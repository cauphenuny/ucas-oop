# PerflowScheduler 实现计划与仓库对照

日期：2025-10-31

目标：实现一个名为 `PerflowScheduler` 的分段线性流加速调度器并把它与仓库的 loader/注册机制对接，保证与 `DiffusionPipeline` 和现有 `SchedulerMixin`/`ConfigMixin` 接口兼容。

概述
- 功能：在若干时间段（K segments）内使用线性流近似和可选学习的参数 theta 对去噪方向进行修正，从而减少采样步数或提高质量/速度比。
- 所在路径（建议）：
  - `src/diffusers/schedulers/scheduling_perflow.py`（实现类）
  - `src/diffusers/schedulers/__init__.py`（注册导出）
  - `src/diffusers/loaders/single_file_utils.py`（在 `_legacy_load_scheduler` 中添加识别/映射分支，支持单文件 legacy checkpoint）
  - `tests/schedulers/test_scheduler_perflow.py`（单元/集成测试）

与仓库现有 API 的关键点（来自源码对照）
- `SchedulerMixin`（在 `src/diffusers/schedulers/scheduling_utils.py`）定义了加载/保存的行为：
  - `from_pretrained(pretrained_model_name_or_path, subfolder=None, **kwargs)` 应该使用 `load_config` -> `from_config` 的流程。
  - `save_pretrained(save_directory)` 应调用 `save_config` 并支持 PushToHubMixin。
- 其他调度器实现（例如 `EulerDiscreteScheduler`、`DPMSolverMultistepScheduler`）通常继承 `SchedulerMixin` 与 `ConfigMixin`，并实现 `set_timesteps()` 与 `step()`。
- `loaders/single_file_utils._legacy_load_scheduler` 会在从旧 checkpoint 导入时，基于 checkpoint 元数据选择合适的 scheduler 实例，Perflow 需在此处加入兼容分支。

设计要点与契约

1) 类签名

```py
class PerflowScheduler(SchedulerMixin, ConfigMixin):
    def __init__(self, *, num_train_timesteps: int = 1000, segments: int = 4, segment_steps: list | None = None,
                 theta_init: Optional[Tensor] = None, learn_theta: bool = False, **kwargs):
        # 保留 config 的属性
```

实现要求（方法）：
- `set_timesteps(num_inference_steps: int, **kwargs)`：构造内部时间表 `self.timesteps`，支持与其他 scheduler 一致的参数（例如 `num_inference_steps`、`timesteps_spacing`）。
- `step(model_output: Tensor, timestep: int, sample: Tensor, **kwargs) -> SchedulerOutput`：执行 perflow 单步推进，返回 `SchedulerOutput(prev_sample=next_sample)` 或遵循仓库约定的返回类型。
- `state_dict()` / `load_state_dict(state_dict, strict: bool = True)`：保存/加载 learned theta 与必要的 meta（K、segment boundaries、version）。

数值稳定性
- 在 step() 中加入 NaN/Inf 检测与 clipping（例如：如果检测到 NaN，回退到 baseline scheduler 的 step 输出并记录 warning）。
- 对除法或根号操作加 eps 防护。

2) state_dict 格式约定（最小契约）
- 返回格式示例：
```py
{
  "perflow": {
    "theta": Tensor(...),        # 学习到的参数数组，或为空
    "segments": int,
    "segment_boundaries": [t0, t1, ...],
    "version": "0.1"
  },
  "config": { ... }  # 可选，便于 from_pretrained 恢复
}
```
- `load_state_dict` 必须支持 `strict=False`，在不完全匹配时尽量部分加载并记录未匹配项。

3) from_pretrained / Loader 行为
- 按照 `SchedulerMixin.from_pretrained` 的惯例，支持两种主要路径：
  - 仅配置模式（`from_config`）：用户仅传入 config（或用 `from_pretrained` 但仓库仅含 `scheduler_config.json`），返回基于配置的实例（theta 使用初始值）。
  - 带权重的模式（`from_pretrained(..., subfolder="scheduler")`）：若存在 `scheduler/pytorch_model.bin` 或 `scheduler/scheduler_state.safetensors`，则读取 state_dict 并调用 `load_state_dict`。
- 在 `loaders/single_file_utils._legacy_load_scheduler` 中添加对 legacy checkpoint 的识别：当 checkpoint 含有特征键（例如 `perflow.theta` 或自定义 marker）时，构建 config 并返回 PerflowScheduler 的实例或直接从 checkpoint 恢复。

4) 在 `schedulers/__init__.py` 的注册
- 在 `_import_structure` 中加入 `"scheduling_perflow": ["PerflowScheduler"]`，并在 TYPE_CHECKING 节点导入以保持文档与类型兼容。

测试计划
- 单元测试：
  - `test_set_timesteps`：在多种 `num_inference_steps` 下验证 `self.timesteps` 的生成符合预期边界。
  - `test_step_baseline_compat`：在数值稳定输入下，对比 Perflow 在 `learn_theta=False` 时输出与 baseline scheduler（例如 Euler/DDIM）一致或相近（取决实现）。
  - `test_state_dict_roundtrip`：保存 state_dict -> load_state_dict 且在 strict=False 下尝试部分加载。
- 集成测试：
  - `test_pipeline_integration`：在 `DiffusionPipeline` 中替换 scheduler 为 Perflow，在固定 seed 下跑小样本（例如 5 步 vs baseline 50 步），对比输出的形状、无 crash，并记录时间（wall-clock）。

示例配置 schema（可写入 `scheduler_config.json`）
```json
{
  "class": "PerflowScheduler",
  "num_train_timesteps": 1000,
  "segments": 4,
  "segment_boundaries": [999, 750, 500, 250, 0],
  "theta_init": null,
  "learn_theta": false,
  "version": "0.1"
}
```

实现骨架（伪代码，放入 `scheduling_perflow.py`）
```py
from .scheduling_utils import SchedulerMixin, SchedulerOutput
from ..configuration_utils import ConfigMixin
import torch

class PerflowScheduler(SchedulerMixin, ConfigMixin):
    def __init__(self, num_train_timesteps=1000, segments=4, segment_boundaries=None, theta_init=None, learn_theta=False, **kwargs):
        # 保存 config
        self.num_train_timesteps = num_train_timesteps
        self.segments = segments
        self.segment_boundaries = segment_boundaries
        self.learn_theta = learn_theta
        if theta_init is not None:
            self.theta = torch.tensor(theta_init)
        else:
            self.theta = None  # lazy init

    def set_timesteps(self, num_inference_steps: int, **kwargs):
        # 构建 self.timesteps
        pass

    def step(self, model_output, timestep, sample, **kwargs) -> SchedulerOutput:
        # 主要算法：get baseline prediction -> apply per-segment correction using theta -> return next sample
        pass

    def state_dict(self):
        return {"perflow": {"theta": self.theta, "segments": self.segments, "segment_boundaries": self.segment_boundaries}}

    def load_state_dict(self, state_dict, strict=True):
        # 加载并返回 loading_info
        pass
```

Loader（legacy single-file）补充提示
- 在 `loaders/single_file_utils._legacy_load_scheduler` 中添加：
  - 检查 checkpoint 是否包含 `perflow` 相关键（例如 `'perflow.theta'` 或者 `"perflow/theta"`），若包含则：
    - 解析出 segments 与 theta，构造 config -> `PerflowScheduler.from_config(config)` -> `scheduler.load_state_dict(state)`。
  - 如果 checkpoint 没有 perflow 标记，则保持现有逻辑。

开发/调试建议
- 开发时先实现最小可运行的非学习版本（`learn_theta=False`，theta 为 None 或 zeros），保证 `set_timesteps` 与 `step` 的 API 与 base scheduler 匹配并通过单元测试；之后再实现可学习的 theta（需要考虑训练时如何更新 theta——若计划在训练中学习 theta，需设计 `optimizer` 接口或训练脚本）。
- 在实现中尽量复用现有 scheduler 的 helper（例如 computing alphas/sigmas），避免重复实现时间表构造的细节。

下一步（可选，选择一项）
- 生成 `scheduling_perflow.py` 的骨架实现并提交补丁；
- 修改 `schedulers/__init__.py` 注册并在 `loaders/single_file_utils.py` 中加入 legacy mapping（我可先列出将要修改的具体片段并应用补丁）；
- 创建单元测试骨架并运行本地 pytest（需要时我会运行测试并修复语法错误）。

---

文件位置：
`/Users/ycp/Source/Courses/oop/assignments/Project/report0/perflow.md`

如果你想，我现在可以：
- A: 生成 `scheduling_perflow.py` 的类骨架并把文件加入仓库（我会先列出具体改动）；或
- B: 直接在 `loaders/single_file_utils.py` 中添加 legacy 识别的补丁草案；或
- C: 先创建测试用例骨架（`tests/schedulers/test_scheduler_perflow.py`）。

你希望我先做哪一项？
