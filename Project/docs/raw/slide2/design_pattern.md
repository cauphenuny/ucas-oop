# Diffusers 代码库设计模式分析与重构建议

## 一、现有设计模式分析

### 1. **Mixin 模式** ✅ 已实现
代码库广泛使用 Mixin 模式来组合功能：

- **ConfigMixin**: 提供配置管理功能 [1](#0-0) 
- **PushToHubMixin**: 提供 Hub 上传功能
- **SchedulerMixin**: 调度器基类 [2](#0-1) 
- **ModelMixin**: 模型基类 [3](#0-2) 

**优点**: 实现了良好的功能复用和组合
**改进空间**: 部分 Mixin 职责过重，可以进一步拆分

### 2. **工厂方法模式** ✅ 已实现
通过 `from_pretrained()` 和 `from_config()` 类方法创建对象： [4](#0-3) [5](#0-4) 

**优点**: 统一的对象创建接口
**改进建议**: 可以引入抽象工厂模式来更好地组织不同类型的对象创建

### 3. **模板方法模式** ✅ 已实现
`ConfigMixin` 定义了配置加载和保存的模板流程： [6](#0-5) 

### 4. **策略模式** ✅ 已实现
不同的调度器（DDPM, DDIM, Euler 等）实现相同的接口： [7](#0-6) 

**优点**: 可以轻松切换不同的调度算法
**改进建议**: 可以添加调度器选择器（Selector）来根据任务自动推荐合适的调度器

### 5. **观察者模式** ✅ 已实现
通过回调系统实现： [8](#0-7) [9](#0-8) 

**优点**: 允许在推理过程中插入自定义逻辑
**改进建议**: 可以添加更细粒度的事件系统

### 6. **注册表模式** ✅ 部分实现
Pipeline 使用 `register_modules()` 注册组件： [10](#0-9) 

### 7. **装饰器模式** ✅ 已实现
`register_to_config` 装饰器自动注册配置参数： [11](#0-10) 

### 8. **不可变对象模式** ✅ 已实现
`FrozenDict` 确保配置不被意外修改： [12](#0-11) 

### 9. **适配器模式** ✅ 已实现
Loaders 目录下的各种加载器适配不同格式：

- LoRA 加载器 [13](#0-12) 
- IP-Adapter 等

## 二、可以添加的设计模式

### 1. **建造者模式 (Builder Pattern)** ⭐ 强烈推荐

**当前问题**: Pipeline 的构造涉及多个组件（unet, vae, scheduler, text_encoder 等），参数众多，容易出错。

**建议实现**:
```python
class PipelineBuilder:
    """流式构建 Pipeline"""
    def with_unet(self, unet) -> Self
    def with_vae(self, vae) -> Self
    def with_scheduler(self, scheduler) -> Self
    def with_text_encoder(self, encoder) -> Self
    def build() -> DiffusionPipeline
```

**优点**:
- 更清晰的 API
- 支持可选组件的灵活配置
- 便于验证配置的完整性

### 2. **外观模式 (Facade Pattern)** ⭐ 推荐

**当前问题**: 用户需要了解太多内部细节（调度器类型、模型架构、LoRA 加载等）

**建议实现**:
```python
class SimpleDiffusionFacade:
    """简化的高级接口"""
    @staticmethod
    def quick_setup(task_type: str, quality: str) -> DiffusionPipeline
        # 根据任务类型自动选择最佳配置
    
    @staticmethod  
    def from_checkpoint_with_lora(base_model: str, lora_path: str) -> DiffusionPipeline
        # 一步完成复杂的配置
```

### 3. **责任链模式 (Chain of Responsibility)** ⭐ 推荐

**当前问题**: 回调系统可以更系统化地处理各种钩子和处理器

**建议实现**:
```python
class ProcessingChain:
    """处理链，用于图像预处理、后处理等"""
    def add_handler(self, handler: Handler) -> Self
    def process(self, data: Any) -> Any
```

**应用场景**:
- 图像预处理管道（归一化、调整大小、增强等）
- 后处理管道（去噪、锐化、色彩校正等）
- 权重加载和转换的链式处理

### 4. **抽象工厂模式 (Abstract Factory)** 推荐

**当前问题**: 创建一套相关对象（如特定架构的 unet + vae + scheduler）时缺乏统一接口

**建议实现**:
```python
class ModelArchitectureFactory(ABC):
    @abstractmethod
    def create_unet(self) -> UNet2DConditionModel
    
    @abstractmethod  
    def create_vae(self) -> AutoencoderKL
    
    @abstractmethod
    def create_scheduler(self) -> SchedulerMixin

class StableDiffusionFactory(ModelArchitectureFactory):
    """为 SD 1.5/2.0 创建一套组件"""
    
class StableDiffusionXLFactory(ModelArchitectureFactory):
    """为 SDXL 创建一套组件"""
```

### 5. **状态模式 (State Pattern)** 推荐

**当前问题**: Pipeline 的不同状态（训练、推理、CPU offload、量化等）管理比较分散

**建议实现**:
```python
class PipelineState(ABC):
    @abstractmethod
    def to_inference(self, pipeline) -> None
    
    @abstractmethod
    def to_training(self, pipeline) -> None

class InferenceState(PipelineState):
    """推理状态的行为"""
    
class OffloadedState(PipelineState):
    """CPU offload 状态的行为"""
```

**优点**:
- 集中管理状态转换逻辑
- 避免 if-else 分支过多
- 更容易添加新状态

### 6. **命令模式 (Command Pattern)** 可选

**应用场景**: 
- 批量操作（批量保存、批量转换）
- 可撤销操作
- 操作历史记录

```python
class PipelineCommand(ABC):
    @abstractmethod
    def execute(self, pipeline: DiffusionPipeline) -> Any
    
    @abstractmethod
    def undo(self, pipeline: DiffusionPipeline) -> None

class LoadLoRACommand(PipelineCommand):
    """加载 LoRA 的命令，可撤销"""
```

### 7. **原型模式 (Prototype Pattern)** 可选

**应用场景**: 快速创建 Pipeline 的变体

```python
class DiffusionPipeline:
    def clone(self, **overrides) -> Self:
        """克隆 pipeline 并修改部分组件"""
```

### 8. **组合模式 (Composite Pattern)** 可选

**应用场景**: 更好地组织模型组件的层次结构

```python
class ModelComponent(ABC):
    def forward(self, *args) -> Any
    def parameters(self) -> Iterator[Parameter]

class CompositeModel(ModelComponent):
    """可以包含多个子组件的复合模型"""
    def add_component(self, component: ModelComponent)
    def remove_component(self, component: ModelComponent)
```

## 三、具体重构建议

### 重构优先级

**高优先级** (立即实施):
1. **建造者模式**: 简化复杂 Pipeline 的构建
2. **外观模式**: 提供简单易用的高级 API
3. **责任链模式**: 重构预处理和后处理逻辑

**中优先级** (逐步实施):
4. **抽象工厂模式**: 统一不同架构的组件创建
5. **状态模式**: 更好地管理 Pipeline 状态

**低优先级** (按需实施):
6. **命令模式**: 支持复杂操作和撤销功能
7. **原型模式**: 便于实验和快速原型开发

### 重构注意事项

1. **保持向后兼容**: 新设计应作为现有 API 的补充，而非替换
2. **渐进式重构**: 先在新功能中试用，验证效果后再逐步推广
3. **文档和示例**: 为新模式提供清晰的文档和使用示例
4. **性能考虑**: 确保设计模式不会引入显著的性能开销
5. **测试覆盖**: 为新模式编写全面的测试

## Notes

Diffusers 代码库已经采用了许多优秀的设计模式，特别是 Mixin 模式、工厂方法模式和策略模式的使用非常恰当。主要改进空间在于：

1. **简化复杂度**: 通过建造者模式和外观模式降低使用门槛
2. **增强可扩展性**: 通过抽象工厂和状态模式提高架构的灵活性
3. **改善可维护性**: 通过责任链模式使处理流程更加模块化

这些改进不需要大规模重写，可以作为现有架构的自然延伸，逐步引入到代码库中。

### Citations

**File:** src/diffusers/configuration_utils.py (L56-86)
```python
class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)

```

**File:** src/diffusers/configuration_utils.py (L88-103)
```python
class ConfigMixin:
    r"""
    Base class for all configuration classes. All configuration parameters are stored under `self.config`. Also
    provides the [`~ConfigMixin.from_config`] and [`~ConfigMixin.save_config`] methods for loading, downloading, and
    saving classes that inherit from [`ConfigMixin`].

    Class attributes:
        - **config_name** (`str`) -- A filename under which the config should stored when calling
          [`~ConfigMixin.save_config`] (should be overridden by parent class).
        - **ignore_for_config** (`List[str]`) -- A list of attributes that should not be saved in the config (should be
          overridden by subclass).
        - **has_compatibles** (`bool`) -- Whether the class has compatible classes (should be overridden by subclass).
        - **_deprecated_kwargs** (`List[str]`) -- Keyword arguments that are deprecated. Note that the `init` function
          should only have a `kwargs` argument if at least one argument is deprecated (should be overridden by
          subclass).
    """
```

**File:** src/diffusers/configuration_utils.py (L191-225)
```python
    def from_config(
        cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs
    ) -> Union[Self, Tuple[Self, Dict[str, Any]]]:
        r"""
        Instantiate a Python class from a config dictionary.

        Parameters:
            config (`Dict[str, Any]`):
                A config dictionary from which the Python class is instantiated. Make sure to only load configuration
                files of compatible classes.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it is loaded) and initiate the Python class.
                `**kwargs` are passed directly to the underlying scheduler/model's `__init__` method and eventually
                overwrite the same named arguments in `config`.

        Returns:
            [`ModelMixin`] or [`SchedulerMixin`]:
                A model or scheduler object instantiated from a config dictionary.

        Examples:

        ```python
        >>> from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler

        >>> # Download scheduler from huggingface.co and cache.
        >>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

        >>> # Instantiate DDIM scheduler class with same config as DDPM
        >>> scheduler = DDIMScheduler.from_config(scheduler.config)

        >>> # Instantiate PNDM scheduler class with same config as DDPM
        >>> scheduler = PNDMScheduler.from_config(scheduler.config)
        ```
```

**File:** src/diffusers/configuration_utils.py (L291-460)
```python
    @classmethod
    @validate_hf_hub_args
    def load_config(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        return_unused_kwargs=False,
        return_commit_hash=False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""
        Load a model or scheduler configuration.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing model weights saved with
                      [`~ConfigMixin.save_config`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False):
                Whether unused keyword arguments of the config are returned.
            return_commit_hash (`bool`, *optional*, defaults to `False):
                Whether the `commit_hash` of the loaded configuration are returned.

        Returns:
            `dict`:
                A dictionary of all the parameters stored in a JSON configuration file.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", None)
        user_agent = kwargs.pop("user_agent", {})
        dduf_entries: Optional[Dict[str, DDUFEntry]] = kwargs.pop("dduf_entries", None)

        user_agent = {**user_agent, "file_type": "config"}
        user_agent = http_user_agent(user_agent)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if cls.config_name is None:
            raise ValueError(
                "`self.config_name` is not defined. Note that one should not load a config from "
                "`ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
            )
        # Custom path for now
        if dduf_entries:
            if subfolder is not None:
                raise ValueError(
                    "DDUF file only allow for 1 level of directory (e.g transformer/model1/model.safetentors is not allowed). "
                    "Please check the DDUF structure"
                )
            config_file = cls._get_config_file_from_dduf(pretrained_model_name_or_path, dduf_entries)
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            ):
                config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.config_name)):
                # Load from a PyTorch checkpoint
                config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `token` or log in with `hf auth login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
                )
            except HfHubHTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )
        try:
            config_dict = cls._dict_from_json_file(config_file, dduf_entries=dduf_entries)

            commit_hash = extract_commit_hash(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")

        if not (return_unused_kwargs or return_commit_hash):
            return config_dict

        outputs = (config_dict,)

        if return_unused_kwargs:
            outputs += (kwargs,)

        if return_commit_hash:
            outputs += (commit_hash,)

        return outputs
```

**File:** src/diffusers/configuration_utils.py (L654-701)
```python
def register_to_config(init):
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )

        # Take note of the parameters that were not present in the loaded config
        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))

        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init
```

**File:** src/diffusers/schedulers/scheduling_utils.py (L34-50)
```python
class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15

```

**File:** src/diffusers/schedulers/scheduling_utils.py (L75-89)
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
```

**File:** src/diffusers/schedulers/scheduling_utils.py (L95-154)
```python
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ) -> Self:
        r"""
        Instantiate a scheduler from a pre-defined JSON configuration file in a local directory or Hub repository.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the scheduler
                      configuration saved with [`~SchedulerMixin.save_pretrained`].
            subfolder (`str`, *optional*):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.

        > [!TIP] > To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in
        with `hf > auth login`. You can also activate the special >
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a >
        firewalled environment.

        """
        config, kwargs, commit_hash = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            **kwargs,
        )
        return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
```

**File:** src/diffusers/models/modeling_utils.py (L233-250)
```python
class ModelMixin(torch.nn.Module, PushToHubMixin):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the model configuration and provides methods for loading, downloading and
    saving models.

        - **config_name** ([`str`]) -- Filename to save a model to when calling [`~models.ModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = None
    _no_split_modules = None
    _keep_in_fp32_modules = None
    _skip_layerwise_casting_patterns = None
    _supports_group_offloading = True
```

**File:** src/diffusers/callbacks.py (L7-44)
```python
class PipelineCallback(ConfigMixin):
    """
    Base class for all the official callbacks used in a pipeline. This class provides a structure for implementing
    custom callbacks and ensures that all callbacks have a consistent interface.

    Please implement the following:
        `tensor_inputs`: This should return a list of tensor inputs specific to your callback. You will only be able to
        include
            variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
        `callback_fn`: This method defines the core functionality of your callback.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(self, cutoff_step_ratio=1.0, cutoff_step_index=None):
        super().__init__()

        if (cutoff_step_ratio is None and cutoff_step_index is None) or (
            cutoff_step_ratio is not None and cutoff_step_index is not None
        ):
            raise ValueError("Either cutoff_step_ratio or cutoff_step_index should be provided, not both or none.")

        if cutoff_step_ratio is not None and (
            not isinstance(cutoff_step_ratio, float) or not (0.0 <= cutoff_step_ratio <= 1.0)
        ):
            raise ValueError("cutoff_step_ratio must be a float between 0.0 and 1.0.")

    @property
    def tensor_inputs(self) -> List[str]:
        raise NotImplementedError(f"You need to set the attribute `tensor_inputs` for {self.__class__}")

    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs) -> Dict[str, Any]:
        raise NotImplementedError(f"You need to implement the method `callback_fn` for {self.__class__}")

    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        return self.callback_fn(pipeline, step_index, timestep, callback_kwargs)

```

**File:** src/diffusers/callbacks.py (L46-67)
```python
class MultiPipelineCallbacks:
    """
    This class is designed to handle multiple pipeline callbacks. It accepts a list of PipelineCallback objects and
    provides a unified interface for calling all of them.
    """

    def __init__(self, callbacks: List[PipelineCallback]):
        self.callbacks = callbacks

    @property
    def tensor_inputs(self) -> List[str]:
        return [input for callback in self.callbacks for input in callback.tensor_inputs]

    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        """
        Calls all the callbacks in order with the given arguments and returns the final callback_kwargs.
        """
        for callback in self.callbacks:
            callback_kwargs = callback(pipeline, step_index, timestep, callback_kwargs)

        return callback_kwargs

```

**File:** src/diffusers/pipelines/pipeline_utils.py (L207-220)
```python
    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)
```

**File:** src/diffusers/loaders/lora_base.py (L68-102)
```python
def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False, adapter_names=None):
    """
    Fuses LoRAs for the text encoder.

    Args:
        text_encoder (`torch.nn.Module`):
            The text encoder module to set the adapter layers for. If `None`, it will try to get the `text_encoder`
            attribute.
        lora_scale (`float`, defaults to 1.0):
            Controls how much to influence the outputs with the LoRA parameters.
        safe_fusing (`bool`, defaults to `False`):
            Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
        adapter_names (`List[str]` or `str`):
            The names of the adapters to use.
    """
    merge_kwargs = {"safe_merge": safe_fusing}

    for module in text_encoder.modules():
        if isinstance(module, BaseTunerLayer):
            if lora_scale != 1.0:
                module.scale_layer(lora_scale)

            # For BC with previous PEFT versions, we need to check the signature
            # of the `merge` method to see if it supports the `adapter_names` argument.
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. "
                    "Please upgrade to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)

```
