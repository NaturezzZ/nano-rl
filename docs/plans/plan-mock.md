# nano-rl Mock Runtime 设计方案

> 目标：为 nano-rl 设计一套可配置、可组合、可验证的 mock runtime。
> 所有需要 GPU、权重 load/store、外部数据获取的部分都必须有对应 mock，
> 且 mock 与真实实现走同一 controller、同一 protocol、同一 Ray actor 边界。

---

## 0. 背景与核心诉求

当前仓库已经有：

- YAML-first 的 `RuntimeConfig` / `LaunchConfig`；
- 数量式 GPU topology 和 `ResolvedGpuPlan`；
- CPU-only `GpuLeaseManagerCore` 和 shared GPU residency FSM；
- `RolloutBackend` 协议与 `VllmRolloutBackend`；
- `TrainerBackend` 协议与 `Fsdp2TrainerBackend` / `FakeTrainerBackend`；
- `WeightRegistryActorCore`、`WeightTransferPlanner`、`SampleQueueActorCore`；
- local smoke runtime，但它主要通过 deterministic role adapter 组合起来，还不是一套完整的 YAML 可选 mock backend 体系。

这份方案的目标不是另写一个绕过主链路的测试脚本，而是把 mock 变成正式 runtime backend：

1. `rollout.backend: vllm | huggingface | mock`；
2. `trainer.backend: fsdp2 | mock`；
3. `data.source_type` 支持真实 HDFS/HDFS-FUSE 以及 mock data；
4. 权重 load/store/transfer 支持真实 checkpoint/artifact 以及 mock memory/file manifest；
5. reward model 支持真实 reward function 以及 deterministic mock reward；
6. 所有需要 GPU 的行为都收敛在 high-level backend 方法里 mock，不新增第一阶段必做的 `DeviceBackend`；
7. mock 可以全量启用，也可以与真实模块混用。

第一阶段的 MVP mock 面只包括五类 high-level 模块：

- `RolloutBackend`
- `TrainerBackend`
- `PromptSource` / dataloader
- `WeightStore` / checkpoint
- `RewardBackend` / reward model

GPU lease、GPU topology、shared GPU toggle 是这些 backend 必须遵守的调度不变量，不是单独被 mock
的一层 backend。

---

## 1. 设计目标与非目标

### 1.1 设计目标

- **同接口替换**：mock backend 必须实现生产 backend 的同一协议，不能引入只在 mock 下存在的数据通路。
- **YAML 可选择**：每个需要替换的模块都能在 YAML 中选择真实实现或 mock 实现。
- **CPU-only 可跑通**：全 mock 配置不 import vLLM、torch、transformers，不要求 CUDA、checkpoint、HDFS、Ray GPU。
- **保留关键不变量**：即使是 mock，也必须保留 GPU lease、policy_version、weight checksum、sample TTL、policy lag、queue backpressure、rank/replica assignment。
- **确定性**：同 seed、同 config、同 prompt 应该产生同样的 token、logprob、reward、loss、weight checksum。
- **可注入故障**：mock 应能模拟 OOM、stale lease、weight corruption、data EOF、slow load、trainer failure 等关键错误。
- **可混用**：例如 `rollout.backend: mock` + `trainer.backend: fsdp2`，或 `data.source_type: mock_generated` + `rollout.backend: vllm`。
- **可观测**：mock 事件应进入已有 `MetricsActorCore` / health event / result payload，便于调试和测试断言。

### 1.2 非目标

- mock 不追求 vLLM 生成质量，不模拟真实 decoding kernel。
- mock 不追求 FSDP2 数值等价，不验证真实分布式训练收敛。
- mock 不绕过 `GpuLeaseManagerCore`，不允许在没有正确 lease 的情况下执行 rollout/train 方法。
- mock 不替代小模型真实端到端测试；它用于结构、协议、调度、版本、失败恢复的快速验证。
- v0.1 不因为 mock 引入多机、Kubernetes、Slurm、SSH 或其他执行层。

---

## 2. 总体原则

### 2.1 Controller 不分叉

不新增 `MockController`。生产链路与 mock 链路都走：

```text
main.py
  -> load YAML
  -> RuntimeConfig / LaunchConfig
  -> RayDriver / ControllerCore
  -> RolloutManager
  -> RolloutBackend
  -> SampleQueue
  -> TrainerBackend
  -> WeightRegistry / WeightTransfer
```

mock 只替换叶子模块和外部边界，不替换 loop 语义源。

### 2.2 Mock 不绕过 protocol

所有 mock 输出都必须使用现有协议模型或其扩展：

- rollout 输出：`GenerationOutput` -> `SampleRecord`；
- trainer 输出：`OptimizerStepResult` / `TrainStats`；
- 权重元数据：`WeightMeta`；
- 权重传输：`WeightTransferPlan` / `WeightShardSource`；
- 样本队列：`SampleRef` / `TrainBatch`；
- GPU 访问：`GpuLease`。

### 2.3 Mock 也要严格校验 GPU lease

即使没有真实 CUDA，mock rollout/trainer/offload 仍必须检查：

- lease role 是否匹配；
- lease gpu_id 是否匹配；
- holder_id 是否匹配；
- lease_epoch 是否仍是当前 epoch；
- shared GPU 是否处在正确 active role。

这样可以用 CPU-only 测试捕获生产中最危险的调度错误。

### 2.4 Mock 配置应默认 deterministic strict

默认行为：

- deterministic seed；
- no sleep；
- no random failure；
- strict lease check；
- strict version check；
- strict artifact/data bypass check。

失败、延迟、随机性必须显式配置。

---

## 3. 需要 mock 的模块矩阵

| 范围 | 生产实现 | Mock 实现 | 统一接口 | 必须保留的不变量 |
| --- | --- | --- | --- | --- |
| Rollout GPU compute | vLLM engine or direct Hugging Face Transformers generation | deterministic mock rollout engine | `RolloutBackend` | rollout lease、active weight、policy_version、checksum、tokens/logprobs |
| Trainer GPU compute | PyTorch FSDP2 | deterministic mock trainer | `TrainerBackend` | trainer lease、rank/world_size、train_step、weight export |
| Offload / hydrate | vLLM sleep、FSDP2 model/optimizer offload | backend 内 state-only transition | `RolloutBackend` / `TrainerBackend` + `GpuResidencyManagerCore` | shared GPU role exclusivity |
| Weight store | checkpoint dir、artifact/manifest path | memory store 或 local JSON manifest | `WeightStore` | version、parent、checksum、format、status |
| Weight transfer | objectref、locality-aware checkpoint | mock objectref、mock manifest、mock reshard | `WeightTransferPlanner` + `WeightMaterializer` | per-replica source、target_gpu_ids、target_worker_ids |
| Data fetch | HDFS URI、HDFS-FUSE path | inline prompts、generated prompts、local mock jsonl | `PromptSource` | prompt order、prompt id、epoch/offset |
| Reward | user reward / CPU actor pool | deterministic reward | `RewardBackend` 或现有 role surface | reward_source、reward value |
| Artifact validation | HDFS CLI/path probes | mock source validation | `validate_input_artifacts` | 不对 mock source 调真实 HDFS |

这里刻意不把 GPU device 抽成第一阶段 mock 模块。GPU ownership、active role、lease epoch 仍由现有
`GpuLeaseManagerCore` 与 `GpuResidencyManagerCore` 负责；mock backend 只是在自己的 high-level
方法入口校验 lease。也就是说，第一阶段 mock 的对象是 rollout / trainer / data / weight / reward
这些业务边界，而不是 CUDA device 本身。

---

## 4. YAML 设计

### 4.1 顶层 mock 开关

新增可选顶层段：

```yaml
mock:
  enabled: true
  seed: 42
  strict:
    leases: true
    versions: true
    no_gpu_imports: true
    no_external_data_probe: true
  timing:
    rollout_sleep_ms: 0
    trainer_sleep_ms: 0
    weight_io_sleep_ms: 0
  failure_injection:
    enabled: false
    fail_after_rollout_requests: null
    fail_on_train_step: null
    fail_weight_version: null
    corrupt_weight_checksum: false
    stale_lease_epoch_delta: 0
```

含义：

- `mock.enabled` 只是便捷全局标记，不强制所有模块都使用 mock；
- 各模块仍以自己的 `backend` / `source_type` / `store.backend` 为准；
- `strict.no_gpu_imports: true` 时，任何 mock-only 配置不应 import `vllm`、`torch`、`transformers`；
- failure injection 仅在 mock backend 内生效，生产 backend 不读取这些字段。

### 4.2 Rollout backend

把现有：

```yaml
rollout:
  backend: vllm
```

扩展为：

```yaml
rollout:
  backend: mock
  mock:
    response_template: "{prompt} :: response@v{policy_version}"
    tokenization: sha256_bytes
    max_response_tokens: 16
    logprob_mode: linear
    finish_reason: stop
    include_policy_segments: false
```

生产配置仍使用：

```yaml
rollout:
  backend: vllm
  vllm:
    dtype: bf16
    max_model_len: 4096
    trust_remote_code: false
    engine_kwargs: {}
    sampling_params:
      temperature: 1.0
      top_p: 1.0
```

校验规则：

- `rollout.backend=vllm` 时允许 `vllm` 段，忽略或禁止 `mock` 段由 strict 配置决定；
- `rollout.backend=mock` 时不要求 vLLM dependency，不构造真实 engine；
- mock 仍必须执行 `activate_weight()` 后才能 `generate()`；
- mock `generate()` 的 `target_policy_version` 必须等于 active weight version。

### 4.3 Trainer backend

把现有：

```yaml
trainer:
  backend: fsdp2
  fsdp2:
    mixed_precision: bf16
    sharding: full_shard
```

扩展为：

```yaml
trainer:
  backend: mock
  global_batch_size: 1024
  mock:
    initial_loss: 1.0
    loss_decay: reciprocal
    export_format: vllm_compatible
    checkpoint_dir: null
    optimizer_state: tracked
    require_rank0_export: true
```

生产配置仍使用：

```yaml
trainer:
  backend: fsdp2
  global_batch_size: 1024
  checkpoint_dir: /tmp/nano-rl-checkpoints
  fsdp2:
    mixed_precision: bf16
    sharding: full_shard
```

校验规则：

- `trainer.backend=fsdp2` 时 `fsdp2` 必填；
- `trainer.backend=mock` 时 `fsdp2` 不必填，`checkpoint_dir` 可选；
- mock trainer 仍按 rank 创建，每个 rank 只接受自己的 trainer lease；
- rank 0 才能 export weight；
- mock export 的 `WeightMeta.parent_version`、`trainer_step`、`checksum` 必须可追溯。

### 4.4 Data source

把现有：

```yaml
data:
  source_type: hdfs_uri
  data_path: hdfs://namenode/datasets/prompts.jsonl
  prompt_column: prompt
```

扩展为：

```yaml
data:
  source_type: mock_inline
  prompt_column: prompt
  mock_inline:
    prompts:
      - hello nano-rl
      - explain gpu lease
    repeat: 1
```

或：

```yaml
data:
  source_type: mock_generated
  prompt_column: prompt
  mock_generated:
    count: 128
    template: "prompt-{index}-seed-{seed}"
    start_index: 0
```

也支持本地 mock jsonl：

```yaml
data:
  source_type: mock_jsonl
  data_path: ./tests/fixtures/prompts.jsonl
  prompt_column: prompt
```

建议 source enum：

- `hdfs_uri`：生产 HDFS CLI probe；
- `hdfs_fuse_path`：生产 local path under mount root；
- `mock_inline`：YAML 内直接给 prompts；
- `mock_generated`：按 count/template 生成；
- `mock_jsonl`：读本地 jsonl，不经过 HDFS policy。

校验规则：

- `mock_inline` 不需要 `data_path`；
- `mock_generated` 不需要 `data_path`；
- `mock_jsonl` 的 `data_path` 必须是相对 repo 或绝对本地路径；
- `validate_input_artifacts()` 不应对 mock source 调 HDFS；
- 所有 data source 输出统一为 `PromptRecord`，再由 dataloader/driver 喂给 `RolloutManagerCore.enqueue_prompts()`。

### 4.5 Weight transfer 与 weight store

现有 `weight_transfer.method` 表达传输策略，但还没有独立的 weight store 接口。建议新增：

```yaml
weight_transfer:
  method: mock_manifest
  allow_rollout_only_artifact_pull: true
  max_versions_in_flight: 2
  store:
    backend: mock_memory
    manifest_dir: null
    checksum_mode: semantic_sha256
```

生产配置：

```yaml
weight_transfer:
  method: locality_aware_checkpoint
  allow_rollout_only_artifact_pull: true
  max_versions_in_flight: 2
  store:
    backend: checkpoint
```

建议枚举：

- `method: objectref`：现有 Ray object store path；
- `method: locality_aware_checkpoint`：现有生产默认；
- `method: mock_objectref`：不放大 tensor，只生成 object ref key 和 mock payload；
- `method: mock_manifest`：生成本地或内存 manifest，rollout activation 从 manifest materialize。

建议 store backend：

- `checkpoint`：生产 checkpoint/artifact path；
- `mock_memory`：进程内字典保存 `version_id -> MockWeightPayload`；
- `mock_filesystem`：写小 JSON manifest，便于人工检查和跨进程 Ray actor 读取。

校验规则：

- 全 mock 单进程可以用 `mock_memory`；
- `run.start_ray_actors=true` 且 mock actor 跨进程时，优先用 `mock_filesystem` 或 Ray objectref，避免各 actor memory 不共享；
- `WeightRegistryActorCore` 仍只保存 `WeightMeta` 与 activation status，不直接保存 mock payload；
- `WeightStore` 只负责 payload/manifest 读写，不能替代 registry。

### 4.6 GPU lease 与 topology

mock runtime 不新增 `runtime.device` 配置，也不引入第一阶段必做的 `DeviceBackend`。GPU topology
继续使用现有数量式配置：

```yaml
runtime:
  local:
    num_gpus: 8
  ray:
    gpu_manager:
      topology:
        rollout_only_gpus: 4
        shared_gpus: 4
```

规则：

- `runtime.local.num_gpus` 在 mock 配置中仍表示 resolved GPU plan 使用的逻辑 GPU 数；
- Ray custom resources 仍按 `rollout_gpu_i` / `train_gpu_i` 展开；
- long-lived actors 仍保持 `num_gpus=0`；
- mock backend 不申请 Ray `num_gpus=1`；
- `MockRolloutBackend` / `MockTrainerBackend` 必须只接受 `GpuLease`，不能接受裸 gpu_id 后直接执行；
- shared GPU toggle 继续由 `GpuResidencyManagerCore` 驱动，mock backend 只实现 hydrate/offload 的状态副作用。

---

## 5. 建议新增接口

### 5.1 PromptSource

新增 `nano_rl/runtime/data_sources.py`：

```python
class PromptSource(Protocol):
    def iter_prompts(self, *, limit: int | None = None) -> Iterable[PromptRecord]:
        ...

class PromptRecord(BaseModel):
    prompt_id: str
    prompt: str
    source_type: str
    offset: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

实现：

- `HdfsUriPromptSource`
- `HdfsFusePromptSource`
- `MockInlinePromptSource`
- `MockGeneratedPromptSource`
- `MockJsonlPromptSource`

Controller 或 driver 不关心来源，只消费 `PromptRecord.prompt`。

### 5.2 GPU lease 不新增独立接口

第一阶段不新增 `nano_rl/runtime/device.py`，也不要求 `CudaDeviceBackend` / `MockDeviceBackend`。

原因：

- 现有 `GpuLeaseManagerCore` 已经负责 GPU ownership、active role、holder、lease epoch；
- 现有 `GpuResidencyManagerCore` 已经负责 shared GPU rollout/train toggle 的状态机；
- rollout/trainer backend 方法本身已经有 `lease` 参数，直接在 high-level backend 内校验即可；
- 额外抽一个 device 层会把 MVP 复杂度推高，但没有增加新的 mock 覆盖面。

未来如果真实 backend 中出现大量重复的 CUDA side effect，例如 `CUDA_VISIBLE_DEVICES`、
memory snapshot、device synchronize、OOM 注入、allocator 统计，再考虑抽出可选 `DeviceRuntime`
工具层。它不应是 mock runtime 的第一阶段接口。

### 5.3 RolloutBackend

现有 `RolloutBackend` 协议保留：

```python
class RolloutBackend(Protocol):
    @property
    def active_weight(self) -> WeightMeta | None: ...

    def activate_weight(
        self,
        meta: WeightMeta,
        *,
        lease: GpuLease | Sequence[GpuLease],
        transfer_source: WeightShardSource | None = None,
    ) -> None: ...

    def generate(
        self,
        *,
        prompt: str,
        target_policy_version: int,
        request_metadata: Mapping[str, Any] | None = None,
        lease: GpuLease | Sequence[GpuLease],
        request_id: str | None = None,
    ) -> GenerationOutput: ...
```

新增 `MockRolloutBackend`：

- 使用同一 `GenerationOutput`；
- `activate_weight()` 记录 active weight 和 transfer source；
- `generate()` 根据 prompt、policy version、seed、request id 生成 deterministic response/tokens/logprobs；
- 严格检查 active weight version；
- 支持 failure injection；
- 支持可选 `policy_segments`，用于测试 partial rollout / mixed policy sample。

### 5.4 TrainerBackend

现有 `TrainerBackend` 协议保留。当前 `FakeTrainerBackend` 已经接近目标，建议改造为正式 mock backend：

- 类名可保留 `FakeTrainerBackend` 作为兼容 alias；
- 新增导出名 `MockTrainerBackend`；
- `TrainerBackendConfig.backend` 从 `fsdp2 | fake` 调整为 `fsdp2 | mock`，保留 `fake` deprecated alias 一段时间；
- `optimize()` 的 loss 曲线由 config 控制；
- `export_weight()` 通过 `WeightStore` 写 mock payload 或 manifest；
- `hydrate()` / `offload()` 更新 mock residency；
- 所有方法继续检查 trainer lease。

### 5.5 WeightStore 与 WeightMaterializer

新增 `nano_rl/runtime/weight_store.py`：

```python
class WeightPayload(BaseModel):
    version_id: int
    parent_version: int | None = None
    trainer_step: int | None = None
    checksum: str
    tensors: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class WeightStore(Protocol):
    def bootstrap(self, model_path: str, tokenizer_path: str | None) -> WeightMeta:
        ...

    def export_from_trainer(
        self,
        *,
        parent: WeightMeta,
        rank: int,
        train_step: int,
        payload: WeightPayload | None = None,
    ) -> WeightMeta:
        ...

    def materialize_for_rollout(
        self,
        *,
        meta: WeightMeta,
        source: WeightShardSource,
    ) -> WeightPayload | dict[str, object]:
        ...
```

实现：

- `CheckpointWeightStore`：生产 checkpoint/artifact path；
- `MockMemoryWeightStore`：进程内保存小 payload；
- `MockFilesystemWeightStore`：写 JSON manifest，适合 Ray actor 跨进程读取。

`WeightTransferPlanner` 继续只产生 plan；真正 materialize 由 rollout backend activation 调用 weight store。

### 5.6 RewardBackend

当前 `RewardActorRole` 是 deterministic length reward。建议接口化：

```python
class RewardBackend(Protocol):
    @property
    def name(self) -> str: ...
    def score(self, prompt: str, response: str, metadata: Mapping[str, Any] | None = None) -> float: ...
```

短期保留默认 `MockRewardBackend` 即可，未来再接真实 reward function。

---

## 6. 端到端 mock 流程

### 6.1 全 mock，本地 CPU-only

```text
main.py --config recipes/mock_disaggregated.yaml --skip-artifact-validation
  -> load LaunchConfig
  -> data.source_type=mock_generated
  -> WeightStore.bootstrap() 生成 version 0 WeightMeta
  -> MockRolloutBackend.activate_weight(v0)
  -> MockRolloutBackend.generate()
  -> SampleQueueActorCore.submit_sample()
  -> shared GPU enter_train_window()
  -> MockTrainerBackend.optimize()
  -> MockTrainerBackend.export_weight() 生成 version 1
  -> WeightRegistryActorCore.register()/begin_activation()
  -> WeightTransferPlanner.build_plan()
  -> MockWeightStore.materialize_for_rollout()
  -> MockRolloutBackend.activate_weight(v1)
  -> return_to_rollout()
```

该流程应不 import vLLM/torch，不访问 HDFS，不要求真实 checkpoint。

### 6.2 Ray actor graph + mock backend

当：

```yaml
run:
  start_ray_actors: true
trainer:
  backend: mock
rollout:
  backend: mock
weight_transfer:
  store:
    backend: mock_filesystem
```

Ray actor graph 仍应启动：

- `GpuLeaseManagerActor` CPU actor；
- `RolloutManagerActor` CPU actor；
- `RolloutReplicaControllerActor` 使用 `MockRolloutBackend`；
- `TrainerRankActor` 使用 `MockTrainerBackend`；
- actors 仍使用 role-scoped custom resources；
- 不使用 Ray `num_gpus=1`。

如果 `mock_memory` 无法跨 Ray worker 共享，应在 config validation 阶段提示使用 `mock_filesystem` 或 `mock_objectref`。

### 6.3 混合模式示例

#### mock data + 真实 rollout/trainer

用于本地小数据调试：

```yaml
data:
  source_type: mock_inline
rollout:
  backend: vllm
trainer:
  backend: fsdp2
```

#### mock rollout + 真实 trainer

用于验证 trainer/FSDP2 batch 消费和权重 export：

```yaml
rollout:
  backend: mock
trainer:
  backend: fsdp2
weight_transfer:
  store:
    backend: checkpoint
```

#### 真实 rollout + mock trainer

用于验证 vLLM activation/generation 与 queue/registry，不跑 FSDP2：

```yaml
rollout:
  backend: vllm
trainer:
  backend: mock
```

---

## 7. 配置模型变更计划

### 7.1 `nano_rl/config.py`

建议新增/调整：

- `MockConfig`
- `MockStrictConfig`
- `MockTimingConfig`
- `MockFailureInjectionConfig`
- `DataSourceType` 扩展 mock source；
- `MockInlineDataConfig`
- `MockGeneratedDataConfig`
- `MockJsonlDataConfig`
- `TrainerBackendName = Literal["fsdp2", "mock"]`
- `RolloutBackendName = Literal["vllm", "huggingface", "mock"]`
- `MockTrainerConfig`
- `VllmConfig`
- `MockRolloutConfig`
- `WeightStoreConfig`
- `WeightTransferMethod` 扩展 `mock_objectref` / `mock_manifest`，或保持 method 不扩展、只用 `store.backend=mock_*`。

推荐尽量少扩展 `WeightTransferMethod`：

- transfer method 表达拓扑策略；
- store backend 表达 payload 实现；
- 这样 `locality_aware_checkpoint + mock_filesystem` 可以测试生产 transfer plan，但不写真实大 checkpoint。

### 7.2 兼容性

- 现有 `recipes/collocated.yaml` / `disaggregated.yaml` 不应失效；
- 如果新字段缺省：
  - `mock.enabled=false`；
  - `rollout.backend=vllm`；
  - `trainer.backend=fsdp2`；
  - `weight_transfer.store.backend=checkpoint`；
- `FakeTrainerBackend` 可保留 alias，但 YAML 新写法应使用 `trainer.backend: mock`。

### 7.3 Cross-field validation

新增校验：

- `trainer.backend=fsdp2` 时 `trainer.fsdp2` 必填；
- `trainer.backend=mock` 时禁止要求 `trainer.checkpoint_dir`；
- `rollout.backend=vllm` 时真实 actor start 需要 vLLM 可 import，但 config parse 不 import；
- `rollout.backend=huggingface` 时真实 actor start 需要 `transformers` 和 `torch` 可 import，但 config parse 不 import；
- `rollout.backend=mock` 时 `rollout.mock` 可选，有默认 deterministic 配置；
- `data.source_type` 必须出现在 `runtime.storage.allowed_input_sources`，或者 mock source 被 `mock.enabled=true` 显式允许；
- `mock.strict.no_external_data_probe=true` 时 mock data source 不走 HDFS probe；
- `run.start_ray_actors=true + weight_store.backend=mock_memory` 应 fail-fast 或降级提示，因为 Ray worker memory 不共享；
- mock backend 不改变 GPU topology 的数量合法性校验；`runtime.local.num_gpus`、topology、placement 仍按现有规则验证。

---

## 8. 代码落点

### 8.1 新文件

- `nano_rl/runtime/data_sources.py`
  - `PromptRecord`
  - `PromptSource`
  - `build_prompt_source()`
  - mock/generated/jsonl/HDFS implementations

- `nano_rl/runtime/weight_store.py`
  - `WeightPayload`
  - `WeightStore`
  - `MockMemoryWeightStore`
  - `MockFilesystemWeightStore`
  - future `CheckpointWeightStore`

- `nano_rl/runtime/backends/mock_rollout_backend.py`
  - `MockRolloutBackendConfig`
  - `MockRolloutBackend`

### 8.2 修改文件

- `nano_rl/runtime/backends/vllm_backend.py`
  - 保留协议；
  - `build_rollout_backend()` 按 backend name 分发到 vLLM、Hugging Face 或 mock。

- `nano_rl/runtime/backends/huggingface_backend.py`
  - `HuggingFaceRolloutBackend`
  - lazy-import `transformers` / `torch`，并实现同一个 `RolloutBackend` protocol。

- `nano_rl/runtime/backends/trainer_backend.py`
  - 将 `FakeTrainerBackend` 正式化为 `MockTrainerBackend`；
  - `build_trainer_backend()` 支持 `backend=mock`。

- `nano_rl/runtime/ray/actors.py`
  - `RolloutReplicaControllerActor` 根据 backend config 构造 rollout backend；
  - `TrainerRankActor` 根据 backend config 构造 trainer backend；
  - actor state 中暴露 backend name。

- `nano_rl/runtime/ray/placement.py`
  - 不改变 GPU topology；
  - launch plan metadata 中带上 backend name，便于 dry-run 检查。

- `nano_rl/runtime/controller.py`
  - 从 dataloader/prompt source 接收 prompts；
  - bootstrap/export/activate weight 时调用 weight store；
  - 保持 `ControllerCore` loop 不分叉。

- `nano_rl/runtime/artifacts.py`
  - mock data source 不走 HDFS probe；
  - mock model path 不要求存在；
  - 生产 source 仍 fail-fast。

- `docs/protocols/runtime-config.schema.yaml`
  - 同步 mock config schema。

- `recipes/mock_collocated.yaml`
  - 全 mock collocated example。

- `recipes/mock_disaggregated.yaml`
  - 全 mock disaggregated example。

- `README.md`
  - 增加 mock runtime 入口命令。

如果该方案进入实现阶段，还应同步 `docs/plans/plan-design.md` 与 `docs/architecture/design.html`，因为 mock backend 会成为正式架构能力。

---

## 9. Mock rollout 详细行为

### 9.1 activate_weight

输入：

- `WeightMeta`
- rollout `GpuLease` 或 TP group leases
- `WeightShardSource`

行为：

1. 校验 leases 覆盖 backend config 中的 `gpu_ids`；
2. 校验 role 都是 `rollout`；
3. 校验 holder 与 `holder_id` / `holder_ids` 匹配；
4. 如果配置了 weight store，则 materialize payload；
5. 记录：
   - active version；
   - checksum；
   - transfer source kind；
   - lease epochs；
   - activation count。

输出：无，失败抛结构化异常。

### 9.2 generate

输入：

- prompt；
- target policy version；
- request metadata；
- request id；
- leases。

行为：

1. 重复执行 lease 校验；
2. 校验 active weight exists；
3. 校验 active weight version 等于 target policy version；
4. 根据 seed、prompt、request id、policy version 生成 response；
5. 根据 response 生成 token ids；
6. 生成 deterministic logprobs；
7. 返回 `GenerationOutput`。

默认算法：

```text
response = response_template.format(prompt=prompt, policy_version=target_policy_version)
token_seed = sha256(seed + response + checksum)
tokens = first N bytes of token_seed
logprobs = [-0.01, -0.02, ...]
old_logprobs = logprobs
```

可选故障：

- 第 N 个 request 抛 `MockRolloutError`；
- 生成 stale policy segment；
- 返回空 completion；
- 模拟 OOM；
- 模拟 activation failure。

---

## 10. Mock trainer 详细行为

### 10.1 initialize_rank

只初始化 rank-local metadata，不触碰 CUDA：

- rank；
- world size；
- group epoch；
- comm epoch；
- train_step；
- weight_version；
- residency。

### 10.2 hydrate

输入 trainer lease，行为：

1. 校验 role 是 `trainer`；
2. 校验 gpu_id 与 rank assignment 匹配；
3. 校验 holder 是 `trainer-rank-{rank}`；
4. 将 residency 标记为 `mock_gpu_resident` 或 `cpu_standby`；
5. 返回 `TrainStateBundle`。

### 10.3 optimize

输入 `TrainBatch` 和 trainer lease，行为：

1. 校验 lease；
2. 校验 batch 不为空；
3. 递增 train_step；
4. 根据 loss config 生成 deterministic loss；
5. 返回 `OptimizerStepResult`。

默认 loss：

```text
loss = initial_loss / max(1, train_step + rank + batch.num_sequences)
```

### 10.4 export_weight

仅 rank 0 允许：

1. `version_id = parent.version_id + 1`；
2. `parent_version = parent.version_id`；
3. `trainer_step = current train_step`；
4. checksum 使用 parent checksum、version、train_step、seed 计算；
5. 通过 `WeightStore.export_from_trainer()` 写 payload 或 manifest；
6. 返回新的 `WeightMeta`。

### 10.5 offload

只改变 residency，返回 `TrainStateBundle`。如果未初始化则 fail-fast。

---

## 11. Mock weight store 详细行为

### 11.1 MockMemoryWeightStore

适用范围：

- local `ControllerCore`；
- unit tests；
- 不跨 Ray worker。

内部结构：

```python
versions: dict[int, WeightPayload]
manifests: dict[int, dict[str, object]]
```

优点：

- 快；
- 不写文件；
- 易断言。

限制：

- Ray actor 跨进程不可共享；
- 不能作为 `run.start_ray_actors=true` 的默认 store。

### 11.2 MockFilesystemWeightStore

适用范围：

- Ray actor graph；
- 需要人工查看 manifest；
- 需要跨进程传递 lightweight weight payload。

目录结构：

```text
.nano-rl-mock/
  weights/
    version-0/
      manifest.json
    version-1/
      manifest.json
```

manifest 示例：

```json
{
  "version_id": 1,
  "parent_version": 0,
  "trainer_step": 1,
  "format": "vllm_compatible",
  "checksum": "sha256:...",
  "created_by": "trainer-rank-0",
  "payload": {
    "kind": "mock_weight",
    "num_parameters": 1024,
    "seed": 42
  }
}
```

### 11.3 与 WeightTransferPlanner 的关系

`WeightTransferPlanner` 不读写权重，只决定 source：

- shared GPU replica -> `shared_gpu_reshard`；
- rollout-only replica -> `artifact_pull`；
- objectref mode -> `ray_object_ref`。

mock store 接收这些 source 后产生 mock payload：

- `shared_gpu_reshard`：记录 source rank/gpu，不拷贝真实 tensor；
- `artifact_pull`：读 mock manifest；
- `ray_object_ref`：读 mock object key。

这样可以用 mock 测试真实 transfer topology。

---

## 12. Mock data 详细行为

### 12.1 PromptRecord

每条 prompt 不应只是裸字符串，至少内部应带：

- `prompt_id`；
- `prompt`；
- `source_type`；
- `offset`；
- `metadata`。

进入现有 `RolloutManagerCore` 时可以只传 `prompt`，但 request metadata 应保留 prompt_id，方便 trace。

### 12.2 MockInlinePromptSource

用于最小调试：

```yaml
data:
  source_type: mock_inline
  mock_inline:
    prompts: ["hello", "world"]
    repeat: 1
```

输出顺序严格等于 YAML 顺序。

### 12.3 MockGeneratedPromptSource

用于压力测试 queue/backpressure：

```yaml
data:
  source_type: mock_generated
  mock_generated:
    count: 10000
    template: "prompt-{index}"
```

支持：

- `count`；
- `start_index`；
- `template`；
- `shuffle: false | deterministic`。

### 12.4 MockJsonlPromptSource

用于小 fixture：

```jsonl
{"prompt": "hello", "difficulty": "easy"}
{"prompt": "explain lease", "difficulty": "medium"}
```

规则：

- 使用结构化 JSON parser；
- 缺少 `prompt_column` fail-fast；
- 空行跳过；
- 非 JSON 行报错；
- metadata 保留其他字段。

---

## 13. Artifact validation 策略

`validate_input_artifacts()` 应根据 source/backend 决定 probe：

| 配置 | 行为 |
| --- | --- |
| `data.source_type=hdfs_uri` | 调 HDFS CLI probe |
| `data.source_type=hdfs_fuse_path` | 检查 mount root 与文件存在 |
| `data.source_type=mock_inline` | 不访问外部系统 |
| `data.source_type=mock_generated` | 不访问外部系统 |
| `data.source_type=mock_jsonl` | 检查本地文件存在并可读 |
| `trainer.backend=mock` | 不要求 checkpoint_dir |
| `rollout.backend=mock` | 不要求 model_path 存在 |
| `weight_store.backend=mock_memory` | 不检查 artifact path |
| `weight_store.backend=mock_filesystem` | 检查/创建 manifest_dir 的父目录 |

生产 source 仍保持 fail-fast。

---

## 14. Ray actor wiring

### 14.1 RolloutReplicaControllerActor

constructor 继续接收 `backend_config`，但 config 增加 backend name：

```json
{
  "backend": "mock",
  "gpu_ids": [0, 1],
  "holder_ids": ["rollout-dp-0-tp-0", "rollout-dp-0-tp-1"],
  "mock": {
    "seed": 42,
    "response_template": "{prompt} :: response@v{policy_version}"
  }
}
```

actor 内部：

```text
build_rollout_backend(config)
  -> MockRolloutBackend if backend=mock
  -> HuggingFaceRolloutBackend if backend=huggingface
  -> VllmRolloutBackend if backend=vllm
```

### 14.2 TrainerRankActor

constructor 继续接收 `backend_config`：

```json
{
  "backend": "mock",
  "rank": 0,
  "world_size": 4,
  "gpu_id": 4,
  "holder_id": "trainer-rank-0",
  "mock": {
    "initial_loss": 1.0,
    "loss_decay": "reciprocal"
  }
}
```

actor 内部：

```text
build_trainer_backend(config)
  -> MockTrainerBackend if backend=mock
  -> Fsdp2TrainerBackend if backend=fsdp2
```

### 14.3 RayLaunchPlan

dry-run 输出中增加：

```json
{
  "backend_summary": {
    "device": "mock",
    "rollout": "mock",
    "trainer": "mock",
    "weight_store": "mock_filesystem",
    "data": "mock_generated"
  }
}
```

这样可以从 `main.py --emit-resolved-config` 或 train plan 直接确认是否仍会触发真实依赖。

---

## 15. 示例配置

Mock 示例只保留两个用户入口：

- `recipes/mock_collocated.yaml`
- `recipes/mock_disaggregated.yaml`

二者都默认启动本地 Ray actor graph：

```yaml
mock:
  enabled: true
  seed: 42
  strict:
    leases: true
    versions: true
    no_gpu_imports: true
    no_external_data_probe: true

run:
  intent: train
  emit_resolved_config: false
  start_ray_actors: true

runtime:
  backend: ray
  local:
    num_gpus: 8
    cpus: 16
  storage:
    allowed_input_sources: [mock_generated]
  ray:
    address: auto
    namespace: nano-rl-mock
    gpu_manager:
      enabled: true
      lease_manager: true
      role_classes:
        rollout_manager: nano_rl.runtime.roles.RolloutManagerRole
        rollout_replica_controller: nano_rl.runtime.roles.RolloutReplicaControllerRole
        rollout_worker: nano_rl.runtime.roles.RolloutWorkerRole
        trainer_rank: nano_rl.runtime.roles.TrainerRankRole
      topology:
        rollout_only_gpus: 4
        shared_gpus: 4
        idle_gpus: 0
      hybrid_toggle:
        strategy: swap_on_toggle
        rollout_drain_timeout_sec: 1
        train_drain_timeout_sec: 1
        cuda_quiesce_timeout_sec: 1
        offload:
          trainer_model: cpu
          trainer_optimizer: cpu
          rollout_engine: teardown_and_reload
          cpu_memory_budget_gb: 16
          residual_gpu_memory_budget_mb: 0
    placement:
      trainer:
        num_ranks: 4
        gpus_per_rank: 1
      rollout:
        num_replicas: 4
        gpus_per_replica: 2
        tensor_parallel_size: 2

mode: disaggregated

parallel:
  trainer:
    data_parallel_size: 4
    fsdp_world_size: 4
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
  rollout:
    data_parallel_size: 4
    tensor_parallel_size: 2

model:
  model_path: mock://qwen
  tokenizer_path: mock://qwen-tokenizer

data:
  source_type: mock_generated
  prompt_column: prompt
  mock_generated:
    count: 32
    template: "prompt-{index}"

algorithm:
  name: ppo
  max_steps: 2
  learning_rate: 1.0e-6
  seed: 42

trainer:
  backend: mock
  global_batch_size: 8
  mock:
    initial_loss: 1.0
    loss_decay: reciprocal
    export_format: vllm_compatible

rollout:
  backend: mock
  mock:
    response_template: "{prompt} :: response@v{policy_version}"
    tokenization: sha256_bytes
    max_response_tokens: 16
    logprob_mode: linear
  partial_rollout:
    enabled: true
    pause_mode: keep
    clear_cache: true
    mixed_policy_samples: train
  hybrid:
    enabled: true
    precision_mix:
      bf16: 1.0
      fp8: 0.0
    policy_pin:
      latest_ratio: 0.8
      lagged_ratio: 0.2

weight_transfer:
  method: locality_aware_checkpoint
  allow_rollout_only_artifact_pull: true
  max_versions_in_flight: 2
  store:
    backend: mock_filesystem
    manifest_dir: ./.nano-rl-mock/disaggregated/weights
    checksum_mode: semantic_sha256

control:
  max_policy_lag: 2
  sample_ttl_sec: 300
  queue_high_watermark: 1024
  max_pending_rollout_refs: 32
  max_pending_train_refs: 4
```

原因：Ray actors 可能在不同 worker 进程，`mock_memory` 不可靠；启动 Ray 的 mock 示例必须使用 `mock_filesystem` 或其他跨进程可见的 store。

---

## 16. 测试计划

### 16.1 Config tests

新增：

- `tests/test_mock_config.py`
  - mock config parse；
  - production examples 仍 parse；
  - `trainer.backend=mock` 不要求 `fsdp2`；
  - `rollout.backend=mock` 不要求 vLLM；
  - `mock_memory + start_ray_actors=true` fail-fast；
  - mock data source artifact validation 不调 HDFS。

### 16.2 Backend unit tests

新增：

- `tests/test_mock_rollout_backend.py`
  - activate before generate required；
  - lease role/gpu/holder 校验；
  - deterministic output；
  - active policy version mismatch fail；
  - transfer_source 记录；
  - failure injection。

- `tests/test_mock_trainer_backend.py`
  - initialize/hydrate/optimize/offload；
  - rank0 export；
  - non-rank0 export fail；
  - deterministic checksum；
  - stale lease fail。

- `tests/test_weight_store.py`
  - memory store bootstrap/export/materialize；
  - filesystem manifest 写入/读取；
  - corrupt checksum injection。

- `tests/test_data_sources.py`
  - inline/generated/jsonl；
  - bad prompt column fail；
  - deterministic generated order。

### 16.3 Composition tests

新增：

- `tests/test_mock_controller.py`
  - full mock smoke iteration；
  - disaggregated mixed source plan 保持 `artifact_pull` / `shared_gpu_reshard`；
  - policy lag drop；
  - queue high watermark；
  - train failure 后 batch release + return_to_rollout。

- `tests/test_mock_ray_actor_graph.py`
  - `run.start_ray_actors=false` plan includes backend summary；
  - `run.start_ray_actors=true` dry-run actor graph with mock backend configs；
  - actor class constructors不 import vLLM/torch when backend=mock。

### 16.4 Import isolation tests

用 monkeypatch 阻止 import：

```python
def fake_import_module(name):
    if name in {"vllm", "torch", "transformers"}:
        raise AssertionError(f"unexpected import: {name}")
```

全 mock 路径应通过。

### 16.5 建议验证命令

```bash
python3 -m pytest -q
python3 -m compileall -q main.py nano_rl scripts/smoke_local_runtime.py
python3 main.py --config recipes/mock_disaggregated.yaml --emit-resolved-config
python3 main.py --config recipes/mock_disaggregated.yaml --skip-artifact-validation
python3 scripts/smoke_local_runtime.py --config recipes/mock_disaggregated.yaml --prompt "hello"
```

---

## 17. 分阶段实施计划

### Phase 1: Schema 与文档

目标：先让 mock 成为正式配置能力。

改动：

- `nano_rl/config.py` 增加 mock config models；
- `docs/protocols/runtime-config.schema.yaml` 同步；
- 新增 `recipes/mock_collocated.yaml`；
- 新增 `recipes/mock_disaggregated.yaml`；
- README 增加 mock runtime 入口；
- 本文档作为 implementation checklist。

验收：

- production examples parse；
- mock examples parse；
- `--emit-resolved-config` 输出 backend summary。

### Phase 2: Rollout/Trainer mock backend

目标：所有 GPU compute 叶子都可 mock。

改动：

- 新增 `MockRolloutBackend`；
- 将 `FakeTrainerBackend` 正式化为 `MockTrainerBackend`；
- factory 支持 `backend=mock`；
- Ray actor wrappers 按 config 构造 backend。

验收：

- mock rollout/trainer unit tests；
- no vLLM/torch import tests；
- local controller 能用 mock backend 而不是 role-only fake path。

### Phase 3: Data source mock

目标：不依赖 HDFS 或本地大数据也能走真实 prompt ingestion。

改动：

- 新增 `PromptSource`；
- 支持 inline/generated/jsonl；
- controller/driver 增加从 source 拉 prompt 的入口；
- artifact validation 分支处理 mock source。

验收：

- generated 1000 prompts 可稳定进入 backlog；
- queue backpressure 可由 generated source 测出；
- mock source 不访问 HDFS CLI。

### Phase 4: Weight store mock

目标：权重 load/store 不依赖真实 checkpoint，但保留 version/manifest/transfer 语义。

改动：

- 新增 `WeightStore`；
- mock memory/filesystem store；
- trainer export 与 rollout activation 接入 store；
- transfer plan 保持独立。

验收：

- version 0 bootstrap；
- train 后 version 1 export；
- rollout-only replica 走 artifact_pull mock manifest；
- shared replica 走 shared_gpu_reshard mock materialize；
- checksum 可重复。

### Phase 5: 故障注入与可观测性

目标：用 mock 测恢复路径和错误分支。

改动：

- mock failure injection；
- health event / metrics event；
- result payload 中暴露 mock events。

验收：

- rollout failure 标记版本 activation failed；
- trainer failure release batch 并 return_to_rollout；
- stale lease 被 fail-fast；
- corrupt weight checksum 被拒绝。

### Phase 6: Living docs 同步

如果开始实现该方案，需要同步：

- `docs/plans/plan-design.md`：加入 mock runtime 作为正式 v0.1 开发/验证模式；
- `docs/architecture/design.html`：加入 mock backend/component diagram；
- `docs/architecture/mode-fsm.md`：说明 mock 不改变 mode FSM，只替换 backend leaf；
- `AGENTS.md`：加入 mock validation commands。

---

## 18. 关键设计决策

### 18.1 为什么不只保留 `FakeTrainerBackend`

`FakeTrainerBackend` 目前主要服务单元测试。正式 mock runtime 需要覆盖：

- rollout；
- trainer；
- weight load/store；
- data source；
- Ray actor backend config；
- artifact validation；
- failure injection。

因此应该把 fake trainer 升级为 mock backend 体系的一部分，而不是继续把 mock 逻辑散落在 tests 和 smoke role adapter 中。

### 18.2 为什么 mock 仍要保留 GPU topology

mock 的主要价值不是节省几行配置，而是在没有 GPU 的机器上验证调度不变量：

- shared GPU 只能一个 active role；
- rollout-only GPU 不能 trainer lease；
- TP group 不跨生命周期区域；
- trainer rank 与 shared GPU 一一对应；
- rollout replica 与 TP workers 覆盖 rollout assignment set。

如果全 mock 时去掉 GPU topology，就无法提前捕获这些错误。

### 18.3 为什么 weight registry 和 weight store 分离

`WeightRegistryActorCore` 管版本状态：

- registered；
- activating；
- active_global；
- failed；
- deprecated。

`WeightStore` 管 payload：

- checkpoint path；
- manifest；
- mock memory；
- mock filesystem。

二者分离后，mock store 可以替换真实 checkpoint，而不会绕开 activation 状态机。

### 18.4 为什么 data mock 也要接口化

当前 config 已经有 `data.source_type` 和 artifact validation。如果 mock data 只是脚本参数，就无法测试：

- source validation；
- prompt ingestion；
- backlog；
- queue high watermark；
- sample TTL；
- policy lag；
- prompt metadata tracing。

因此 data mock 应进入正式配置模型。

---

## 19. 最小可落地版本

如果需要先做最小实现，建议只做以下切片：

1. `trainer.backend: mock` 使用现有 `FakeTrainerBackend` alias；
2. `rollout.backend: mock` 新增 `MockRolloutBackend`；
3. `data.source_type: mock_inline | mock_generated`；
4. `weight_transfer.store.backend: mock_memory`；
5. 新增 `recipes/mock_disaggregated.yaml`；
6. `ControllerCore.run_smoke_iteration()` 改为使用 backend factories；
7. 测试 full mock smoke iteration。

这个切片已经能覆盖用户最关心的三类外部依赖：

- GPU compute；
- 权重 load/store；
- data 获取。

后续再补：

- `mock_filesystem`；
- Ray actor graph 跨进程 mock；
- failure injection；
- living HTML design doc 同步。

---

## 20. 完成定义

该 mock 方案实现完成后，应满足：

- 一份全 mock YAML 可以在 CPU-only 环境执行，不需要 CUDA/vLLM/torch/HDFS/checkpoint；
- mock rollout 与 vLLM rollout 实现同一 `RolloutBackend`；
- mock trainer 与 FSDP2 trainer 实现同一 `TrainerBackend`；
- mock weight store 不绕过 `WeightRegistryActorCore` 和 `WeightTransferPlanner`；
- mock data source 不绕过 prompt ingestion 和 queue；
- GPU lease、policy version、weight checksum、sample TTL、policy lag 都在 mock 路径中被真实检查；
- production examples 继续保持兼容；
- 测试覆盖 config、backend、data、weight、controller composition 和 Ray actor wiring。
