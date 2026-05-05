# nano-rl 设计文档（v0.1 规划）

> 目标：为“小而全”的 RL 框架定义一套可落地架构，支持：
>
> 1) **fully sync** rollout/train；
> 2) **standalone + hybrid** rollout/train；
> 3) trainer 使用 **PyTorch FSDP2**；
> 4) rollout backend 使用 **vLLM**；
> 5) 执行层采用 **Ray-native actor runtime**，类似 verl 的单 job 内 actor 编排思路；
> 6) 用户侧统一通过仓库根目录的 **`main.py`** 加载 YAML 配置并启动。

---

## 1. 设计目标与非目标

### 1.1 设计目标

- 在单机下统一执行模型更新 + rollout 采样流程；
- 提供两条用户可选择的部署形态，并归一化到内部 canonical mode：
  - `collocated` -> `fully_sync`
  - `disaggregated` -> `standalone_hybrid`
- 抽象稳定协议：sample、weights、metrics、fault；
- 支持 vLLM rollout 与 FSDP2 trainer 的权重/版本协同；
- 使用 Ray 原生 actor / ObjectRef / placement group 管理 trainer、rollout、reward、queue 与控制器；
- 提供单一用户启动入口 `main.py`，从 YAML 配置中读取本机资源、并行维度、训练配置与运行模式，并归一成 `LaunchConfig`；
- 具备最小可运维能力：配置校验、dry-run、健康检查、可观测性。

### 1.2 非目标（v0.1）

- 不覆盖所有 RL 算法，仅先支持 PPO-like on-policy 主路径；
- 不实现复杂多租户调度；
- v0.1 不实现多机、K8s/Slurm/SSH adapter；Ray local runtime 是默认且唯一的执行层；
- 不把 Ray 当作权重仓库：大模型权重仍通过 checkpoint/artifact path 传递，Ray 只传控制消息、样本引用和状态。

---

## 2. 总体架构

```
+-------------------- Ray Driver / Control Plane ---------------------+
| Config Loader | ray.init | ControllerActor | Mode FSM | Recovery    |
+----------------------------+----------------------------------------+
                             |
                             v
+------------------------ Ray Resource Plane -------------------------+
| GpuLeaseManagerActor owns physical GPU active-role leases           |
|   user YAML provides counts only; LaunchConfig expands gpu_plan      |
|   rollout and trainer actor sets may overlap on shared_gpus          |
|   role actors use logical Ray resources + lease epoch CUDA gates     |
+----------------------------+----------------------------------------+
                             |
                             v
+-------------------------- Ray Data Plane ---------------------------+
| TrainerGroup(logical ranks) <-> WeightRegistryActor <-> RolloutPool |
| RewardActorPool              SampleQueueActor        MetricsActor   |
| Ray Object Store: sample refs / metric payloads / small metadata    |
+--------------------------------------------------------------------+
```

### 2.1 核心组件

1. **`main.py`**：用户启动入口，加载 YAML config，处理少量执行控制参数，生成 `LaunchConfig`；
2. **Ray Driver / RayClusterController**：初始化 Ray runtime、校验资源、创建 `ControllerActor`；
3. **ControllerActor**：唯一 loop 语义源，统一驱动 train-step/rollout-step；
4. **Mode FSM**：模式状态机（fully sync、standalone hybrid、degraded）；
5. **GpuLeaseManagerActor**：CPU actor，负责物理 GPU 的 active role、lease epoch、失败状态和 CUDA gate；
6. **TrainerGroup (logical FSDP2 ranks)**：由自动分配到 `shared_gpus` 的 `TrainerRankActor` 组成，负责参数更新、梯度同步、checkpoint；
7. **RolloutManager / RolloutReplicaController / RolloutWorkerActor**：全局 rollout 生命周期、每个 DP replica 的 TP group 控制，以及实际 vLLM GPU worker；
8. **RewardActorPool**：reward 计算与可插拔 reward function；
9. **WeightRegistryActor**：权重版本注册、激活、健康状态；
10. **WeightTransferPlanner / WeightTransfer module**：独立决定 trainer 更新后的权重如何进入每个 rollout replica，支持 `objectref` 和 locality-aware artifact/reshard 策略；
11. **SampleQueueActor**：样本传输与背压（trajectory、token-level stats）；
12. **MetricsActor**：统一指标、事件和 tracing。

### 2.2 Ray-native 执行层

Ray 不是外部 adapter，而是 nano-rl v0.1 的执行模型：

1. `RayDriver.train()` 总是先校验配置并生成 `RayLaunchPlan`；
2. 当 `run.start_ray_actors: false` 时，train 路径返回 `planned_backend_integrated` 计划，不启动 Ray actor graph，适合 config/dry-run/CI 验证；
3. 当 `run.start_ray_actors: true` 时，`RayActorGraphLauncher` 先通过 `RayClusterController` 初始化 Ray：`runtime.ray.address=auto` 先尝试连接已有 cluster，连接失败则创建本机 single-machine Ray cluster；显式非 `auto` address 保持连接失败即 fail-fast；
4. Ray 初始化完成后，`RayActorGraphLauncher` 按 `RayLaunchPlan` 创建真实 Ray actor graph；
5. `RayTrainingLoop` 在 driver 侧串接已启动的 Ray actors，执行 prompt ingestion、rollout generation、reward scoring、sample queue reservation、shared GPU lease toggle、trainer optimize、rank0 weight export、weight registry activation 和 rollout weight reactivation；
6. `gpu_plan` 由数量自动展开：用户只写 `rollout_only_gpus`、`shared_gpus`、rollout DP/TP 和 trainer rank 数，不手写物理 GPU id；
7. `TrainerGroup` 是多个 `TrainerRankActor` 的逻辑集合，rank 映射到 `shared_gpus`，真实 FSDP2 多 rank 运行需要模型/checkpoint artifact 与稳定 rendezvous/store endpoint；
8. `RolloutManagerActor` 管理多个 `RolloutReplicaControllerActor`；每个 DP replica controller 管理一个 vLLM TP group 和若干 `RolloutWorkerActor`；
9. `RewardActorPool` 与 `SampleQueueActor` 通过 Ray ObjectRef 传输样本批次；
10. 大权重默认不通过 Ray object store 广播，`TrainerRank(rank=0)` 发布 `WeightMeta(model_path=...)`，`WeightTransferPlanner` 按 `weight_transfer.method` 为每个 rollout replica 选择 objectref、shared GPU 本地 reshard 或 artifact pull。

这样做的目标是让启动、路由、故障感知都留在 Ray 内部完成，同时避免把模型权重这种大对象错误地塞进 Ray object store。

### 2.3 Ray actor 类型与 GPU lease

v0.1 采用“数量配置、自动分配、lease gate”的 GPU 资源模型。用户 YAML 不手写物理 GPU id，只提供本机 GPU 总数、`rollout_only_gpus`、`shared_gpus`、rollout DP/TP 和 trainer rank 数。`LaunchConfig` 将这些数量确定性展开为 `ResolvedGpuPlan`，用于 dry-run、日志、actor 创建和故障定位。

`rollout_only_gpus` 与 `shared_gpus` 是物理 GPU 生命周期区域，不是用户手写的 GPU id：

```yaml
runtime:
  local:
    num_gpus: 8
  ray:
    gpu_manager:
      lease_manager: true
      topology:
        rollout_only_gpus: 4
        shared_gpus: 4
        idle_gpus: 0
    placement:
      trainer:
        num_ranks: 4
        gpus_per_rank: 1
      rollout:
        num_replicas: 4
        gpus_per_replica: 2
        tensor_parallel_size: 2
```

该配置会自动展开为：

```yaml
resolved_gpu_plan:
  rollout_only_gpu_ids: [0, 1, 2, 3]
  shared_gpu_ids: [4, 5, 6, 7]
  rollout_replicas:
    - replica_id: rollout-dp-0
      gpu_ids: [0, 1]
    - replica_id: rollout-dp-1
      gpu_ids: [2, 3]
    - replica_id: rollout-dp-2
      gpu_ids: [4, 5]
    - replica_id: rollout-dp-3
      gpu_ids: [6, 7]
  trainer_ranks:
    - rank: 0
      gpu_id: 4
    - rank: 1
      gpu_id: 5
    - rank: 2
      gpu_id: 6
    - rank: 3
      gpu_id: 7
```

分配规则：

- 可见 GPU 集合来自 `runtime.local.num_gpus`、`CUDA_VISIBLE_DEVICES` 或 Ray node inventory；v0.1 的本地 core 用 `0..num_gpus-1` 做 deterministic dry-run；
- 前 `rollout_only_gpus` 张卡只给 rollout actor set；
- 接下来的 `shared_gpus` 张卡同时进入 rollout actor set 和 trainer actor set，两个 set 之间允许重叠；
- rollout actor set 内部 GPU 不重复，trainer actor set 内部 GPU 不重复；
- `rollout_only_gpus` 和 `shared_gpus` 都必须能被 rollout `tensor_parallel_size` 整除，避免一个 vLLM TP group 横跨两种生命周期区域。

长生命周期 rollout/trainer execution actors 不应通过 Ray `num_gpus=1` 表达 GPU 资源。Ray 不理解“rollout 和 trainer 在时间上互斥但 actor 同时存活”，如果两组 actor 都申请 `num_gpus=1`，会把 overlap 误判成需要额外 GPU token。推荐用 role-scoped custom resources 表达集合内唯一性：

```text
RolloutReplicaControllerActor(num_gpus=0, resources={"rollout_gpu_4": 1, "rollout_gpu_5": 1})
TrainerRankActor(num_gpus=0, resources={"train_gpu_4": 1})
```

每个 `RolloutReplicaControllerActor` 按 vLLM TP 组一次性占用该 replica 的所有 `rollout_gpu_i` custom resources；`TrainerRankActor` 按 rank 占用一个 `train_gpu_i` custom resource。`rollout_gpu_4` 与 `train_gpu_4` 是两个逻辑资源，允许对应同一张物理 GPU。真正的跨 role 互斥由 CPU-only `GpuLeaseManagerActor` 执行。所有会执行 CUDA 的方法都必须携带并校验 `(gpu_id, role, holder_id, lease_epoch)`；这包括 `generate`、`train_step`、`activate_weight`、hydrate/offload 后的 wake 等会改变 GPU residency 的路径。

实现上先生成一个 `RayLaunchPlan`，把 node custom resources 和 actor specs 明确展开：

- `node_custom_resources` 为 rollout actor set 声明 `rollout_gpu_i`，为 trainer actor set 声明 `train_gpu_i`；
- `RolloutReplicaControllerActor` 使用 `num_gpus=0, resources={"rollout_gpu_i": 1, ...}`，一个 replica 按 TP 组占用多张 rollout GPU；
- `RolloutWorkerActor` 仍保持 `num_gpus=0`，其 CUDA-facing 行为由 replica/controller/backend 和 rollout lease gate 约束；
- `TrainerRankActor` 使用 `num_gpus=0, resources={"train_gpu_i": 1}`；
- `shared_gpus` 对应的同一个 `gpu_id` 会同时出现 `rollout_gpu_i` 与 `train_gpu_i`，但没有任何长生命周期 actor 申请 Ray `num_gpus=1`；
- dry-run 输出 `ray_launch_plan`，用于检查 actor placement、resource inventory、constructor args 和 GPU overlap 是否符合预期。

当前实现已经把 Ray actor graph 接到 vLLM/FSDP2 backend adapter，真实依赖仍保持 lazy import：

- `nano_rl.runtime.ray.cluster.RayClusterController`：封装 Ray cluster connect-or-create 逻辑；`address=auto` 先连已有 cluster，失败后创建带 `node_custom_resources` 的本机 cluster；显式 address 连接失败时不 fallback；
- `nano_rl.runtime.ray.launcher.RayActorGraphLauncher`：根据 `RayLaunchPlan` 创建真实 Ray actor graph；`run.start_ray_actors=false` 只返回 plan，`run.start_ray_actors=true` 才初始化 Ray 并启动 actors；
- `nano_rl.runtime.ray.training.RayTrainingLoop`：在真实 actor graph 启动后执行 bounded training loop；mock configs 已经能通过 `main.py` 完成多 step rollout -> reward -> train -> publish weight；
- `nano_rl.runtime.ray.actors.RolloutReplicaControllerActor`：构造 `VllmRolloutBackend`，在 `activate_weight()` / `generate()` 中把 Ray actor 调用转成 vLLM backend adapter 调用；
- `nano_rl.runtime.ray.actors.TrainerRankActor`：构造 `Fsdp2TrainerBackend`，在 `initialize_rank()` / `hydrate()` / `optimize()` / `export_weight()` / `offload()` 中转接 FSDP2 trainer backend adapter；
- `nano_rl.runtime.backends.vllm_backend.VllmRolloutBackend`：lazy-import vLLM，统一 `activate_weight` / `generate` 输出，并在 CUDA-facing 方法入口校验 rollout lease；
- `nano_rl.runtime.backends.fsdp2_backend.Fsdp2TrainerBackend` 与 `FakeTrainerBackend`：保留 FSDP2 rank/process-group/comm-epoch 边界；真实 FSDP2 运行依赖可加载的模型、`trainer.checkpoint_dir`、torch/FSDP2/transformers 依赖和多 rank rendezvous/store endpoint；
- `nano_rl.runtime.offload.GpuResidencyManagerCore`：把 shared GPU rollout/train 切换建成可执行 residency 状态机，Controller 的 train enter/exit 已经通过它执行 offload/hydrate hooks 和 lease epoch 校验。

| Actor | Ray 资源 | 主要职责 |
| --- | --- | --- |
| `ControllerActor` | CPU | 训练主循环、模式状态机、train trigger、背压和恢复策略 |
| `GpuLeaseManagerActor` | CPU | 物理 GPU active role、lease epoch、holder、失败状态与 fail-fast gate |
| `RolloutManagerActor` | CPU | rollout 生命周期、input backlog、queue 水位、权重同步、train/rollout 切换协同 |
| `RolloutReplicaControllerActor` | TP group logical rollout GPU resources | 每个 rollout DP replica 一个；按 TP 组占用 `rollout_gpu_i` custom resources，管理 vLLM backend adapter、权重激活、drain/offload/wake 和 replica 失败 |
| `RolloutWorkerActor` | `num_gpus=0` actor / TP rank role | 每个 vLLM TP rank 一个；不申请 Ray GPU token，CUDA-facing 行为必须通过 rollout lease 与 replica/backend gate |
| `TrainerCoordinatorActor` | CPU | 汇总 trainer ranks，建立 rank/world size，调度 train window |
| `TrainerRankActor` | logical train GPU resource | FSDP2 rank、train step、梯度同步、checkpoint shard |
| `SampleQueueActor` | CPU | 样本队列、TTL、policy lag guard、ObjectRef 背压 |
| `WeightRegistryActor` | CPU | 权重版本、激活状态、rollback 标记 |
| `WeightTransferPlanner` | CPU module | per-version/per-replica 权重传输计划；`objectref` 用 Ray object store，`locality_aware_checkpoint` 对 shared GPU reshard、对 rollout-only GPU artifact hydrate |
| `MetricsActor` | CPU | 指标、事件、health trace |

### 2.4 GPU topology 与 role 状态

每个物理 GPU 的 lease 状态包括：

- `gpu_id`：resolved plan 展开的物理 GPU id；
- `topology`：`rollout_only`、`shared` 或 `idle`；
- `active_role`：`rollout`、`trainer` 或 `None`；
- `holder_id`：当前 lease 持有者，例如 `rollout-dp-2-tp-0` 或 `trainer-rank-0`；
- `lease_epoch`：每次 active role 或 holder 变化时递增；
- `failed_reason`：GPU 或 actor 组失败时的结构化原因。

因此，`rollout_only` 表示只支持 rollout lease；`shared` 表示支持 rollout/trainer 两种 lease，但任意时刻只能有一个 role active；`idle` 表示预留 GPU，不参与 v0.1 默认执行。

每个 role 的局部状态：

- `ABSENT`：该 GPU / replica 不支持这个 role；
- `INIT`：正在构建 runtime；
- `READY`：已初始化但未占用 CUDA 执行窗口；
- `ACTIVE`：当前 role 可以发起 CUDA kernel；
- `DRAINING`：停止接新请求，等待 in-flight batch 结束；
- `PAUSED`：保留必要元数据，不允许执行；
- `UNLOADED`：已释放 CUDA memory，仅保留恢复所需 metadata；
- `FAILED`：role 失效，需要 GPU、replica 或 group 级恢复。

只有 shared GPU 才允许进入完整 toggle 状态：

- `ROLLOUT_ACTIVE`
- `ROLLOUT_DRAINING`
- `ROLLOUT_OFFLOADING_CPU`
- `TRAIN_HYDRATING_GPU`
- `TRAIN_READY_BARRIER`
- `TRAIN_ACTIVE`
- `TRAIN_DRAINING`
- `TRAIN_OFFLOADING_CPU`
- `ROLLOUT_WAKING_GPU`
- `ROLLOUT_READY_BARRIER`

默认 toggle 策略是 `swap_on_toggle`：inactive role 释放 CUDA memory，只保留配置、rank、weight version、optimizer/engine metadata。未来如果模型足够小，可以增加 `dual_resident`，但仍必须保证 inactive role 不执行 CUDA kernel。

### 2.5 Toggle 语义

`ControllerActor` 不直接向 inactive role 发 CUDA 请求。所有共享 GPU 的切换必须经过 `GpuLeaseManagerActor` 和对应 role actor 的 drain/offload/hydrate 协议。`GpuLeaseManagerActor` 首先检查 topology：当 GPU 属于 `rollout_only` 时，授予 trainer lease 必须立即返回 `GpuRoleUnsupported(role=trainer)`，不能等待未来变成 shared GPU。

1. `rollout -> train`：
   - 只切换 `shared_gpus` 对应的 rollout replicas；`rollout_only_gpus` 对应的 replicas 继续接收 prompt 并持续 rollout；
   - `RolloutManagerActor` 停止向会占用 shared GPUs 的 replicas 发新 prompt；
   - 等待 in-flight generation 完成或超时取消；
   - flush 已完成样本到 `SampleQueueActor`；
   - 释放 vLLM KV cache 和 rollout CUDA memory；
   - `GpuLeaseManagerActor` 将 shared GPUs 的 lease 授予对应 `TrainerRankActor`；
   - `TrainerRankActor` 加载/恢复训练状态；
   - 所有 trainer ranks 到达 train barrier 后，`TrainerCoordinatorActor` 开始一个 train window。
2. `train -> rollout`：
   - `TrainerRankActor` 完成当前 micro-step 或在安全点暂停；
   - rank0 发布 `WeightMeta`；
   - 释放训练临时 CUDA memory；
   - `GpuLeaseManagerActor` 将 shared GPUs 的 lease 归还给对应 rollout workers；
   - `RolloutReplicaControllerActor` 激活目标权重版本并唤醒 TP group；
   - rollout replicas 到达 rollout barrier 后，`RolloutManagerActor` 恢复发 prompt。

如果任一 shared GPU lease 切换、rollout offload、trainer hydrate 或 barrier 失败，整个相关 trainer group 或 rollout replica 进入 `PAUSED` / `DEGRADED_SYNC`，不能让部分 FSDP ranks 或部分 TP ranks 继续推进。

#### 2.5.1 Role residency 与 offload 边界

`swap_on_toggle` 的语义不是销毁 role actor，而是切换同一张 shared GPU 上两个独立 role actor 的 **residency** 与 **CUDA 执行权**。Shared GPU 内始终只有一个 role 可以处于 GPU-active，另一个 role 必须处于 CPU standby 或 unloaded metadata standby。

| Window | Rollout role | Trainer role | 通信组状态 |
| --- | --- | --- | --- |
| `ROLLOUT_ACTIVE` | vLLM engine 持有 GPU weights/KV cache，接收 prompt | FSDP2 rank 进程存活，model shard、optimizer state、scheduler/RNG/scaler 在 CPU standby | trainer process group 存活但 idle，不发起 collective |
| `ROLLOUT_DRAINING` | 停止接新 prompt，等待或取消 in-flight generation，flush sample refs | 保持 CPU standby | 不发起 trainer collective |
| `ROLLOUT_OFFLOADING_CPU` | vLLM sleep/offload 或 teardown engine；KV cache 必须释放，weights 按策略 offload CPU 或丢弃待重载 | 准备 hydrate 到 GPU | rollout 私有通信资源随 rollout backend 暂停，trainer group 不销毁 |
| `TRAIN_HYDRATING_GPU` | rollout role 已 paused | model shard 与 optimizer state 从 CPU standby 恢复到 GPU | 所有 rank 使用同一 `comm_epoch` 进入 train enter barrier |
| `TRAIN_READY_BARRIER` | rollout role 保持 paused | 等待所有 trainer ranks ready | barrier 完成后才能开始 train collective |
| `TRAIN_ACTIVE` | rollout role 不允许执行 CUDA kernel | model shard 与 optimizer state 在 GPU，所有 rank 进入同一 FSDP2 train window | trainer process group active，collective 顺序由 `TrainerCoordinatorActor` 控制 |
| `TRAIN_DRAINING` | 保持 paused | 完成当前 micro-step，等待 async work，进入 train exit barrier | 所有 rank 一起退出 train window |
| `TRAIN_OFFLOADING_CPU` | 准备 wake/reload 目标 `WeightMeta` | model shard 与 optimizer state offload 到 CPU standby，释放 CUDA tensors/cache | trainer process group 保持存活但 idle |
| `ROLLOUT_WAKING_GPU` | vLLM wake/load 新权重，重建 KV cache 预算 | CPU standby | rollout replicas 到达 rollout barrier 后恢复 prompt |
| `ROLLOUT_READY_BARRIER` | rollout role 已加载目标版本 | trainer 保持 CPU standby | rollout barrier 完成后恢复 prompt |

训练侧 CPU standby 必须保存一个 rank-local `TrainStateBundle`：

- model shard tensors，推荐使用 pinned CPU memory；
- optimizer state shards，包括 Adam moments 等大状态；
- scheduler state、RNG state、grad scaler/mixed precision state；
- FSDP2 metadata、rank/world size、`policy_version`、`train_step`、checkpoint cursor；
- shape/dtype/checksum 元数据，用于 hydrate 前后做一致性校验。

`TrainerRank.suspend_to_cpu()` 的顺序必须是：完成当前 micro-step -> 等待所有 async collective work -> `dist.barrier()` -> `zero_grad(set_to_none=True)` -> model/optimizer shard copy 到 CPU -> 删除 CUDA tensor 引用 -> `torch.cuda.synchronize()` -> `empty_cache()` -> 上报 `TRAIN_CPU_STANDBY`。`TrainerRank.resume_to_gpu()` 的顺序必须是：按同一 rank/world metadata 分配 CUDA tensors -> 从 CPU bundle 拷贝 model shard 和 optimizer state -> 重建 FSDP2 runtime handles -> `dist.barrier()` -> 上报 `TRAIN_READY`。正常 toggle 不允许跳过 optimizer state；否则下一次 train window 会丢失动量和自适应学习率状态。

rollout 侧优先把 `RolloutWorker.suspend_to_cpu()` 映射到 vLLM sleep/offload 能力：同权重短暂停可用 sleep level 1，训练后需要换权重时优先使用 level 2 + partial wake/load weights + wake KV cache。若当前 vLLM 版本或部署形态不支持 sleep/offload，adapter 可以 teardown vLLM engine，但必须保留 `RolloutStateHandle(policy_version, tokenizer_version, engine_config, weight_meta)`，并把该路径标记为更慢的 fallback；不能因为 rollout engine 重建而影响 trainer process group。

#### 2.5.2 通信组生命周期

FSDP2 通信组与 GPU residency 解耦。`TrainerCoordinatorActor` 在 bootstrap 时固定 rank mapping 和 rendezvous metadata，并在首个 train window 初始化稳定的 `TrainerCommSession`，包含 `world_id`、rank mapping、store endpoint、process group handle、device mesh metadata 和 `comm_epoch`。一旦 session healthy，正常 `rollout <-> train` toggle **不调用** `destroy_process_group()`，也不重新分配 rank；只让同一批 rank 在 CPU standby 与 GPU-active 之间切换。

通信组保护规则：

- 所有 FSDP/NCCL collective 只允许发生在 `TRAIN_ACTIVE` 或受控的 train enter/exit barrier 中；
- 进入 train 前，所有 shared GPU 上的 rollout replicas 必须先完成 rollout offload，再一起 hydrate trainer state；
- 退出 train 前，所有 rank 必须完成 optimizer step、等待 async work、进入 barrier，然后才能 offload CUDA tensors；
- 任何 rank hydrate/offload 失败时，本轮 train window fail-fast，整个 `TrainerGroup` 进入 `PAUSED` 或整体 rebuild，不能让健康 rank 单独继续；
- process group rebuild 只允许发生在 GPU/rank 失败、NCCL error、作业恢复或 shutdown 路径中；rebuild 必须使用 Ray control-plane 做 torch.distributed 之外的 out-of-band rendezvous，生成新的 `comm_epoch`，并让旧 epoch 的 pending train batch 全部失效；
- vLLM tensor-parallel/pipeline-parallel 通信资源属于 rollout backend 私有资源。每个 `RolloutReplicaControllerActor` 管理一个 DP replica 的 TP group，sleep/wake/offload、activation 和失败都以 replica 为单位处理，不能把 rollout 通信组和 FSDP2 trainer process group 混用。

这条约束的工程原因是：`destroy_process_group()` 需要所有 ranks 以一致顺序执行，运行中 destroy/reinit 还需要 torch.distributed 之外的同步；把它绑到高频 role toggle 上会把正常调度路径变成最容易 deadlock 的路径。因此 normal toggle 保持 process group 活着，只释放 CUDA tensor residency。

保留 process group 可能仍会留下少量 CUDA context/NCCL bookkeeping 显存；设计目标是释放 model weights、optimizer state、activation/KV cache 等大块 residency，而不是追求完全 0 MiB。`cuda_quiesce_timeout_sec` 之后如果 residual GPU memory 超过预算，相关 GPU/replica 必须上报 `SHARED_GPU_OFFLOAD_FAILED`，由 controller 暂停或整组 rebuild。

#### 2.5.3 Collocated 与 disaggregated 的差异

`collocated` 归一化为 `fully_sync` 后，`rollout_only_gpus=0`，所有 rollout GPU 都在 shared pool 内。每个 iteration 都是全局相位切换：全部 rollout replicas drain/offload -> 全体 ranks hydrate train -> train -> 全体 ranks offload -> 全部 rollout replicas wake。此模式下 `max_policy_lag=0`，没有 standalone rollout 继续采旧权重。

`disaggregated` 归一化为 `standalone_hybrid` 后，只有 shared GPUs 对应的 rollout replicas 进入上述切换；rollout-only GPUs 不进入 train barrier，继续用受控旧权重采样。训练窗口期间 `SampleQueueActor` 必须按 `policy_version`、TTL 和 queue 水位过滤样本；新权重发布后，所有 rollout replicas 一起进入全局暂停式权重切换，到达同一个目标版本后再恢复发 prompt。

### 2.6 `main.py` 启动入口与 YAML 配置

仓库根目录必须保留一个用户可见的启动入口，但启动入口只消费完整 YAML 配置，不把训练参数拆成一行行命令参数。

YAML 是训练配置的主入口。本机 GPU 数量、GPU pool 数量、并行维度、模型、数据、训练超参、collocated/disaggregated 部署形态与控制阈值都应写在 YAML 中。具体物理 `gpu_id` 由系统在 resolved config 中展开，`main.py` 不应该为每个训练字段都增加同名命令行参数，避免出现“YAML 一套、命令一套”的双配置系统。

`main.py` 只承担用户入口职责，不直接承载训练循环。它应完成：

1. 读取 YAML config；
2. 读取 YAML 中的 `run.intent`，决定执行 `train`、`dry_run` 或 `validate`；
3. 将配置中的用户部署形态归一化，例如 `collocated -> fully_sync`、`disaggregated -> standalone_hybrid`；
4. 归一化模型权重/tokenizer 路径，并校验训练数据的 storage source；模型与 checkpoint 路径不声明 `source_type`，也不因路径不是 HDFS/HDFS-FUSE 而被拒绝；
5. 校验本机 GPU pool 数量、并行维度、batch size 与模式约束；
6. 将 resolved config 写入仓库根目录下被 git ignore 的 `.nano_rl/resolved-config.json`，确保最终配置可审计且不占用 stdout；
7. 生成不可变 `LaunchConfig`；
8. 调用 Ray Driver 启动、dry-run 或仅校验。

YAML 参数分为四层：

- **拓扑与资源参数**：单机 GPU/CPU 数、`rollout_only_gpus`、`shared_gpus`、Ray address/namespace；
- **并行维度参数**：trainer data parallel / FSDP world size、rollout data parallel、rollout tensor parallel、reward actor 数；
- **训练语义参数**：model、dataset、algorithm、learning rate、batch size、max steps、seed、checkpoint；
- **模式与控制参数**：用户入口 `collocated/disaggregated`，内部 canonical mode `fully_sync/standalone_hybrid`，以及 `max_policy_lag`、sample TTL、queue 水位、ObjectRef 背压阈值。

单机资源参数在 v0.1 中用于 Ray local runtime 资源校验与 placement group 规划。`runtime.ray.address=auto` 允许启动时先连接已有 cluster，连接失败后由启动控制器创建本机 single-machine Ray cluster；v0.1 仍不负责 SSH/Slurm/K8s 或多机 Ray cluster 拉起。多机扩展留到后续版本。

### 2.7 启动期输入 artifact 校验

模型权重、tokenizer 与训练数据是 `LaunchConfig` 的前置输入。`main.py` 必须在初始化 Ray、创建 `ControllerActor` 或启动任何 GPU role 之前完成必要的 fail-fast 校验；失败时直接报错退出，不进入部分启动状态。

v0.1 的模型与 checkpoint 路径按普通 artifact path 处理，不要求声明 HDFS/HDFS-FUSE source type，也不把路径限制在某个 HDFS FUSE mount root 下。`model.model_path` 与 `model.tokenizer_path` 只负责告诉 trainer/rollout/backend loader 从哪里加载；路径是否真实可读由后端加载器或可选 artifact probe 处理，不作为配置 schema 的 storage source 约束。

训练数据仍保留显式输入来源。v0.1 只支持两类数据输入来源：

- `hdfs_uri`：形如 `hdfs://namenode/path` 的 HDFS URI；
- `hdfs_fuse_path`：通过 HDFS FUSE 暴露出来的本机绝对路径，必须位于 `runtime.storage.hdfs_fuse.mount_root` 下。

数据不允许把普通本地目录、相对路径、HTTP/S3/object-store URI 或未声明来源的路径伪装成启动输入。后续如果要支持对象存储，应显式扩展 schema 与校验器，而不是放宽字符串路径。

最小校验规则：

- `model.model_path` 必须是非空字符串；默认示例使用 `/mnt/hdfs/nano-ai/models/qwen`；
- `model.tokenizer_path` 若为空，默认等于 `model.model_path`，不需要额外声明 `source_type`；
- `data.source_type` 必须声明为 `hdfs_uri` 或 `hdfs_fuse_path`，并且在 `runtime.storage.allowed_input_sources` 内；
- `data.data_path` 必须存在且可读，可以是单文件或目录；如果是目录，必须能枚举到至少一个支持的数据文件；
- `data.prompt_column` 必须能在启动抽样校验中解析到，避免 worker 启动后才发现数据列错误；
- 数据的 HDFS URI 通过 HDFS client 做 exists/list/read probe；数据的 FUSE path 通过 POSIX stat/list/open probe，并校验路径没有逃逸出 `mount_root`；
- 校验失败统一抛出 `InvalidInputArtifactError`，错误信息必须包含失败字段、数据 source type、原始路径和具体原因。

`run.intent=validate` 与 `dry_run` 也必须执行以上校验；区别只是校验成功后不启动或不完整启动 runtime。

### 2.8 Hugging Face 格式的测试 artifact 准备

v0.1 的模型 ckpt 与纯文本训练数据默认按 Hugging Face 原生格式准备。模型 ckpt 可以直接作为普通目录写入 runtime YAML，例如 `/mnt/hdfs/nano-ai/models/qwen`，不需要 `source_type`；训练数据仍按 `data.source_type` 走显式 storage source 校验。

测试用最小默认资产：

- 模型 ckpt：`Qwen/Qwen3-0.6B`，保持 Hugging Face model snapshot 目录结构，例如 `config.json`、tokenizer 文件与 safetensors shard；
- 纯文本数据集：`roneneldan/TinyStories`，默认使用 `train` split 的 `text` 列，可保存成 `datasets.load_from_disk` 可读取的本地 Hugging Face datasets 目录；
- 训练数据默认保存完整 split；如果需要更小的 smoke-test 资产，可以显式传 `--dataset-max-bytes 100M` 或 `--dataset-max-rows 10000`；
- smoke test 可以只 materialize 前 N 行数据，但必须保留 `prompt_column/text_column` 元数据，避免后续 runtime 校验拿不到文本列。

仓库脚本 `scripts/download_hf_assets.py` 只负责下载与落盘 manifest，不负责启动 Ray、不写训练循环，也不把模型/数据字段变成第二套训练命令行参数。脚本参数只覆盖 artifact 准备本身，例如 `--model-id`、`--dataset-id`、`--dataset-split`、`--dataset-text-column`、`--dataset-max-rows`、`--dataset-max-bytes` 与 `--output-dir`。

---

## 3. 两种运行模式定义

### 3.1 Fully Sync 模式

### 语义

- 每个 train iteration 之前必须使用同一版本权重完成 rollout；
- rollout 与训练形成严格 barrier。

### 时序

1. trainer 发布 `W_t`；
2. rollout 全部 worker 切换到 `W_t` 并确认；
3. rollout 采样完成并回传 batch；
4. trainer 基于 batch 更新得到 `W_{t+1}`。

### 优点/成本

- 优点：收敛语义清晰、调试简单；
- 成本：吞吐受最慢阶段限制，资源空转更明显。

### 3.2 Standalone + Hybrid 模式

### 语义

- `rollout_only_gpus` 与 trainer 解耦并行，持续做 rollout；
- `shared_gpus` 在非训练窗口由 rollout replicas 使用，在训练窗口通过 lease 切给 trainer ranks；
- rollout 可使用 `W_t, W_{t-1}, ...` 的受控旧权重窗口；
- hybrid 不表示多机混合部署，而是单机 shared GPU pool 在 rollout/train role 之间切换 CUDA 执行权。

### 时序（简化）

1. 所有 rollout replicas 默认处于 rollout；
2. 触发训练窗口时，shared GPU 对应的 rollout replicas drain/offload；
3. rollout-only replicas 不等待训练，继续向 sample queue 输送样本；
4. trainer ranks 获得 shared GPU lease 后消费样本并更新权重；
5. rank0 发布新权重版本；
6. trainer ranks 释放训练状态，shared GPU lease 回到 rollout，相关 replicas 按策略激活新权重。

### 关键约束

- 版本滞后窗口 `max_policy_lag`；
- 样本 TTL 与丢弃规则；
- hybrid toggle 必须有 drain timeout、CUDA quiesce timeout 和失败回滚策略。

---

## 4. FSDP2 Trainer 详细设计

Trainer 侧的设计目标是：用尽量小的抽象把 PPO-like 训练主路径跑通，同时把 FSDP2、权重发布、样本版本检查和 GPU lease 管理边界分清楚。Trainer 不直接管理全局 rollout 生命周期，也不直接向 rollout worker 发命令；它只消费 `SampleQueueActor` 中已经入队且满足版本约束的样本，并发布新的权重版本。

### 4.1 角色边界

`TrainerGroup` 是逻辑组件，由下面几类对象组成：

- `TrainerCoordinatorActor`：CPU actor，负责 train window 编排、rank/world size 分配、batch 消费、group epoch 管理和错误归一化；
- `TrainerRankActor`：独立 trainer rank actor，Ray `num_gpus=0`，通过自动分配的 `train_gpu_i` custom resource 映射到 `shared_gpus`，负责 FSDP2 model/optimizer/scheduler、micro-batch forward/backward 和 optimizer step；
- `RendezvousState`：由 coordinator 维护的轻量状态，包含 `group_epoch`、`world_size`、`rank -> gpu_id`、`master_addr`、`master_port`、`timeout_sec`；
- `WeightExportTask`：rank0 或独立 CPU task，负责把训练状态导出成 rollout 可消费的权重 artifact；
- `MetricsActor`：接收 step time、tokens、loss、KL、grad norm、OOM、checkpoint/export 耗时等指标。

`TrainerCoordinatorActor` 可以是 Ray actor；`TrainerRankActor` 是独立 actor/进程，与 rollout actor 分离故障域。长生命周期 trainer actor 不申请 Ray `num_gpus=1`，而是使用 `train_gpu_{id}` 这类 role-scoped custom resource，并在执行 CUDA 前校验 `GpuLeaseManagerActor` 发放的 trainer lease。

### 4.2 TrainerRank 生命周期

每个 `TrainerRankActor` 只映射到 `shared_gpus` 上，生命周期由 GPU lease active-role gate 控制：

1. `ABSENT`：`rollout_only_gpus` 没有 trainer rank；
2. `INIT`：加载模型配置、tokenizer metadata、optimizer/scheduler 配置，但不得执行 CUDA kernel；
3. `READY`：训练 role 可被激活，持有必要 metadata；
4. `TRAIN_ACTIVE`：shared GPU 已经释放 rollout CUDA 状态并授予 trainer lease，rank 在首个 healthy epoch 初始化 process group，后续窗口复用该 group 并执行 FSDP2；
5. `DRAINING`：当前 micro-step 或 optimizer step 到达安全点；
6. `PAUSED`：保留训练 metadata，但不能继续执行；
7. `CPU_STANDBY` / `UNLOADED`：正常 toggle 优先把 model shard、optimizer state、scheduler/RNG/scaler 保留在 CPU standby；只有 fallback 或恢复路径才退化到仅保留 checkpoint/weight version/rank metadata；
8. `FAILED`：rank 失败，整个 `TrainerGroup` 必须视作失败并重建。

Trainer 不允许绕过 `GpuLeaseManagerActor` 直接调用 inactive rank。任何 `train_step` 调用都必须先检查 trainer lease、`lease_epoch` 和 `group_epoch`，并且 GPU 处于 trainer active 状态。

### 4.3 并行策略

v0.1 的 trainer 并行策略以 FSDP2 为核心：

- **Data parallel / FSDP world**：`fsdp_world_size == shared_gpus == trainer.num_ranks`；
- **Tensor parallel**：v0.1 不启用 trainer tensor parallel，配置面保留但必须为 `1`；
- **Pipeline parallel**：v0.1 不启用，配置面保留但必须为 `1`；
- **Gradient accumulation**：通过 micro-batch 累积得到 `global_batch_size`；
- **Sharding**：优先支持 `full_shard`，后续按需要扩展 `hybrid_shard`；
- **Precision**：默认 `bf16`，`fp16` 只作为可选配置；
- **Activation checkpointing**：作为显存控制开关进入 FSDP2 配置；
- **FSDP2 CPU offload**：指 active training 期间的 FSDP2 offload，v0.1 默认关闭，避免单机性能不可控；它不同于 hybrid toggle 之间必须执行的 CPU standby/residency 迁移。

推荐的 batch 关系：

```text
global_batch_size =
  per_rank_micro_batch_size
  * grad_accum_steps
  * fsdp_world_size
```

如果 YAML 显式给出 `global_batch_size`，`LaunchConfig` 校验器需要推导或校验 `per_rank_micro_batch_size` 与 `grad_accum_steps`。如果不能整除，应在 `ray.init(...)` 前 fail-fast。

### 4.4 TrainBatch 输入契约

Trainer 从 `SampleQueueActor` 消费的是训练 batch，而不是裸字符串 prompt。建议内部模型：

```yaml
train_batch_id: str
target_update_step: int
sample_refs: [ObjectRef]
num_sequences: int
num_tokens: int
policy_version_min: int
policy_version_max: int
policy_version_histogram: map[int, int]
created_at: ts
expires_at: ts
```

每个 sample 至少应包含：

- `prompt`、`response`、`tokens`、`attention_mask`；
- rollout policy 的 `policy_version`；partial rollout 样本还需要 `policy_segments`；
- rollout 侧采集到的 token logprobs，作为 PPO old logprobs；
- reward 或 reward ref；
- 生成配置、worker id、latency、finish reason 等 debug metadata。

Trainer 在 materialize batch 后必须执行：

1. 检查 `current_version - sample.policy_version <= max_policy_lag`，其中 partial rollout 的 `sample.policy_version` 取最旧 behavior version；
2. 检查 batch 内版本跨度是否超过配置；
3. 丢弃 TTL 过期样本；
4. 记录被丢弃样本数量和原因；
5. 对 batch 做 padding/packing/micro-batch split；
6. 将 `policy_version_histogram` 写入 train stats。

### 4.5 单步训练流程

一个 train window 内可以包含一个或多个 optimizer updates。单步流程：

1. `TrainerCoordinatorActor` 从 `SampleQueueActor` reserve 一个或多个 `TrainBatch`；
2. coordinator 将 batch refs 广播给所有 active ranks；
3. ranks materialize 本 rank 需要的 sample slice；
4. ranks 基于 rollout old logprobs、当前 policy logprobs、reward/advantage 计算 PPO loss；
5. FSDP2 执行 forward/backward、gradient reduction、gradient clipping；
6. 满足 accumulation 后执行 optimizer step；
7. coordinator 汇总 `TrainStats`；
8. rank0 判断是否到达 publish/checkpoint 周期；
9. 成功后 ack queue 消费；失败则释放 reservation 或标记 batch 不可重试，并把 shared GPU lease 切回 rollout。

Queue ack 必须晚于 optimizer step 成功。否则 trainer OOM 或 rank failure 会造成样本丢失但训练没有更新。失败路径不能把 batch 长期留在 reserved，也不能让 shared GPU 停在 trainer active 状态等待下一轮。

### 4.6 权重导出与发布

Trainer 侧的权重发布分成三步：

1. **Export**：从 FSDP2 state dict 导出 checkpoint 或 rollout 格式权重；
2. **Validate**：检查 manifest、文件完整性、tokenizer/chat template metadata、checksum；
3. **Register**：向 `WeightRegistryActor` 注册 `WeightMeta`。

建议权重 artifact 布局：

```text
weights/
  policy-v000123/
    manifest.json
    config.json
    tokenizer.json
    tokenizer_config.json
    generation_config.json
    model.safetensors.index.json
    model-00001-of-000NN.safetensors
```

`version_id` 由 `WeightRegistryActor` 单调分配或由 trainer step 单调映射，但必须全局唯一。`parent_version` 指向上一个成功训练基线。注册成功不代表所有 rollout workers 已经激活，只代表该版本可供激活。

权重发布和权重传输分开处理。trainer/exporter 只负责产出 `WeightMeta`、artifact/manifest 和 checksum；`WeightTransferPlanner` 在 rollout 激活前根据 `weight_transfer.method` 生成 `WeightTransferPlan`。这样未来多机时可以替换 transfer backend，而不改 FSDP2 trainer、vLLM rollout 或 registry 状态机。

v0.1 默认不把大权重放进 Ray object store。Ray 只传 `WeightMeta`、`WeightTransferPlan` 和少量 metadata，其中 URI/path 指向 backend loader 可读取的模型或 checkpoint artifact。

### 4.7 Trainer 失败语义

Trainer 失败以 group 为单位处理：

- 任一 `TrainerRank` OOM：当前 optimizer update 失败，释放 queue reservation，记录 `TRAINER_OOM`；
- 任一 rank 心跳丢失：整个 `TrainerGroup` 进入 failed，销毁 process group 并按 `group_epoch+1` 重建；
- checkpoint/export 失败：训练状态可以继续，但该版本不得注册为可用权重；
- 连续 OOM 超阈值：进入 `PAUSED`，等待用户调小 batch 或启用更激进 checkpointing；
- 部分 rank 成功、部分 rank 失败的 step 不允许被提交。

---

## 5. vLLM Rollout 详细设计

Rollout 侧的目标是持续生成带版本号的样本，并在权重切换、shared GPU lease 切换和 queue 背压下保持可控。Rollout 不负责训练语义，也不直接修改权重 registry；它只按指定权重版本生成，并把生成结果交给 sample/reward pipeline。

### 5.1 角色边界

Rollout 采用三层 actor 结构，由以下对象组成：

- `RolloutManagerActor`：全局 CPU actor，负责 rollout 生命周期、初始化和拉起 replica/worker actors、input backlog、queue 水位、权重同步、rollout/train 状态切换协同；
- `RolloutReplicaControllerActor`：每个 rollout DP replica 一个长生命周期 actor，Ray `num_gpus=0`，但按 TP 组占用该 replica 的 `rollout_gpu_i` custom resources；它构造 vLLM backend adapter，并管理权重激活、通信组、drain/offload/wake 和失败恢复；
- `RolloutWorkerActor`：每个 vLLM TP rank 一个 `num_gpus=0` actor / role boundary，维护 TP rank 元数据和未来 worker/shard 边界；它不单独抢占 `rollout_gpu_i`，CUDA-facing 行为必须通过 replica/backend gate 与 rollout lease；
- `PromptSource`：从 dataset reader 或 controller 提供 prompt batch；
- `RewardActorPool`：可选 CPU/GPU actors，负责 reward function、rule reward、model reward 或后处理；
- `SampleQueueActor`：接收已完成 sample refs；
- `WeightRegistryActor`：提供 latest/healthy weight version 和 activation 状态。

v0.1 的本地 smoke runtime 可以把 replica controller 和 worker 适配器简化到纯 Python 对象；真实 Ray/vLLM 路径已经把 `RolloutReplicaControllerActor` 接到 `VllmRolloutBackend` adapter。若 vLLM engine 需要更强进程隔离，可以在后续让 worker actor 或 backend 子进程承载更细粒度 shard，但 CUDA 执行仍必须受 GPU lease token gate 约束。

当前 runtime core 已把 `RolloutManagerCore` 做成 dataloader 与 rollout backend 之间的显式控制面：

- dataloader/controller 调用 `enqueue_prompts(prompts)` 把输入放入 manager 的 input backlog；
- manager 维护 `in_flight` 与 `in_flight_by_replica`，按 replica capacity 做 round-robin dispatch；
- `dispatch_from_backlog(...)` 同时检查 output queue depth、`queue_high_watermark`、`max_pending_rollout_refs` 展开的 in-flight 上限和本轮 `max_new_requests`；
- `ControllerCore.pump_rollout_until_blocked(...)` 会重复调用 dispatch/generate/submit，直到 backlog 清空、queue high watermark、capacity 或本轮预算阻塞；
- 当 output queue 水位已到 high watermark 时，manager 返回空 dispatch，而不是继续制造 sample refs；
- 权重切换时 `pause_for_weight(reason)` 会让新 dispatch fail-fast，activation 完成后 `resume()` 才恢复。

### 5.2 RolloutWorker 生命周期

Rollout role 的状态：

1. `INIT`：读取 tokenizer、generation config、vLLM engine config；
2. `READY`：engine 可被激活，但未必有最新权重；
3. `ACTIVATING_WEIGHT`：加载或切换目标 `WeightMeta`；
4. `ROLLOUT_ACTIVE`：接受 generation request；
5. `DRAINING`：停止接新请求，等待 in-flight 完成或取消；
6. `PAUSED`：保留 worker metadata，不执行 CUDA；
7. `UNLOADED`：释放 vLLM engine/KV cache/CUDA memory；
8. `FAILED`：worker 失败，等待 replica 或 pool 恢复。

Shared GPU 对应的 rollout replica 在 train window 前必须从 `ROLLOUT_ACTIVE` 走到 `UNLOADED` 或等价的 CUDA quiesced 状态；`rollout_only_gpus` 对应的 replicas 不参与 train barrier，持续保持 rollout。

### 5.3 GenerationRequest 契约

Rollout coordinator 发给 worker 的请求建议包含：

```yaml
request_id: str
prompt_batch_ref: ObjectRef
target_policy_version: int
generation:
  max_tokens: int
  temperature: float
  top_p: float
  stop: [str]
return_logprobs: true
deadline_at: ts
metadata:
  prompt_bucket: str
  controller_step: int
```

Worker 输出 `RolloutBatchResult`：

```yaml
request_id: str
worker_id: str
policy_version: int
weight_checksum: str
sample_ref: ObjectRef
num_sequences: int
num_tokens: int
finish_reason_histogram: map[str, int]
latency_ms: int
error: optional[str]
```

`return_logprobs` 对 PPO 主路径是硬约束；如果 rollout backend 无法返回 token logprobs，该 batch 不能作为 PPO train batch，只能进入 debug 或非 PPO 算法路径。

### 5.4 Rollout 并行策略

v0.1 的 rollout 并行以 data parallel 为主：

- 每个 rollout DP replica 一个 `RolloutReplicaControllerActor`，该 actor 按 TP 组占用 `tensor_parallel_size` 个 `rollout_gpu_i` custom resources；
- 每个 replica 内有 `tensor_parallel_size` 个 `RolloutWorkerActor` / TP rank role，记录自动分配的 GPU id，但长生命周期 actor 仍 `num_gpus=0`；
- 每个 worker 内部交给 vLLM continuous batching；
- `RolloutManagerActor` 按 prompt 长度桶和 replica 负载分发，避免长样本拖慢所有 worker；
- `max_pending_rollout_refs` 限制 worker 侧未入队或未消费的 sample refs；
- `queue_high_watermark` 触发限流；
- output queue 水位低于 high watermark 且存在 in-flight capacity 时，manager 从 input backlog 补采样。

当 `tensor_parallel_size > 1` 时，TP group 由 `RolloutReplicaControllerActor` 统一管理：

- 一个 replica controller 聚合多个 `RolloutWorkerActor`；
- replica 内统一创建 vLLM TP 通信组；
- replica 作为一个 logical rollout endpoint 接受 request；
- activation、drain、offload、wake、failure 都以 replica 为单位处理；
- replica 中任一 TP worker 失败，该 replica 整体 failed 或 rebuild。

### 5.5 Policy version 选择

Rollout 请求必须绑定一个 target policy version。选择策略由 `RolloutManagerActor` 执行：

- `fully_sync`：只允许 latest version，且 rollout 前必须所有 worker 激活同一版本；
- `standalone_hybrid`：允许 latest 与 bounded lagged versions 混合；
- `policy_pin.latest_ratio` / `lagged_ratio` 只影响请求分配比例，不改变 `max_policy_lag` 硬约束；
- registry 中标记 failed 或 unhealthy 的版本不得被选择；
- 已经超过 TTL 的 in-flight request 不再入队训练。

所有 sample 都必须记录实际生成时使用的 behavior policy 版本和 `weight_checksum`。非 partial rollout 样本可以只有单个 `policy_version`；partial rollout 样本必须额外记录 token span 级 `policy_segments`。不能用“请求时的 latest”替代实际 active 版本。

### 5.6 Shared GPU 上的 rollout drain

进入 train window 前，shared GPU 上 rollout drain 的步骤：

1. `ControllerActor` 通知 `RolloutManagerActor` 停止向占用 `shared_gpus` 的 replicas 发新请求；
2. 这些 replicas 和其 TP workers 标记 `DRAINING`；
3. in-flight generation 在 `rollout_drain_timeout_sec` 内完成；
4. 完成的 samples flush 到 `SampleQueueActor`；
5. 超时请求取消，并记录 cancellation reason；
6. replica controller / vLLM backend 释放 KV cache、engine CUDA memory 和可能的 CUDA graph；
7. `GpuLeaseManagerActor` 检查 CUDA quiesce 并递增 lease epoch；
8. shared GPUs 的 lease 切换到 trainer role。

这个流程只作用于 `shared_gpus` 对应的 rollout replicas。`rollout_only_gpus` 在 standalone_hybrid 模式下继续接收 prompt 并持续生成。

### 5.7 Partial Rollout 权重切换

v0.1 不做 `eager` / `lazy` / `staged` 三套 activation 策略，也不做小流量探针。权重 ready 后采用一个简单的全局暂停式 **partial rollout** 切换：暂停所有 rollout replicas 的 scheduler，同步新权重，然后让 in-flight requests 在新权重下继续 decode。

1. `WeightRegistryActor` 标记新版本为 `REGISTERED`；
2. `RolloutManagerActor` 停止向所有 rollout replicas 发新请求；
3. 所有 rollout replicas 进入 `PAUSING_FOR_WEIGHT`；
4. vLLM 执行 `pause_generation(mode="keep", clear_cache=true)`，冻结 in-flight requests；
5. 所有 rollout replicas / backend engines 释放旧 KV cache / prefix cache；
6. `RolloutReplicaControllerActor` 为 TP 组读取当前 rollout lease tokens；
7. replica backend 在校验 `(gpu_id, role=rollout, holder_id, lease_epoch)` 后加载同一个目标 `WeightMeta`；
8. replica backend 重新 prefill / rebuild 新权重下的 KV cache；
9. Controller 在 ack 前重新向 `GpuLeaseManagerActor` 校验 lease epoch 没有变化；
10. 所有 replicas 上报目标版本 active；
11. `WeightRegistryActor` 将该版本标记为 `ACTIVE_GLOBAL`；
12. vLLM 执行 `resume_generation()`，in-flight requests 继续生成；
13. `RolloutManagerActor` 恢复发新 prompt。

这个策略牺牲一小段 scheduler pause 时间，但实现面比 staged activation 简单：没有 worker 子集、没有灰度版本、没有 staged rollback。只要权重加载 API 报错、lease epoch 校验失败或 worker 进程崩溃，controller 就把本次切换视为失败，`WeightRegistryActor` 将目标版本标记为 `FAILED`，`RolloutManagerActor` 退出暂停态但不把失败版本作为 active；必要时进入 `PAUSED` 或整体重建。v0.1 不做“先切一部分确认正常再全量”的健康探针。

Partial rollout 的语义是：一个 response 可以包含旧权重生成的前缀和新权重生成的后缀。`clear_cache=true` 时，resume 后的 token 会在新权重下重新计算 KV cache，不继续复用旧权重的 stale KV。这样的 mixed-policy sample 可以进入训练，但不能再只用一个单值 `policy_version` 表达完整行为策略；`SampleRecord` 必须记录 `policy_segments`，并保留 token-level old logprobs。Trainer 做 policy lag guard 时按 sample 中最旧的 behavior version 保守计算。

---

## 6. Trainer-Rollout 数据与权重桥接协议

Bridge 层负责两件事：rollout 样本从生成侧进入训练侧，trainer 权重从训练侧进入 rollout 侧。它不能混入训练 loop 语义，也不能成为大权重广播通道。

### 6.1 样本数据路径

rollout 到 trainer 的路径：

1. `RolloutManagerActor` 选择 prompt batch、target policy version 与 rollout replica；
2. `RolloutReplicaControllerActor` 在 rollout leases 下调用 vLLM backend/TP group 生成 response、tokens、old logprobs；
3. `RewardActorPool` 计算 reward，或把 reward ref 挂到 sample metadata；
4. replica/backend 构造 `SampleRecordBatch` 并放入 Ray object store；
5. replica/backend 将 `ObjectRef` 和 batch metadata 提交给 `SampleQueueActor`；
6. `SampleQueueActor` 根据 TTL、policy lag、水位和 pending refs 决定 accept/drop/defer；
7. `TrainerCoordinatorActor` reserve batch；
8. train step 成功后 ack；
9. train step 失败后 release reservation 或标记不可重试，并恢复 shared GPU rollout lease。

`SampleQueueActor` 的内部状态建议分层：

- `pending`：已接收，未被 trainer reserve；
- `reserved`：已分配给某个 `train_batch_id`，等待 train ack；
- `acked`：训练成功消费，可释放 ObjectRef；
- `dropped`：TTL、policy lag、queue pressure 或 schema 错误导致丢弃；
- `failed`：sample ref 损坏或反序列化失败。

Queue 必须提供幂等提交：相同 `sample_id` 重复提交时，不能重复训练。建议用 `{sample_id, policy_version, worker_id}` 做去重键。

### 6.2 权重数据路径

trainer 到 rollout 的路径：

1. `TrainerRank(rank=0)` 或 exporter 生成临时 artifact；
2. exporter 写入临时目录，例如 `policy-v000123.tmp/`；
3. exporter 生成 manifest 与 checksum；
4. exporter 原子提交为 `policy-v000123/`；
5. exporter 调用 `WeightRegistryActor.register(meta)`；
6. registry 把版本标记为 `REGISTERED`；
7. `WeightTransferPlanner` 根据 `weight_transfer.method` 和 `ResolvedGpuPlan` 生成 `WeightTransferPlan`；
8. `RolloutManagerActor` 暂停所有 rollout replicas；
9. replica controllers 协调 TP workers 释放旧 KV cache，并按各自 `WeightShardSource` materialize 同一个目标版本；
10. replicas ack active target version；
11. registry 在所有要求的 workers ack 后更新 latest served version。

`weight_transfer.method` 目前有两种：

- `objectref`：trainer/exporter 把 rollout 可消费的权重对象或 shard refs 放进 Ray object store，`WeightTransferPlan.sources[*].kind=ray_object_ref`。这个路径实现简单，适合小模型测试、fake backend 或调试，但对真实大模型过重：object store 会复制/序列化大张量，跨 replica fan-out 容易挤压 sample refs。
- `locality_aware_checkpoint`：默认生产向路径。shared GPU 上的 rollout replica 在训练窗口之后会回到同一批物理 GPU，这些 GPU 已经持有 FSDP2 更新后的 rank-local 权重或 CPU standby state；对应 `WeightShardSource.kind=shared_gpu_reshard`，activation 只需要在 lease 切回 rollout 后把本地 trainer shard reshard 成 vLLM TP layout。rollout-only GPU 没有 trainer-resident 权重，对应 `WeightShardSource.kind=artifact_pull`，从 `artifact_uri` / `manifest_uri` 拉取需要的 vLLM shard。

这解决了 hybrid 场景里的非对称性：shared GPU 不做全量远程传输，只做本地格式转换/reshard；rollout-only GPU 通过版本 artifact hydrate。一个 rollout TP group 不允许横跨 rollout-only/shared 生命周期区域，因此 planner 不需要处理半个 TP group 本地、半个 TP group 远程的模糊状态。

`WeightTransferPlan` 是 per-version/per-replica 的控制面对象，不承载大 tensor：

```yaml
version_id: int
method: [objectref, locality_aware_checkpoint]
artifact_uri: str
manifest_uri: str
sources:
  rollout-dp-2:
    kind: shared_gpu_reshard
    target_gpu_ids: [4, 5]
    source_rank_ids: [0, 1]
    source_gpu_ids: [4, 5]
  rollout-dp-0:
    kind: artifact_pull
    target_gpu_ids: [0, 1]
    artifact_uri: /tmp/nano-rl-checkpoints/version-123
```

未来多机时，这个模块可以继续扩展 `kind`，例如 distributed checkpoint、NCCL/UCX peer transfer、RDMA object store 或 parameter-server style weight service；Controller 和 registry 仍只看版本、计划、ack 和失败事件。

Weight state 建议：

- `CREATING`：trainer/exporter 正在写；
- `REGISTERED`：artifact 校验通过，可以被激活；
- `ACTIVATING`：所有 rollout replicas 已暂停，正在批量加载；
- `ACTIVE_GLOBAL`：所有要求的 worker 已服务该版本；
- `FAILED`：artifact 不可用，或批量加载过程中任一必要 worker 失败；
- `DEPRECATED`：仍可回滚，但不再用于新请求；
- `GC_ELIGIBLE`：超过保留窗口，可以清理。

`REGISTERED` 与 `ACTIVE_GLOBAL` 必须分开。训练侧发布成功并不意味着 rollout 侧已经全部切换。

### 6.3 WeightMeta

```yaml
version_id: int
parent_version: int
created_at: ts
model_path: str
tokenizer_path: str
format: [hf, safetensors, vllm_compatible]
checksum: str
```

补充字段建议：

```yaml
trainer_step: int
artifact_uri: str
manifest_uri: str
tokenizer_hash: str
chat_template_hash: str
created_by: str
status: [registered, activating, active_global, failed, deprecated]
```

### 6.4 WeightTransferPlan

`WeightTransferPlan` 是独立模块的输出，不写入 `WeightMeta` 本体，避免 registry 变成传输后端的耦合点。它的职责是把一个全局权重版本拆成每个 rollout replica 的 materialization source：

- `ray_object_ref`：对应 `objectref` method；
- `shared_gpu_reshard`：目标 rollout TP group 的 GPU 全部是 shared GPU，并且这些 GPU 有对应 FSDP2 trainer rank；
- `artifact_pull`：目标 rollout TP group 不拥有 trainer shard，必须从 artifact/manifest hydrate。

每个 source 至少包含 `replica_id`、`target_gpu_ids`、`target_worker_ids`、`kind` 和 `reason`。`shared_gpu_reshard` 还包含 `source_rank_ids/source_gpu_ids`，`artifact_pull` 包含 `artifact_uri/manifest_uri`，`ray_object_ref` 包含 object ref key 或实际 Ray ref 的外部索引。实际大 tensor 不进入该 plan。

### 6.5 SampleRecord

```yaml
sample_id: str
policy_version: int  # 最旧 behavior policy version，用于保守 lag guard
policy_version_max: int
policy_segments:
  - version_id: int
    start_token: int
    end_token: int
    weight_checksum: str
prompt: str
response: str
tokens: [int]
logprobs: [float]
reward: float
advantages: [float]
meta:
  worker_id: str
  latency_ms: int
  precision_tag: str
```

补充字段建议：

```yaml
request_id: str
weight_checksum: str
finish_reason: str
created_at: ts
expires_at: ts
prompt_tokens: int
response_tokens: int
old_logprobs: [float]
partial_rollout: bool
reward_source: str
drop_reason: optional[str]
```

### 6.6 TrainBatch

`TrainBatch` 是 trainer 消费侧的聚合协议，不需要作为外部持久 schema，但实现时应有结构化模型：

```yaml
train_batch_id: str
sample_refs: [ObjectRef]
sample_ids: [str]
policy_version_min: int
policy_version_max: int
policy_version_histogram: map[int, int]
num_sequences: int
num_tokens: int
reserved_by: str
reserved_at: ts
expires_at: ts
```

`reserved_by` 通常是 `TrainerCoordinatorActor` 的 logical id。`reserved_at` 到 train ack 之间如果 coordinator 失联，queue 可以按 lease timeout 释放 reservation。

### 6.7 健康事件协议

最小事件集：

- `ROLLOUT_WORKER_STALE_WEIGHT`：worker 实际 active 版本与请求版本不一致；
- `ROLLOUT_QUEUE_BACKPRESSURE`：queue 水位或 pending refs 超阈值；
- `ROLLOUT_GENERATION_TIMEOUT`：generation 超过 request deadline；
- `SAMPLE_DROPPED_POLICY_LAG`：样本版本滞后超过 `max_policy_lag`；
- `SAMPLE_DROPPED_TTL`：样本过期；
- `TRAINER_OOM`：trainer rank OOM；
- `TRAINER_GROUP_FAILED`：FSDP2 group 失效；
- `WEIGHT_EXPORT_FAILED`：trainer 导出失败；
- `WEIGHT_DISTRIBUTION_FAILED`：rollout 激活失败；
- `SHARED_GPU_TOGGLE_TIMEOUT`：shared GPU lease 切换或 replica drain 超时；
- `SHARED_GPU_OFFLOAD_FAILED`：model/optimizer 或 rollout engine 无法从 GPU 安全 offload 到 CPU；
- `SHARED_GPU_HYDRATE_FAILED`：CPU standby state 无法恢复到 GPU active；
- `TRAINER_COMM_GROUP_UNHEALTHY`：process group 心跳、barrier 或 collective 顺序异常；
- `TRAINER_COMM_GROUP_REBUILT`：旧 `comm_epoch` 失效并完成整组重建；
- `OBJECT_STORE_PRESSURE`：Ray object store 压力过高。

事件 payload 至少包含：

```yaml
event_id: str
event_type: str
severity: [info, warning, error, fatal]
source_actor: str
gpu_id: optional[int]
policy_version: optional[int]
group_epoch: optional[int]
created_at: ts
details: map[str, any]
```

### 6.8 Ray 传输约定

- 控制消息：Pydantic model 直接作为 Ray actor method 参数；
- 样本批次：大 batch 放入 Ray object store，以 `ObjectRef` 在 actor 间传递；
- 权重版本：默认只传 `WeightMeta` 与 `WeightTransferPlan`，其中 `model_path` / `tokenizer_path` / artifact URI 指向 backend loader 可读取的普通 artifact path；只有显式设置 `weight_transfer.method=objectref` 时才允许把权重对象或 shard refs 放进 Ray object store；
- 指标事件：小 payload 直接发给 `MetricsActor`，高频 token 级明细先本地聚合再上报；
- 错误：actor method 抛出的异常由 `ControllerActor` 归一化为 health event。

Ray object store 默认只适合样本 batch、reward batch 和中等大小 metadata。除非处于 `objectref` transfer method 的显式实验路径，否则禁止用于：

- FSDP2 full state dict；
- vLLM 权重 shard；
- 大 tokenizer/model artifact；
- 长期保留的 checkpoint。

### 6.9 一致性边界

Bridge 层要保证下面的最小一致性：

- sample 进入 trainer 前必须知道真实 behavior policy version；
- partial rollout sample 进入 trainer 前必须带 `policy_segments`，不能把 mixed-policy response 伪装成单一行为策略；
- train batch 必须记录 batch 内版本分布；
- weight version 注册必须晚于 artifact 校验；
- rollout activation ack 必须来自实际加载该版本的 replica/worker；v0.1 要求所有 rollout replicas 批量切到同一目标版本后再恢复发 prompt；
- queue ack 必须晚于 optimizer step 成功；
- failed weight 不能被新 rollout request 选择；
- failed train step 不能消费掉 queue reservation。

standalone_hybrid 允许 bounded staleness，但不允许 unbounded async。任何绕过 `policy_version`、TTL、weight status 的数据路径都不应该进入 v0.1。

---

## 7. 调度与背压控制

调度层只做资源和节奏控制，不把 PPO loss、vLLM 生成细节或权重格式转换写死在 controller 中。Controller 的核心职责是：决定什么时候发 rollout、什么时候开 train window、什么时候限流、什么时候降级或暂停。

### 7.1 Ray placement 与资源声明

每个角色都显式声明资源，避免隐式抢占：

- GPU：`GpuLeaseManagerActor` 维护每张物理 GPU 的 active role、holder、lease epoch 和失败状态；
- trainer：通过自动分配到 `shared_gpus` 的 `TrainerRankActor` 形成 `TrainerGroup` ranks；
- rollout：通过自动分配到 `rollout_only_gpus + shared_gpus` 的 `RolloutReplicaControllerActor` 形成 rollout replicas，每个 replica 按 TP 组占用 `rollout_gpu_i` custom resources；
- reward：`num_reward_actors`、`cpus_per_actor` 或可选 GPU；
- queue/metrics/controller：默认 CPU actor。

`RayActorGraphLauncher` 根据这些配置创建 actor graph，并在 dry-run 阶段输出可检查的 actor/resource plan；真正启动由 `run.start_ray_actors` 控制。

`runtime.ray.gpu_manager.topology` 是物理 GPU 数量分配的源头，resolved config 会展开为 `ResolvedGpuPlan`。`placement.trainer.num_ranks` 必须等于 `shared_gpus`，`placement.rollout.num_replicas * placement.rollout.tensor_parallel_size` 必须等于 `rollout_only_gpus + shared_gpus`。用户不手写 GPU id；dry-run 和 resolved config 展示系统生成的 `gpu_id`。

### 7.2 Fully Sync controller loop

`fully_sync` 对应用户 YAML `mode=collocated`。`rollout_only_gpus=0`，所有 rollout GPUs 都来自 `shared_gpus`。

单个 iteration：

1. Controller 确认所有 shared GPUs 的 lease 处于 rollout role；
2. WeightRegistry 返回当前 latest healthy version `W_t`；
3. RolloutManager 确认所有 rollout replicas 激活 `W_t`；
4. Controller 发起本轮 rollout requests；
5. SampleQueue 达到本轮 train batch 目标后停止发新 rollout；
6. 所有 shared rollout replicas drain；
7. shared GPU leases 切换到 trainer；
8. TrainerCoordinator 建立或恢复 FSDP2 group；
9. Trainer 执行一个或多个 optimizer updates；
10. rank0/exporter 发布 `W_{t+1}`；
11. shared GPU leases 切换回 rollout；
12. rollout workers 激活 `W_{t+1}` 后进入下一轮。

这个模式下 `max_policy_lag` 应为 `0`，`policy_pin.lagged_ratio` 必须为 `0`。如果任一 worker 没有激活 `W_t`，本轮 rollout 不应开始。

### 7.3 Standalone Hybrid controller loop

`standalone_hybrid` 对应用户 YAML `mode=disaggregated`。`rollout_only_gpus` 持续生成，`shared_gpus` 在 rollout/train 之间切换。

主循环：

1. rollout-only replicas 与当前处于 rollout role 的 shared replicas 持续接收 prompt；
2. SampleQueue 按版本、TTL、水位接收样本；
3. Controller 观察 queue depth、token depth、policy lag、trainer idle time；
4. 达到 train trigger 后，只 drain shared rollout replicas；
5. rollout-only replicas 继续生成，但受 queue 背压和 policy lag guard 限制；
6. shared GPU leases 切换到 trainer；
7. trainer 消费 queue 中满足 freshness 的 batch；
8. trainer 发布新 weight version；
9. shared GPU leases 切换回 rollout；
10. RolloutManager 暂停所有 rollout replicas，批量加载新版本并在全局 active 后恢复发 prompt。

Train trigger 可以由下面任一条件触发：

- queue 中可训练 tokens 达到 `trainer.global_batch_size` 所需规模；
- trainer idle 时间超过阈值；
- 距离上次 train window 超过最大间隔；
- policy lag 接近上限，需要推进新版本；
- 手动控制命令要求 flush/train。

如果 queue 还不足以形成有效 train batch，Controller 不应强行开启 train window，除非处于 shutdown/checkpoint/drain 场景。

### 7.4 队列水位

- `queue_high_watermark`：超过则 rollout 降速/限流；
- output queue 未达到 high watermark 且 rollout manager 仍有 in-flight capacity 时，从 input backlog 补采样。

建议增加 token 维度水位，因为 sequence 数不能真实反映训练成本：

- `queue_high_token_watermark`：可选，按 response tokens 控制总 backlog；
- `queue_target_token_watermark`：可选，低于目标水位时提高补采样预算；
- `max_batch_policy_span`：一个 train batch 内允许的最大版本跨度。

### 7.5 ObjectRef 背压

Ray-native 模式下还需要限制未消费的 ObjectRef 数：

- `max_pending_rollout_refs`：rollout 侧最多挂起 sample refs；
- `max_pending_train_refs`：trainer 侧最多挂起 batch refs；
- 超过限制后 `ControllerActor` 暂停发起新 rollout，直到 sample queue 被消费。

ObjectRef 背压要同时看三处：

- worker 本地 in-flight request；
- SampleQueue 已接收未 ack refs；
- TrainerCoordinator 已 reserve 未完成 refs。

如果 object store 压力超过阈值，优先策略是暂停发新 rollout、丢弃过期或过旧样本、释放已 ack refs。不要通过把大对象转存到普通 Python actor 字段来绕过 object store 压力。

### 7.6 自适应策略

- 基于 token/s、P95 latency、trainer consume rate 调整 `num_sequences`；
- 独立控制 prompt 长度桶，避免长样本阻塞全局。

可以先实现保守策略：

- queue 高水位：停止向 lagged version 发请求，只保留 latest；
- queue 持续高水位：降低每 worker 并发；
- trainer 持续空闲：提高 rollout batch size 或增加 prompt 发放；
- rollout latency P95 超阈值：降低长 prompt 桶占比；
- sample drop rate 超阈值：进入 degraded sync 或临时收紧 policy lag。

### 7.7 训练消费策略

- 优先消费最新版本样本；
- 滞后过大样本按策略丢弃或降权。

Trainer reserve batch 时的推荐顺序：

1. 过滤 TTL 过期样本；
2. 过滤 `policy_lag > max_policy_lag` 样本；
3. 优先选择版本跨度小的样本；
4. 尽量组成 token 数接近目标的 batch；
5. 对长样本和短样本分桶，减少 padding 浪费；
6. 如果 queue 中只有旧样本，按策略 drop/degrade，而不是无界等待。

---

## 8. 状态管理与通信组创建维护

状态管理的原则是：每类状态只有一个权威 owner，其他组件通过查询、订阅或 actor call 获取状态，不能本地猜测。这样可以避免 rollout 以为自己已经切到新权重、trainer 以为 queue batch 已被消费、GPU lease manager 以为 CUDA 已释放这类隐蔽错误。

### 8.1 状态 owner

| 状态 | 权威 owner | 说明 |
| --- | --- | --- |
| 运行模式 | `ControllerActor` | `fully_sync` / `standalone_hybrid` / degraded / paused |
| GPU active-role lease | `GpuLeaseManagerActor` | 同一张 shared GPU 同一时刻只能一个 role 有 CUDA 执行权 |
| FSDP2 group | `TrainerCoordinatorActor` | rank 映射、group epoch、rendezvous 参数 |
| Rollout worker pool | `RolloutManagerActor` / `RolloutReplicaControllerActor` | replica 可调度状态、in-flight、TP group、版本分布 |
| Weight version | `WeightRegistryActor` | version status、latest healthy、activation ack |
| Sample lifecycle | `SampleQueueActor` | pending/reserved/acked/dropped refs |
| Health events | `MetricsActor` 或 health sink | 事件聚合、阈值判断输入 |

任何组件都可以缓存状态用于性能优化，但缓存必须带 `epoch` 或 `version`，并且在执行动作前重新确认。

### 8.2 启动状态序列

启动顺序：

1. `main.py` 读取 YAML；
2. config schema 校验；
3. 归一化为 `LaunchConfig`；
4. 输入 artifact fail-fast 校验；
5. `RayClusterController` 初始化 Ray：`auto` 先连接已有 cluster，失败后创建本机 cluster；
6. 创建 placement group；
7. 创建 `ControllerActor`；
8. Controller 创建 `WeightRegistryActor`、`SampleQueueActor`、`MetricsActor`；
9. Controller 创建 `GpuLeaseManagerActor` 并加载 resolved `gpu_plan`；
10. Controller 创建 rollout manager、replica controllers、rollout workers 和 trainer ranks；
11. registry 注册初始 `W_0`；
12. rollout workers 激活 `W_0`；
13. FSM 从 `BOOTSTRAP` 进入目标 active 状态。

在第 12 步之前，不允许开始生成样本。在第 4 步失败时，不应该启动 Ray 或创建任何 GPU actor。

### 8.3 GpuLeaseManager active-role gate

`GpuLeaseManagerActor` 是 CUDA 执行权 owner。它需要提供最小 gate：

```text
grant(role, gpu_id, holder_id, reason)
current_lease(role, gpu_id, holder_id)
assert_active(role, gpu_id, holder_id, expected_epoch)
states()
mark_failed(gpu_id, reason)
```

所有 role method 进入 CUDA 前都要检查：

- GPU 是否 failed；
- active role 是否匹配；
- lease epoch 是否匹配；
- 当前是否处于 draining/paused；
- 是否超过 toggle timeout。

如果检查失败，应 fail-fast 返回结构化错误，而不是等待 inactive role 未来变 active。这样 direct client 即使绕过 coordinator 调用 role actor，也不会在错误窗口执行 CUDA work。

### 8.4 Trainer 通信组创建

FSDP2 process group 由 `TrainerCoordinatorActor` 统一创建和维护。建议协议：

```yaml
group_epoch: int
world_size: int
ranks:
  - rank: 0
    gpu_id: 4
    actor_id: trainer-rank-0
  - rank: 1
    gpu_id: 5
    actor_id: trainer-rank-1
master_addr: str
master_port: int
backend: nccl
timeout_sec: int
```

创建流程：

1. Controller 请求 shared GPUs 的 rollout replicas drain/offload；
2. `GpuLeaseManagerActor` 将 shared GPUs lease 授予 trainer ranks；
3. TrainerCoordinator 生成新的 `group_epoch`；
4. TrainerCoordinator 固定 `rank -> gpu_id`；
5. TrainerCoordinator 下发 rendezvous 参数；
6. 每个 rank 初始化 `torch.distributed` process group；
7. 每个 rank 初始化 FSDP2 wrapper、optimizer、scheduler；
8. TrainerCoordinator 等待所有 rank heartbeat；
9. group 标记为 healthy；
10. train window 开始。

上述创建流程只发生在首个 healthy `group_epoch` 或故障 rebuild。后续正常 train window 复用同一个 process group：ranks 从 CPU standby hydrate model/optimizer state 到 GPU，进入 train enter barrier，然后开始 collectives。

正常 train window 退出流程：

1. TrainerCoordinator 停止发新 train step；
2. ranks 到达安全点；
3. ranks barrier；
4. rank0/exporter 如需发布权重则先完成 export/register；
5. ranks 将 model shard、optimizer state、scheduler/RNG/scaler 等状态 offload 到 CPU standby；
6. ranks 释放训练临时 CUDA tensors/cache；
7. process group 保持存活但 idle，不再发起 collective；
8. GPU lease manager 标记 trainer role quiesced；
9. Controller 允许 shared GPU lease 切回 rollout。

真正销毁或重建 process group 只发生在 GPU/rank failure、NCCL error、作业恢复或 shutdown 路径。重建时必须生成新的 `group_epoch` / `comm_epoch`，并让旧 epoch 的 pending train batch 和 late ack 全部失效。

如果任一 rank 初始化失败或 heartbeat 超时，整个 group epoch 失败。不要尝试把部分 rank 拼到旧 group 中继续训练。

### 8.5 Rollout 通信组维护

v0.1 支持在配置层表达 rollout TP；每个 DP replica 的 `RolloutReplicaControllerActor` 维护一个 TP group：

- `replica_id -> gpu_ids`；
- `worker_id -> gpu_id`；
- `worker_id -> active_policy_version`；
- `worker_id -> in_flight_requests`；
- `worker_id -> health_state`；
- `worker_id -> precision_tag/task_pool`。

当 `tensor_parallel_size > 1` 时：

- replica controller 统一创建 vLLM 通信上下文；
- replica 暴露单个 logical endpoint 给 `RolloutManagerActor`；
- activation/drain/offload/wake/failure 以 replica 为单位；
- group epoch 与 GPU lease epoch 都要进入 request metadata。

### 8.6 Epoch 与幂等

所有长生命周期动作都要带 epoch：

- `lease_epoch`：GPU active role 或 holder 每次变化时递增；
- `role_epoch`：role unload/reload 后递增；
- `trainer_group_epoch`：每次 FSDP2 group 重建递增；
- `rollout_pool_epoch`：worker pool membership 改变时递增；
- `weight_version`：权重单调递增；
- `queue_lease_epoch`：batch reservation lease 递增。

Actor method 需要具备幂等性：

- 重复 `register(weight_meta)` 不应创建两个版本；
- 重复 `submit_sample(sample_id)` 不应重复训练；
- 过期 epoch 的 `activate_weight` ack 应被忽略；
- 过期 epoch 的 `train_step` result 不应 ack queue；
- 重复 `grant(role, gpu_id, holder_id)` 如果状态已经满足，应返回当前 lease 而不是重做危险动作。

### 8.7 心跳与故障检测

最小 heartbeat：

- GPU lease heartbeat：gpu alive、active role、holder、lease epoch、CUDA memory summary；
- trainer heartbeat：rank alive、group epoch、current step、last collective time；
- rollout heartbeat：worker alive、active version、in-flight、tokens/s；
- queue heartbeat：depth、token depth、pending refs、reserved refs、drop rate；
- registry heartbeat：latest registered、latest active、failed versions。

Controller 不直接从单个指标做剧烈动作。推荐做连续 N 次阈值判断：

- policy lag 连续超阈值：进入 `DEGRADED_SYNC`；
- rollout workers 全失联：进入 `PAUSED`；
- trainer OOM 连续超阈值：进入 `PAUSED`；
- weight activation 连续失败：冻结新版本，保留旧 healthy version；
- object store pressure 连续超阈值：暂停 rollout 并清理过期 refs。

---

## 9. 一致性与收敛保障

在 standalone/hybrid 下建议引入：

- **Policy Lag Guard**：`current_version - sample.policy_version <= K`；
- **Importance Ratio Clipping**：限制 off-policy 偏移；
- **Freshness-aware batching**：同 batch 内版本跨度受限。

并在日志中暴露：

- `policy_lag_histogram`
- `accepted_vs_dropped_samples`
- `effective_kl_by_policy_age`

---

## 10. 配置系统设计

建议使用单一主配置 `config.yaml`。配置文件承载可复现训练设置，是本机资源、并行维度、模式、模型、数据、训练超参和执行意图的权威来源。`main.py` 不提供逐字段命令行覆盖。

```yaml
run:
  intent: train
  emit_resolved_config: true
  # false: emit backend-integrated plan only; true: start Ray actor graph.
  start_ray_actors: false

runtime:
  backend: ray
  local:
    num_gpus: 8
    cpus: 64
  storage:
    allowed_input_sources: [hdfs_uri, hdfs_fuse_path]
    hdfs:
      cli: hdfs
      read_probe_timeout_sec: 30
    hdfs_fuse:
      mount_root: /mnt/hdfs
      require_mount: true
      read_probe_timeout_sec: 10
  ray:
    # auto: connect existing Ray cluster first; create local cluster on failure.
    address: auto
    namespace: nano-rl
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
        rollout_drain_timeout_sec: 60
        train_drain_timeout_sec: 60
        cuda_quiesce_timeout_sec: 30
        offload:
          trainer_model: cpu_pinned
          trainer_optimizer: cpu_pinned
          rollout_engine: vllm_sleep
          vllm_sleep_level: 2
          cpu_memory_budget_gb: 512
          residual_gpu_memory_budget_mb: 2048
    placement:
      trainer:
        num_ranks: 4
        gpus_per_rank: 1
      rollout:
        num_replicas: 4
        gpus_per_replica: 2
        tensor_parallel_size: 2
      reward:
        num_actors: 2
        cpus_per_actor: 4

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
  model_path: /mnt/hdfs/nano-ai/models/qwen
  tokenizer_path: /mnt/hdfs/nano-ai/models/qwen

data:
  source_type: hdfs_uri
  data_path: hdfs://namenode/datasets/prompts.jsonl
  prompt_column: prompt

algorithm:
  name: ppo
  max_steps: 1000
  learning_rate: 1.0e-6
  seed: 42

trainer:
  backend: fsdp2
  global_batch_size: 1024
  checkpoint_dir: /tmp/nano-rl-checkpoints
  fsdp2:
    mixed_precision: bf16
    sharding: full_shard
rollout:
  backend: vllm
  partial_rollout:
    enabled: true
    pause_mode: keep
    clear_cache: true
    mixed_policy_samples: train
  hybrid:
    enabled: true
    precision_mix: {bf16: 0.5, fp8: 0.5}
weight_transfer:
  method: locality_aware_checkpoint
  allow_rollout_only_artifact_pull: true
  max_versions_in_flight: 2
control:
  max_policy_lag: 2
  sample_ttl_sec: 300
  queue_high_watermark: 20000
  max_pending_rollout_refs: 128
  max_pending_train_refs: 16
```

配置来源规则：

- 主来源：YAML config；
- 默认值：只用于 YAML 未显式填写的低风险字段；
- 不设计逐字段命令行覆盖；`run.intent`、资源拓扑、模式和训练参数都写入 YAML；
- `run.start_ray_actors` 是是否真正启动 Ray actor graph 并执行 Ray training loop 的唯一 YAML 开关；为 `false` 时仍会输出含 backend adapter 构造参数的 `ray_launch_plan`；
- `runtime.ray.address=auto` 是默认启动策略：先连接已有 Ray cluster，连接失败则创建本机 cluster；显式非 `auto` address 连接失败时直接报错，避免误连到本地 runtime；
- `trainer.checkpoint_dir` 是真实 FSDP2 rank0/exporter 写入版本化 checkpoint 的目录，后续 rollout/vLLM 通过 `WeightMeta` 指向该 artifact；
- `weight_transfer.method` 控制 trainer-to-rollout 权重传输：`objectref` 走 Ray object store refs，`locality_aware_checkpoint` 让 shared GPU 本地 reshard、rollout-only GPU 从 artifact/manifest hydrate；
- 当 `rollout_only_gpus > 0` 且 `weight_transfer.method=locality_aware_checkpoint` 时，必须允许 `allow_rollout_only_artifact_pull=true`，否则没有 trainer-resident 权重的 rollout-only GPU 无法切到新版本；
- 环境变量只允许用于定位默认 config 文件或展开 YAML 中显式引用的本机路径，不隐式覆盖训练语义。

### 10.1 参数归一化与校验

用户 YAML 与 `main.py` 输出的 `LaunchConfig` 必须分层：YAML 接收用户部署形态，`LaunchConfig` 持有归一化后的内部 canonical mode。

- YAML `mode=collocated` 归一化为 `LaunchConfig.mode=fully_sync`；
- YAML `mode=disaggregated` 归一化为 `LaunchConfig.mode=standalone_hybrid`；
- `sync` / `async` 不作为 v0.1 用户入口别名，避免把部署形态误写成时间语义；
- `fully_sync` / `standalone_hybrid` 不作为用户 YAML 输入值，只出现在 resolved config、日志和 FSM 内部；
- 模型配置不包含 `source_type` 字段；`model.model_path` / `model.tokenizer_path` 按普通 artifact path 归一化，不要求 HDFS/HDFS-FUSE；
- `data.source_type` 必须在 `runtime.storage.allowed_input_sources` 内，且 v0.1 只允许 `hdfs_uri` / `hdfs_fuse_path`；
- `data.source_type=hdfs_uri` 时，对应 URI 必须使用 `hdfs://` scheme，并通过 HDFS client 的 exists/list/read probe；
- `data.source_type=hdfs_fuse_path` 时，对应路径必须是绝对路径、位于 `runtime.storage.hdfs_fuse.mount_root` 下，并通过 mount/stat/list/read probe；
- `model.tokenizer_path` 未填写时归一化为 `model.model_path`；
- 输入 artifact 校验失败必须在 `ray.init(...)` 之前以 `InvalidInputArtifactError` fail-fast；
- `runtime.local.num_gpus` 必须等于或大于 `rollout_only_gpus + shared_gpus + idle_gpus`；
- `runtime.ray.gpu_manager.enabled` 与 `runtime.ray.gpu_manager.lease_manager` 在 v0.1 都必须为 true；
- 归一化为 `fully_sync` 后，`rollout_only_gpus` 必须为 0，rollout 与 trainer 共用同一组 `shared_gpus`；
- 归一化为 `standalone_hybrid` 后，`rollout_only_gpus` 与 `shared_gpus` 都必须大于 0，前者持续 rollout，后者在 rollout/train 窗口之间切换；
- `runtime.ray.placement.trainer.num_ranks` 必须等于 `shared_gpus`；
- `runtime.ray.placement.rollout.num_replicas * tensor_parallel_size` 必须等于 `rollout_only_gpus + shared_gpus`；
- `rollout_only_gpus` 和 `shared_gpus` 都必须能被 rollout `tensor_parallel_size` 整除，避免一个 vLLM TP group 横跨两种生命周期区域；
- `weight_transfer.method=locality_aware_checkpoint` 且存在 rollout-only GPU 时，`allow_rollout_only_artifact_pull` 必须为 true；
- `runtime.ray.gpu_manager.hybrid_toggle.offload.trainer_model` 与 `trainer_optimizer` 必须显式配置为 CPU residency，v0.1 推荐 `cpu_pinned`；
- `hybrid_toggle.offload.rollout_engine=vllm_sleep` 时必须配置 `vllm_sleep_level`；若 backend 不支持 sleep/offload，resolved config 必须降级为 `teardown_and_reload` 并在 dry-run 中给出启动成本提示；
- `hybrid_toggle.offload.residual_gpu_memory_budget_mb` 用于 offload 后的显存余量检查，允许保留 CUDA context/NCCL bookkeeping，但不得掩盖 model/optimizer/KV cache 未释放；
- `rollout.partial_rollout.enabled=true` 时必须使用 `pause_mode=keep` 和 `clear_cache=true`，并要求 `SampleRecord` 输出 `policy_segments` 与 token-level old logprobs；
- `trainer.fsdp2` 的 world size 必须与 `parallel.trainer.fsdp_world_size`、`runtime.ray.placement.trainer.num_ranks` 一致；
- `rollout.tensor_parallel_size` 必须与 `parallel.rollout.tensor_parallel_size`、`runtime.ray.placement.rollout.tensor_parallel_size` 一致；
- `fully_sync` 模式下不允许配置需要旧权重采样的 `policy_pin.lagged_ratio > 0`；
- `standalone_hybrid` 模式下必须显式配置 `max_policy_lag` 与 sample TTL。

---

## 11. 可观测性设计

最小指标集：

- trainer：step_time、tokens_per_step、grad_norm、oom_count；
- rollout：tokens_per_sec、p50/p95 latency、error_rate；
- bridge：queue_depth、queue_wait_ms、weight_sync_time；
- quality：reward_mean、kl、policy_lag、sample_drop_rate。

日志要求：所有 sample/train batch 均带 `policy_version`。

---

## 12. 故障恢复

- rollout worker 崩溃：自动重启并回滚到最近健康权重版本；
- 权重下发失败：标记版本不可用，不阻断 trainer（standalone 模式）；
- trainer OOM：自动降 micro-batch 或触发保护性暂停。
- Ray actor 异常退出：`ControllerActor` 根据角色类型决定重启 actor、重建 placement group 或进入 degraded 状态；
- `GpuLeaseManagerActor` 异常退出：Controller 暂停所有 CUDA 新请求，恢复 lease state 后再继续；恢复失败则相关 GPU 标记 failed；
- shared GPU toggle 超时：停止向相关 rollout replica 或 trainer rank 分发新任务，相关 group 进入 `PAUSED`，不得让 inactive role 继续执行；
- trainer rank 部分失败：整个 `TrainerGroup` 视为失败并整体重建，避免 FSDP2 rank 状态不一致；
- object store 压力过高：暂停 rollout 并清理已过期 sample refs。

Checkpoint 建议：

- trainer ckpt 与 `latest_served_weight_version` 一并持久化；
- 恢复时允许 rollout 先用旧权重热身，再渐进追平。

---

## 13. 里程碑（仅文档阶段）

### M0（本次）

- 完成架构文档、协议草案、目录规划、配置草案。

### M1（最小可运行）

- 单机 Ray runtime 下跑通 quantity-only GPU plan 与 `GpuLeaseManagerActor` 编排：`rollout_only_gpus` 持续采样，`shared_gpus` 可切到 train window。

### M2（standalone_hybrid）

- 引入 shared GPU lease toggle、Ray-native 异步队列、policy lag guard、样本 TTL、ObjectRef 背压。

### M3（稳定性与运维）

- health event、自动降级、可观测性面板；多机 Ray cluster 暂不进入 v0.1。

## 14. 建议目录结构（初始化）

```
nano-rl/
  main.py
  README.md
  requirements.txt
  LICENSE
  docs/
    plans/
      plan-design.md
      plan-mock.md
      ref-flashrl-design.md
  nano_rl/
    config.py
    runtime/
      controller.py
      coordinators.py
      roles.py
      sample_queue.py
      slot.py
      weight_registry.py
      ray/
        driver.py
        actors.py
        placement.py
  docs/
    protocols/
      weight-meta.schema.yaml
      sample-record.schema.yaml
      health-events.md
```

---

## 15. 与 FlashRL 参考的关系

- 借鉴点：配置驱动、运行时注入、hybrid rollout 思想；
- 扩展点：加入完整 mode/fsm、异步一致性边界、FSDP2-vLLM 的权重版本协议；
- 目标差异：nano-rl 不沿用 FlashRL 的 raw-Kubernetes platform path；v0.1 选择 Ray-native runtime 作为默认执行模型。


---

## 16. 参考 FlashRL（lastweek/FlashRL）后的补充约束

- 采用 **local-first 与 standalone 共用同一 runtime 协议** 的思路：单机路径不做特化分叉；
- 将 examples 作为主入口而不仅是 demo，至少维护 collocated 与 disaggregated/hybrid 两条最小样例；
- 用户启动入口固定为仓库根目录 `main.py`，内部可委托给 package 代码，但不要让用户绕过 `LaunchConfig` 校验直接启动角色 actor；
- 不复制 FlashRL 的 CRD/operator/pod shim；对应能力由 Ray driver、ControllerActor、placement group 和 actor handle 承担。
