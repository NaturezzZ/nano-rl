# Mode FSM 设计（fully_sync / standalone_hybrid）

## 启动入口模式归一化

YAML 配置中的 `mode` 优先表达资源拓扑，而不是训练循环是否同步：

- `collocated`：rollout 与 trainer 共用同一组 `hybrid` GPU slots，归一化为内部状态机模式 `fully_sync`
- `disaggregated`：存在独立 `rollout_only` slots，同时 `hybrid` slots 在 rollout/train 之间切换，归一化为内部状态机模式 `standalone_hybrid`

`sync` / `async` 不再作为用户入口别名，因为它们描述的是时间语义，容易掩盖 v0.1 真正需要用户选择的资源部署形态。`fully_sync` / `standalone_hybrid` 只出现在 resolved config、日志和 FSM 内部。Mode FSM 内部只处理归一化后的 canonical mode，避免状态转换逻辑同时维护两套命名。

## 单机 GPU Slot 语义

v0.1 只考虑单机 Ray runtime。GPU 分为两类：

- `rollout_only`：持续做 standalone rollout，不参与 trainer rank；
- `hybrid`：非训练窗口做 rollout，训练窗口进入 drain/release/load 流程后切换为 trainer rank。

进入 train window 时只切换 `hybrid` slots，`rollout_only` slots 不进入训练 barrier，继续向 sample queue 供样本。

## Hybrid Slot 切换状态机

Hybrid slot 的状态机描述的是同一张 GPU 上 rollout role 与 trainer role 的 residency 变化。`GpuSlotActor` 是唯一能推进状态的 owner；`ControllerActor` 和两个 coordinator 只能提交 phase request，不能绕过 slot owner 直接唤醒 inactive role。

```text
ROLLOUT_ACTIVE
  -> ROLLOUT_DRAINING
  -> ROLLOUT_OFFLOADING_CPU
  -> TRAIN_HYDRATING_GPU
  -> TRAIN_READY_BARRIER
  -> TRAIN_ACTIVE
  -> TRAIN_DRAINING
  -> TRAIN_OFFLOADING_CPU
  -> ROLLOUT_WAKING_GPU
  -> ROLLOUT_READY_BARRIER
  -> ROLLOUT_ACTIVE
```

每个状态的责任：

- `ROLLOUT_ACTIVE`：vLLM engine 持有 GPU weights/KV cache，trainer rank 只保留 CPU standby state，trainer process group 存活但 idle。
- `ROLLOUT_DRAINING`：停止接新 prompt，等待 in-flight generation 完成或按 deadline 取消，已完成样本 flush 到 `SampleQueueActor`。
- `ROLLOUT_OFFLOADING_CPU`：释放 rollout CUDA residency。优先使用 vLLM sleep/offload；不支持时 teardown engine 并保留 `RolloutStateHandle`。
- `TRAIN_HYDRATING_GPU`：从 CPU `TrainStateBundle` 恢复 FSDP2 model shard、optimizer state、scheduler/RNG/scaler 到 GPU。
- `TRAIN_READY_BARRIER`：所有 hybrid ranks 使用相同 `comm_epoch` 到达 train enter barrier；少一个 rank 就 fail-fast。
- `TRAIN_ACTIVE`：只允许 trainer 发起 CUDA kernel 和 FSDP/NCCL collective，rollout role 保持 paused。
- `TRAIN_DRAINING`：完成当前 micro-step，等待 async collective work，进入 train exit barrier。
- `TRAIN_OFFLOADING_CPU`：model shard 与 optimizer state 回到 CPU standby，释放 CUDA tensors/cache，但不销毁 trainer process group。
- `ROLLOUT_WAKING_GPU`：按目标 `WeightMeta` wake/reload vLLM weights，再分配 KV cache 预算。
- `ROLLOUT_READY_BARRIER`：rollout-capable slots 完成版本激活后，`RolloutCoordinatorActor` 才恢复发 prompt。

## 通信组保护规则

FSDP2 的 process group 生命周期长于一次 train window。正常 role toggle 不 destroy/reinit process group，只切换 tensor residency：

- bootstrap 固定 rank mapping 和 rendezvous metadata，首个 train window 创建稳定的 `TrainerCommSession(world_id, rank_mapping, store_endpoint, comm_epoch)`；
- `ROLLOUT_ACTIVE` 期间 trainer process group 保持 idle，不发 collective；
- 进入 train 前使用 Ray control-plane 收齐所有 `TRAIN_HYDRATING_GPU` 完成信号，再进入 torch distributed barrier；
- 退出 train 前所有 ranks 必须等待 async work 并进入 barrier，然后才能 offload tensors；
- 任一 rank offload/hydrate/barrier 失败时，整组进入 `PAUSED` 或 rebuild，不能让部分 ranks 带旧 optimizer state 继续训练；
- rebuild 必须生成新的 `comm_epoch`，旧 epoch 的 pending train batch 与 checkpoint cursor 失效。

保留 process group 时可能保留少量 CUDA context/NCCL bookkeeping 显存。正常 offload 的目标是释放 model、optimizer、activation、KV cache 等大块 residency；如果 `cuda_quiesce_timeout_sec` 后 residual GPU memory 超过预算，应视为 offload 失败，而不是在正常 toggle 路径上销毁 process group。

Rollout 侧通信资源与 trainer 通信资源分离。v0.1 默认 `tensor_parallel_size=1`；未来若 vLLM tensor parallel 大于 1，必须由 `RolloutGroupActor` 以组为单位 sleep/wake/offload，不允许把 rollout group 与 FSDP2 group 合并。

## 状态定义

- `BOOTSTRAP`：初始化配置、Ray runtime、角色 actor、健康探测
- `FULLY_SYNC_ACTIVE`：fully_sync 运行中
- `STANDALONE_HYBRID_ACTIVE`：standalone_hybrid 运行中
- `DEGRADED_SYNC`：standalone_hybrid 故障回退到同步保守模式
- `PAUSED`：人为或自动暂停（OOM/队列失控）
- `STOPPED`：安全终止

## 状态转换

1. `BOOTSTRAP -> FULLY_SYNC_ACTIVE`
   - 条件：`mode=fully_sync` 且 `TrainerGroup` / `RolloutActorPool` 健康
2. `BOOTSTRAP -> STANDALONE_HYBRID_ACTIVE`
   - 条件：`mode=standalone_hybrid` 且 `SampleQueueActor` + `WeightRegistryActor` 健康
3. `STANDALONE_HYBRID_ACTIVE -> DEGRADED_SYNC`
   - 条件：policy lag 持续超阈值、样本过期率超阈值、或 weight publish 连续失败
4. `DEGRADED_SYNC -> STANDALONE_HYBRID_ACTIVE`
   - 条件：连续 N 个窗口健康（lag、drop_rate、OOM 恢复）
5. `*_ACTIVE -> PAUSED`
   - 条件：严重错误（trainer OOM 连续触发、rollout actor 全失联、Ray object store 压力失控）
6. `PAUSED -> FULLY_SYNC_ACTIVE/STANDALONE_HYBRID_ACTIVE`
   - 条件：人工恢复或自动恢复策略通过

## 守护阈值（建议默认）

- `max_policy_lag = 2`
- `max_sample_drop_rate = 0.15`
- `max_weight_publish_failures = 3`
- `max_consecutive_trainer_oom = 2`

## 恢复策略

- 从 `STANDALONE_HYBRID_ACTIVE` 进入 `DEGRADED_SYNC` 时：
  - 立即冻结 lagged policy 输入；
  - rollout 仅允许 latest policy；
  - 将 queue backlog 与 pending ObjectRef 清理到安全水位。
