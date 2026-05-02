# FlashRL 设计参考总结（基于 `lastweek/FlashRL`）

> 参考仓库：<https://github.com/lastweek/FlashRL>
> 说明：本文件基于你指定的仓库（`lastweek/FlashRL`）重写。

## 1) 项目定位（不是单一优化插件）

FlashRL 在 README 里明确定位为 **learning-first 的 LLM post-training RL 项目**，同时覆盖两条路径：

- **local-first**：本地实验/白盒 agent/GRPO 训练；
- **raw-Kubernetes**：把同一套 runtime 渲染为 `FlashRLJob` 后分布式运行。

这说明它更像“框架 + 运行平台双形态”，而非仅 rollout 侧插件。

## 2) 核心运行时模型

README 给出的核心组件分工可抽象为：

- `FlashRL`：顶层 runtime 组装与生命周期；
- `GRPOController`：RL 主循环、batch、优化、协调；
- `rollout`：生成响应；
- `reward`：打分；
- `training backends`：actor/reference 权重更新；
- `serving backend`：推理服务与权重激活；
- `observability`：日志/指标/ckpt。

**关键启发**：它把“算法控制器”与“后端角色（rollout/serving/training）”分开，天然支持替换不同后端实现。

## 3) 代码结构上的设计信号

按 README 的项目布局，FlashRL 有五个主要面：

- `flashrl/framework`：核心 runtime + controller + 各角色；
- `flashrl/examples`：从最小到复杂的示例阶梯；
- `flashrl/platform`：K8s config compiler / CRD operator / pod runtime；
- `docs`：架构与训练文档；
- `scripts`：平台辅助与 smoke 工具。

对 nano-rl 的直接价值是：

1. **框架内核与平台封装分层**；
2. **examples 作为一等公民**（不仅 demo，而是使用路径）；
3. **CLI 与 runtime entrypoint 分离**（用户入口 vs pod 入口）。

## 4) 对 rollout/train 模式设计的可借鉴点

虽然 FlashRL README 不是以“collocated vs disaggregated”或“fully sync vs standalone hybrid”术语表述，但其架构支持我们抽取：

- 控制器负责统一 loop 协调（可承载 sync 语义）；
- 角色进程化后可部署为独立服务（可承载 standalone/hybrid 语义）；
- serving backend 与 training backend 解耦（利于权重版本策略与滞后控制）。

## 5) 对 nano-rl 的落地建议（基于该参考）

结合你的目标（FSDP2 trainer + vLLM rollout）：

- 在 nano-rl 中保留 `Controller` 作为唯一 loop 语义源；
- rollout / serving / trainer 用统一协议解耦，避免“模式切换要改算法代码”；
- local-first（单机）与 standalone（分布式）共享同一配置 schema；
- 先把 examples 铺好：
  - `minimal_collocated_ppo`（collocated，内部 `fully_sync`）
  - `minimal_disaggregated_hybrid_ppo`（disaggregated，内部 `standalone_hybrid`）

## 6) 与上版总结的差异说明

上一版我错误参考了非你指定的仓库；本版已按你给的 `lastweek/FlashRL` 重新整理，后续 `plan-design.md` 将以本版为准。
