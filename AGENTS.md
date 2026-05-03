# Agent Instructions for nano-rl

This file is the operational guide for coding agents working in this repository.
It applies to the whole repo unless a more specific `AGENTS.md` exists in a
subdirectory.

## What This File Is For

- Give future agents the repo-specific context they should load first.
- Preserve architectural decisions that should not be rediscovered or casually
  changed.
- Point agents at the right validation commands and durable design documents.
- Keep agent work aligned with the current design-first stage of the project.

## Current Project State

`nano-rl` is currently in a design-first / early-implementation stage. An
initial Python runtime skeleton exists, but do not assume a complete trainer,
rollout, or Ray execution implementation exists yet. Treat the existing
documents, schemas, and core runtime tests as the source of truth before adding
code.

Canonical documents:

- `README.md`: short project overview and document index.
- `ref-flashrl-design.md`: reference extraction from `lastweek/FlashRL`.
- `plan-design.md`: main architecture plan for nano-rl.
- `docs/architecture/design.html`: browser-friendly living design document with
  SVG component and rollout-flow diagrams; update it with architecture changes.
- `docs/architecture/mode-fsm.md`: operating mode state machine.
- `docs/protocols/*.yaml`: draft protocol/config schemas.
- `docs/examples/collocated.yaml`: collocated runtime config.
- `docs/examples/disaggregated.yaml`: disaggregated runtime config.
- `nano_rl/`: initial runtime skeleton with config models, resolved GPU plan,
  lease manager, queue, registry, coordinators, metrics, and Ray wrapper
  boundaries.
- `tests/`: narrow validation tests for config normalization and runtime cores.

## Architecture Invariants

- The core target is a small but complete RL framework, not a single rollout
  plugin.
- User-facing `mode` values should describe deployment topology:
  `collocated` and `disaggregated`.
- These user-facing modes normalize to internal canonical modes:
  `collocated -> fully_sync` and `disaggregated -> standalone_hybrid`.
- Do not reintroduce `sync` / `async` as user-facing mode aliases; those names
  describe time semantics rather than the deployment shape users choose in YAML.
- Do not accept `fully_sync` / `standalone_hybrid` as raw user YAML `mode`
  values; they are internal canonical values for resolved config, logs, and FSM.
- The repository-level user startup path is `main.py`. It should load a YAML
  config and produce a validated `LaunchConfig` before any runtime actor is
  started.
- v0.1 only needs to support a single machine. Do not add multi-node Ray,
  Kubernetes, Slurm, or SSH launch behavior unless the design is explicitly
  reopened.
- Training launches must express resource topology as quantities such as local
  GPU count, rollout-only GPU count, shared rollout/train GPU count, rollout
  DP/TP, and trainer ranks in YAML. Do not require users to hand-write physical
  GPU ids; the system must expand a deterministic resolved GPU plan.
- The trainer backend target is PyTorch FSDP2.
- The rollout backend target is vLLM.
- The v0.1 execution layer is Ray-native actor runtime. Ray is not an optional
  platform adapter in this design stage.
- GPU ownership is lease-based: `GpuLeaseManagerActor` is a CPU actor that owns
  physical GPU active-role state and lease epochs. It does not request
  `num_gpus=1`.
- Rollout and trainer execution actors are separate failure domains. Long-lived
  rollout/trainer actors should not both request Ray `num_gpus=1`; use
  role-scoped custom resources such as `rollout_gpu_4` / `train_gpu_4` plus
  lease-token checks before CUDA work.
- GPU topology modes are quantity based: `rollout_only_gpus`,
  `shared_gpus`, and optional `idle_gpus`. Rollout actors are unique within the
  rollout assignment set, trainer ranks are unique within the trainer assignment
  set, and the two sets may overlap through `shared_gpus`.
- Shared GPUs must switch through an explicit lease/toggle state machine. Only
  one role may be active on a shared GPU at a time, and inactive roles must not
  execute CUDA kernels.
- Local-first and standalone/distributed paths should share the same runtime
  protocol surface instead of diverging into separate semantics.
- The controller/orchestrator owns loop semantics. Backend roles such as
  trainer, rollout, serving, queue, and weight bus should remain replaceable.
- `RolloutManagerActor` owns prompt backlog, in-flight accounting, queue
  backpressure response, and rollout pump scheduling. Dataloaders should feed
  prompts into the manager instead of calling rollout workers directly.
- Weight and sample protocols must carry policy/version metadata. Do not add
  data paths that bypass `policy_version` / weight version accounting.
- In `standalone_hybrid`, enforce bounded staleness with policy lag, sample TTL,
  and drop/degrade behavior rather than unbounded async training.

## Implementation Guidance

- Before implementing or changing architecture, write the design update into
  `plan-design.md` and, when relevant, the schemas/examples under `docs/`.
- Keep `docs/architecture/design.html` synchronized with architecture changes
  that affect components, Ray actors, GPU leases, runtime flows, schemas, or
  examples.
- When the user accepts a design direction with wording such as "不错，就这样干",
  update `plan-design.md`, `docs/architecture/design.html`, and relevant
  schemas/examples in the same pass; do not wait for a separate reminder.
- Keep `main.py` thin: locate/read the YAML config, validate, normalize, emit the
  resolved config when requested, and hand off to the Ray driver. Do not put the
  training loop directly in `main.py`.
- Preserve YAML-first semantics. Do not add per-field command-line overrides;
  `run.intent`, resource topology, mode, and training parameters belong in YAML.
- Keep schemas and examples synchronized when changing protocol fields.
- Keep `runtime.ray.gpu_manager.topology` consistent with placement fields:
  `trainer.num_ranks == shared_gpus`, and
  `rollout.num_replicas * rollout.tensor_parallel_size ==
  rollout_only_gpus + shared_gpus`.
- Require `rollout_only_gpus` and `shared_gpus` to be divisible by rollout
  `tensor_parallel_size`, so a vLLM TP group never straddles lifecycle regions.
- Prefer `swap_on_toggle` for shared GPUs in v0.1. A future `dual_resident`
  strategy is allowed only when the inactive role still cannot run CUDA work.
- Use Pydantic or a structured parser for protocol/config handling once code is
  introduced; avoid ad hoc string parsing for YAML/JSON data.
- Keep examples as first-class entry points. The planned minimum examples are:
  `minimal_collocated_ppo` and `minimal_disaggregated_hybrid_ppo`.
- Keep the user entry point separate from role entry points such as trainer,
  rollout, and serving workers.
- Add narrow tests with new code. For protocol changes, include schema or model
  validation tests.
- Do not introduce Kubernetes, Slurm, SSH, or multi-node Ray launchers in v0.1.

## Validation Expectations

The repo is mostly documentation today, so validation is lightweight:

- For Markdown edits, ensure headings and links stay coherent.
- For YAML schema/example edits, parse the changed files with `pyyaml` or an
  equivalent parser.
- Once Python package code exists, add and run the relevant unit tests before
  handing work back.

## Git Hygiene

- The working tree may contain user changes. Never revert unrelated files.
- Keep edits scoped to the requested task.
- Do not convert design documents into implementation stubs unless the user asks
  for implementation.
