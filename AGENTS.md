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

`nano-rl` is currently in a design-first stage. Do not assume a complete Python
package or runtime implementation exists yet. Treat the existing documents and
schemas as the source of truth before adding code.

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
- Training launches must express resource topology parameters such as local GPU
  count, rollout-only GPU slot allocation, hybrid GPU slot allocation, and
  parallel dimensions in YAML. Do not build a second training-parameter surface
  in command-line flags.
- The trainer backend target is PyTorch FSDP2.
- The rollout backend target is vLLM.
- The v0.1 execution layer is Ray-native actor runtime. Ray is not an optional
  platform adapter in this design stage.
- Ray GPU ownership is slot-based: create one `GpuSlotActor` per physical GPU
  slot, and let that actor own the Ray `num_gpus=1` token for the slot.
- Role actors/roles such as rollout and train live under `GpuSlotActor`
  ownership. Do not create two independent Ray actors that both request
  `num_gpus=1` for the same GPU.
- GPU slot modes are `rollout_only`, `hybrid`, and optional `idle`.
  `rollout_only` slots keep doing standalone rollout continuously. `hybrid`
  slots do rollout when not training and toggle into trainer ranks during train
  windows.
- Hybrid GPU slots must switch through an explicit toggle state machine. Only
  one role may be active on a hybrid slot at a time, and inactive roles must not
  execute CUDA kernels.
- Local-first and standalone/distributed paths should share the same runtime
  protocol surface instead of diverging into separate semantics.
- The controller/orchestrator owns loop semantics. Backend roles such as
  trainer, rollout, serving, queue, and weight bus should remain replaceable.
- Weight and sample protocols must carry policy/version metadata. Do not add
  data paths that bypass `policy_version` / weight version accounting.
- In `standalone_hybrid`, enforce bounded staleness with policy lag, sample TTL,
  and drop/degrade behavior rather than unbounded async training.

## Implementation Guidance

- Before implementing or changing architecture, write the design update into
  `plan-design.md` and, when relevant, the schemas/examples under `docs/`.
- Keep `docs/architecture/design.html` synchronized with architecture changes
  that affect components, Ray actors, GPU slots, runtime flows, schemas, or
  examples.
- Keep `main.py` thin: locate/read the YAML config, validate, normalize, emit the
  resolved config when requested, and hand off to the Ray driver. Do not put the
  training loop directly in `main.py`.
- Preserve YAML-first semantics. Do not add per-field command-line overrides;
  `run.intent`, resource topology, mode, and training parameters belong in YAML.
- Keep schemas and examples synchronized when changing protocol fields.
- Keep `runtime.ray.gpu_manager.slots` consistent with placement fields:
  `trainer.num_ranks == hybrid`, and
  `rollout.num_actors * gpus_per_actor == rollout_only + hybrid`.
- Prefer `swap_on_toggle` for hybrid slots in v0.1. A future `dual_resident`
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
