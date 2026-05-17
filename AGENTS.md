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

`nano-rl` is currently in a design-first / early backend-integrated stage. A
Python runtime skeleton exists, and the Ray actor wrappers are connected to
lazy vLLM/FSDP2 backend adapters, but do not assume a complete production
trainer/rollout loop exists yet. Real FSDP2 execution still depends on model
artifacts, `trainer.checkpoint_dir`, and process-group rendezvous metadata.
Treat the existing documents, schemas, and core runtime tests as the source of
truth before adding code.

Canonical documents:

- `README.md`: short project overview and document index.
- `docs/plans/ref-flashrl-design.md`: reference extraction from `lastweek/FlashRL`.
- `docs/plans/plan-design.md`: main architecture plan for nano-rl.
- `docs/plans/plan-mock.md`: mock runtime design plan.
- `docs/architecture/design.html`: browser-friendly living design document with
  SVG component and rollout-flow diagrams; update it with architecture changes.
- `docs/architecture/mode-fsm.md`: operating mode state machine.
- `docs/protocols/*.yaml`: draft protocol/config schemas.
- `recipes/collocated.yaml`: collocated runtime config.
- `recipes/disaggregated.yaml`: disaggregated runtime config.
- `nano_rl/`: runtime skeleton with config models, resolved GPU plan, lease
  manager, queue, registry, coordinators, metrics, backend adapters,
  residency/offload state machine, and Ray wrapper/launcher boundaries.
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
- Rollout replica actors claim the `rollout_gpu_i` custom resources for their
  whole TP group; long-lived actors still use Ray `num_gpus=0`.
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
- Real backend execution boundaries live under `nano_rl.runtime.backends`,
  `nano_rl.runtime.offload`, and `nano_rl.runtime.ray.launcher`. Keep imports
  for vLLM, torch, FSDP2, and Ray startup lazy so CPU-only tests still run.
- `run.start_ray_actors` controls whether `RayDriver.train()` actually starts
  the Ray actor graph. With the default `false`, train emits a backend-integrated
  plan only.
- When `run.start_ray_actors` is true and `runtime.ray.address` is `auto`,
  startup must try to connect to an existing Ray cluster first and create a
  local single-machine Ray cluster only if that connection fails. Explicit
  non-`auto` Ray addresses must fail fast instead of silently falling back to
  local runtime.
- Weight and sample protocols must carry policy/version metadata. Do not add
  data paths that bypass `policy_version` / weight version accounting.
- Trainer-to-rollout weight transfer is an independent module. Keep
  `weight_transfer.method` pluggable; `objectref` is the explicit Ray object
  store path, while the default `locality_aware_checkpoint` path should reshard
  weights already resident on shared GPUs and hydrate rollout-only GPUs from
  artifact/manifest paths.
- In `standalone_hybrid`, enforce bounded staleness with policy lag, sample TTL,
  and drop/degrade behavior rather than unbounded async training.

## Implementation Guidance

- Before implementing or changing architecture, write the design update into
  `docs/plans/plan-design.md` and, when relevant, `docs/protocols/` schemas and
  `recipes/` runtime configs.
- Keep `docs/architecture/design.html` synchronized with architecture changes
  that affect components, Ray actors, GPU leases, runtime flows, schemas, or
  examples.
- When the user accepts a design direction with wording such as "不错，就这样干",
  update `docs/plans/plan-design.md`, `docs/architecture/design.html`, and relevant
  schemas/recipes in the same pass; do not wait for a separate reminder.
- Keep `main.py` thin: locate/read the YAML config, validate, normalize, emit the
  resolved config when requested, and hand off to the Ray driver. Do not put the
  training loop directly in `main.py`.
- Preserve YAML-first semantics. Do not add per-field command-line overrides;
  `run.intent`, resource topology, mode, and training parameters belong in YAML.
- Keep schemas and examples synchronized when changing protocol fields.
- Keep `docs/protocols/weight-transfer-plan.schema.yaml`, examples, and the
  Python `nano_rl.runtime.weight_transfer` module synchronized when changing
  trainer-to-rollout weight movement.
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

## Python Environment

- In this repository, use `.venv-nano-rl/bin/python3` as the Python interpreter
  for Python commands, tests, scripts, and validation.
- Prefer commands such as `.venv-nano-rl/bin/python3 -m pytest ...` and
  `.venv-nano-rl/bin/python3 main.py ...` rather than relying on the system
  Python.
- If an interactive shell already resolves `python3` to
  `.venv-nano-rl/bin/python3`, using `python3` is acceptable after confirming it
  with `which python3`.

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
