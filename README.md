# nano-rl

A tiny but complete RL framework blueprint focused on:

- FSDP2 trainer runtime
- vLLM rollout runtime
- single-machine Ray-native actor execution runtime
- collocated and disaggregated execution profiles, normalized to fully-sync and standalone+hybrid internal modes

This repository is currently in a design-first / early backend-integrated stage.

## Current Implementation

The initial Python runtime skeleton now includes:

- `main.py`: YAML entrypoint and resolved-config emitter.
- `nano_rl.config`: Pydantic config models and `LaunchConfig` normalization.
- `nano_rl.runtime.slot`: deterministic GPU plan models and CPU-only lease manager core.
- `nano_rl.runtime.sample_queue`: in-memory `SampleQueueActor` core.
- `nano_rl.runtime.weight_registry`: in-memory `WeightRegistryActor` core.
- `nano_rl.runtime.weight_transfer`: pluggable trainer-to-rollout weight transfer planner.
- `nano_rl.runtime.coordinators`: rollout manager backlog/pump core and trainer coordinator core.
- `nano_rl.runtime.controller`: dry-run controller plan and local smoke iteration.
- `nano_rl.runtime.offload`: shared GPU residency/offload/hydrate state machine.
- `nano_rl.runtime.backends`: lazy-import vLLM rollout and FSDP2 trainer backend adapters with fake test backends.
- `nano_rl.runtime.ray`: Ray driver boundary, CPU actor wrappers that own backend adapters, actor graph launcher, custom-resource launch plan, and driver-side training loop orchestration.

`RayDriver.train()` always builds the resolved runtime and Ray launch plan.
The non-mock example configs set `run.start_ray_actors: false`, so training
emits the backend-integrated plan without creating long-lived Ray actors. The
mock collocated/disaggregated examples set `run.start_ray_actors: true` and
run the full Ray actor training loop with mock backends: prompt ingestion,
rollout generation, reward scoring, sample queue reservation, trainer optimize,
weight export, and rollout weight reactivation. Real vLLM/FSDP2 execution uses
the same `main.py` path, but still requires Ray, vLLM, PyTorch/FSDP2, model
artifacts, `trainer.checkpoint_dir`, and trainer rendezvous metadata to be
ready.
When `runtime.ray.address: auto`, startup first tries to attach to an existing
Ray cluster. If none is reachable, the Ray startup controller creates a local
single-machine Ray cluster with the resolved role-scoped custom resources.

Local validation:

```bash
python3 -m pytest -q
python3 main.py --config recipes/disaggregated.yaml --emit-resolved-config
python3 main.py --config recipes/mock_collocated.yaml --skip-artifact-validation
python3 scripts/smoke_local_runtime.py --config recipes/disaggregated.yaml --prompt "hello"
```

## Documents

- `AGENTS.md`: repo-specific operating instructions for coding agents.
- `docs/plans/ref-flashrl-design.md`: FlashRL design summary used as reference.
- `docs/plans/plan-design.md`: detailed architecture/design plan for nano-rl.
- `docs/plans/plan-mock.md`: mock runtime design plan.
- `docs/architecture/design.html`: browser-friendly living design document with
  SVG component and rollout-flow diagrams.
- `docs/architecture/mode-fsm.md`: operational state machine for mode transitions and degradation.
- `docs/protocols/runtime-config.schema.yaml`: Ray-native runtime configuration schema.
- `docs/protocols/weight-transfer-plan.schema.yaml`: trainer-to-rollout weight transfer plan schema.
- `recipes/collocated.yaml`: collocated config where every GPU toggles between rollout and train together.
- `recipes/disaggregated.yaml`: disaggregated config with rollout-only GPU count plus shared rollout/train GPU count.

## Asset Download

Test checkpoints and text datasets are prepared as Hugging Face artifacts first,
then referenced from the runtime config through the configured storage path.

```bash
python scripts/download_hf_assets.py --output-dir artifacts/hf
```

By default the script downloads `Qwen/Qwen3-0.6B` and the complete
`roneneldan/TinyStories` `train` split, keeping the `text` column in Hugging
Face `datasets.load_from_disk` format. For smaller smoke-test assets, add
`--dataset-max-bytes 100M` or `--dataset-max-rows 10000`. A full
`Qwen/Qwen3-0.6B` checkpoint is larger than 100MB, so script tests should use
`--model-allow-pattern config.json` instead of pulling the full checkpoint.

## License

MIT
