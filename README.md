# nano-rl

A tiny but complete RL framework blueprint focused on:

- FSDP2 trainer runtime
- vLLM rollout runtime
- single-machine Ray-native actor execution runtime
- collocated and disaggregated execution profiles, normalized to fully-sync and standalone+hybrid internal modes

This repository is currently in design-first stage.

## Current Implementation

The initial Python runtime skeleton now includes:

- `main.py`: YAML entrypoint and resolved-config emitter.
- `nano_rl.config`: Pydantic config models and `LaunchConfig` normalization.
- `nano_rl.runtime.slot`: deterministic GPU plan models and CPU-only lease manager core.
- `nano_rl.runtime.sample_queue`: in-memory `SampleQueueActor` core.
- `nano_rl.runtime.weight_registry`: in-memory `WeightRegistryActor` core.
- `nano_rl.runtime.coordinators`: rollout manager backlog/pump core and trainer coordinator core.
- `nano_rl.runtime.controller`: dry-run controller plan and local smoke iteration.
- `nano_rl.runtime.ray`: Ray driver boundary, CPU actor wrappers, and custom-resource launch plan.

Local validation:

```bash
python3 -m pytest -q
python3 main.py --config docs/examples/disaggregated.yaml --emit-resolved-config
python3 scripts/smoke_local_runtime.py --config docs/examples/disaggregated.yaml --prompt "hello"
```

## Documents

- `AGENTS.md`: repo-specific operating instructions for coding agents.
- `ref-flashrl-design.md`: FlashRL design summary used as reference.
- `plan-design.md`: detailed architecture/design plan for nano-rl.
- `docs/architecture/design.html`: browser-friendly living design document with
  SVG component and rollout-flow diagrams.
- `docs/architecture/mode-fsm.md`: operational state machine for mode transitions and degradation.
- `docs/protocols/runtime-config.schema.yaml`: Ray-native runtime configuration schema.
- `docs/examples/collocated.yaml`: collocated config where every GPU toggles between rollout and train together.
- `docs/examples/disaggregated.yaml`: disaggregated config with rollout-only GPU count plus shared rollout/train GPU count.

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
