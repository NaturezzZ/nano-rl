# nano-rl

A tiny but complete RL framework blueprint focused on:

- FSDP2 trainer runtime
- vLLM rollout runtime
- single-machine Ray-native actor execution runtime
- collocated and disaggregated execution profiles, normalized to fully-sync and standalone+hybrid internal modes

This repository is currently in design-first stage.

## Documents

- `AGENTS.md`: repo-specific operating instructions for coding agents.
- `ref-flashrl-design.md`: FlashRL design summary used as reference.
- `plan-design.md`: detailed architecture/design plan for nano-rl.
- `docs/architecture/design.html`: browser-friendly living design document with
  SVG component and rollout-flow diagrams.
- `docs/architecture/mode-fsm.md`: operational state machine for mode transitions and degradation.
- `docs/protocols/runtime-config.schema.yaml`: Ray-native runtime configuration schema.
- `docs/examples/collocated.yaml`: collocated config where every GPU toggles between rollout and train together.
- `docs/examples/disaggregated.yaml`: disaggregated config with standalone rollout slots plus hybrid train slots.

## Asset Download

Test checkpoints and text datasets are prepared as Hugging Face artifacts first,
then referenced from the runtime config through the configured storage path.

```bash
python scripts/download_hf_assets.py \
  --output-dir artifacts/hf \
  --model-id Qwen/Qwen3-0.6B \
  --dataset-id roneneldan/TinyStories \
  --dataset-split train \
  --dataset-text-column text \
  --dataset-max-bytes 100M
```

`--dataset-max-bytes` defaults to `100M` and caps the saved UTF-8 bytes from the
dataset text column. For even smaller smoke tests, add `--dataset-max-rows
10000`. The cap applies to training data preparation; a full `Qwen/Qwen3-0.6B`
checkpoint is larger than 100MB, so script tests should use
`--model-allow-pattern config.json` instead of pulling the full checkpoint.

## License

MIT
