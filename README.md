# nano-rl

A tiny but complete RL framework blueprint focused on:

- FSDP2 trainer runtime
- vLLM rollout runtime plus a direct Hugging Face Transformers rollout backend
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
- `nano_rl.runtime.backends`: lazy-import vLLM, Hugging Face Transformers, and mock rollout adapters plus FSDP2/mock trainer adapters.
- `nano_rl.runtime.ray`: Ray driver boundary, CPU actor wrappers that own backend adapters, actor graph launcher, custom-resource launch plan, and driver-side training loop orchestration.

`RayDriver.train()` always builds the resolved runtime and Ray launch plan.
The non-mock example configs set `run.start_ray_actors: false`, so training
emits the backend-integrated plan without creating long-lived Ray actors. The
mock collocated/disaggregated examples set `run.start_ray_actors: true` and
run the full Ray actor training loop with mock backends: prompt ingestion,
rollout generation, reward scoring, sample queue reservation, trainer optimize,
weight export, and rollout weight reactivation. Real vLLM/FSDP2 and direct
HuggingFace/FSDP2 execution use the same `main.py` path, but still require Ray,
the selected rollout backend dependencies, PyTorch/FSDP2, model artifacts,
`trainer.checkpoint_dir`, and trainer rendezvous metadata to be ready.
When `runtime.ray.address: auto`, startup first tries to attach to an existing
Ray cluster. If none is reachable, the Ray startup controller creates a local
single-machine Ray cluster with the resolved role-scoped custom resources.

Local validation:

```bash
python3 -m pytest -q
python3 main.py --config recipes/disaggregated.yaml --emit-resolved-config
python3 main.py --config recipes/mock_collocated.yaml --skip-artifact-validation
python3 main.py --config recipes/collocated_qwen3.yaml --emit-resolved-config
python3 scripts/smoke_local_runtime.py --config recipes/disaggregated.yaml --prompt "hello"
```

## YAML Configuration Reference

Runtime config is YAML-first. `main.py` loads one YAML file, validates it with
`nano_rl.config.RuntimeConfig`, normalizes it into `LaunchConfig`, and then
hands execution to `RayDriver`. The schema source of truth is
`docs/protocols/runtime-config.schema.yaml`; the notes below are the practical
field guide for the current v0.1 surface.

Top-level sections:

- `run`: startup intent and whether to create real Ray actors.
- `mock`: optional global defaults for deterministic mock modules.
- `runtime`: local machine, storage policy, Ray actor/resource topology.
- `mode`: user-facing deployment topology.
- `parallel`: trainer and rollout parallel dimensions.
- `model`: model/tokenizer artifact paths.
- `data`: prompt input source.
- `algorithm`: PPO loop settings.
- `trainer`: FSDP2 or mock trainer backend.
- `rollout`: vLLM, Hugging Face, or mock rollout backend.
- `reward`: reward backend; currently mock deterministic reward.
- `weight_transfer`: trainer-to-rollout weight movement policy.
- `control`: bounded staleness, TTL, queue, and in-flight limits.

### `run`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `run.intent` | `train`, `dry_run`, `validate` | `train` builds the launch plan and may run training. `dry_run` returns a plan. `validate` only validates config/artifacts. |
| `run.emit_resolved_config` | `false` | Write `.nano_rl/resolved-config.json` and exit early when true. The CLI flag `--emit-resolved-config` does the same. |
| `run.start_ray_actors` | `false` | When false, `train` emits a backend-integrated plan only. When true, Ray actor graph startup and the training loop are attempted. |

### `mock`

`mock` is optional. It does not force every subsystem to be mock; the concrete
backend fields still decide that.

| Field | Options / default | Meaning |
| --- | --- | --- |
| `mock.enabled` | `false` | Marks a mock-oriented config; used for mock memory hints and strictness expectations. |
| `mock.seed` | `0` | Global deterministic seed for mock paths unless a lower-level seed overrides it. |
| `mock.ray_actor_memory_mb` | `null`, integer `>=1` | Optional Ray actor heap-memory scheduling hint in MiB for mock actor runs. |
| `mock.strict.leases` | `true` | Mock backends should enforce GPU lease checks. |
| `mock.strict.versions` | `true` | Mock backends should enforce policy/weight version checks. |
| `mock.strict.no_gpu_imports` | `true` | Mock-only paths should not import GPU-heavy packages. |
| `mock.strict.no_external_data_probe` | `true` | Mock data sources should not probe HDFS/external storage. |
| `mock.timing.rollout_sleep_ms` | `0` | Global mock rollout latency hint. |
| `mock.timing.trainer_sleep_ms` | `0` | Global mock trainer latency hint. |
| `mock.timing.weight_io_sleep_ms` | `0` | Global mock weight I/O latency hint. |
| `mock.failure_injection.enabled` | `false` | Enables configured mock failures. |
| `mock.failure_injection.fail_after_rollout_requests` | `null`, integer `>=1` | Fail rollout after N requests. |
| `mock.failure_injection.fail_on_train_step` | `null`, integer `>=1` | Fail trainer on a specific step. |
| `mock.failure_injection.fail_weight_version` | `null`, integer `>=0` | Fail activation/export for a weight version. |
| `mock.failure_injection.corrupt_weight_checksum` | `false` | Generate intentionally bad mock weight checksums. |
| `mock.failure_injection.stale_lease_epoch_delta` | `0` | Simulate stale lease epochs. |

### `runtime`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `runtime.backend` | `ray` | v0.1 execution layer is Ray-native only. |
| `runtime.local.num_gpus` | integer `>=1` | Physical local GPU count. Must be at least `rollout_only_gpus + shared_gpus + idle_gpus`. |
| `runtime.local.cpus` | integer `>=1` | CPU count available for local Ray actor planning. |
| `runtime.storage.allowed_input_sources` | any of `hdfs_uri`, `hdfs_fuse_path`, `local_csv`, `mock_inline`, `mock_generated`, `mock_jsonl` | Whitelist for `data.source_type`; the selected source must appear here. |
| `runtime.storage.hdfs.cli` | string | HDFS CLI binary, usually `hdfs`, required for `hdfs_uri` artifact probes. |
| `runtime.storage.hdfs.read_probe_timeout_sec` | `30` | Timeout for HDFS existence/read probes. |
| `runtime.storage.hdfs_fuse.mount_root` | absolute path | Root path for HDFS FUSE data. Required when using `hdfs_fuse_path`. |
| `runtime.storage.hdfs_fuse.require_mount` | `true` | Whether the mount is expected to exist as a real mount. |
| `runtime.storage.hdfs_fuse.read_probe_timeout_sec` | `10` | Timeout for local FUSE read probes. |
| `runtime.ray.address` | `auto` or Ray address string | `auto` first connects to an existing Ray cluster, then creates a local single-machine cluster if none is reachable. Explicit addresses fail fast. |
| `runtime.ray.namespace` | string | Ray namespace used by actors. |
| `runtime.ray.dedup_logs` | `false` | When false, disables Ray repeated-log deduplication before Ray import/init. |
| `runtime.ray.gpu_manager.enabled` | must be `true` | GPU lease manager is required in v0.1. |
| `runtime.ray.gpu_manager.lease_manager` | must be `true` | Lease-token CUDA gate is required in v0.1. |
| `runtime.ray.gpu_manager.role_classes.rollout_manager` | dotted import path | Actor/role class path; current examples use `nano_rl.runtime.roles.RolloutManagerRole`. |
| `runtime.ray.gpu_manager.role_classes.rollout_replica_controller` | dotted import path | Rollout replica controller role path. |
| `runtime.ray.gpu_manager.role_classes.rollout_worker` | dotted import path | Rollout worker role path. |
| `runtime.ray.gpu_manager.role_classes.trainer_rank` | dotted import path or omitted | Required whenever `shared_gpus > 0`. |
| `runtime.ray.gpu_manager.topology.rollout_only_gpus` | integer `>=0` | GPUs that only run rollout. Must be divisible by rollout TP size. |
| `runtime.ray.gpu_manager.topology.shared_gpus` | integer `>=0` | GPUs that toggle between rollout and trainer. Must equal `placement.trainer.num_ranks` and be divisible by rollout TP size. |
| `runtime.ray.gpu_manager.topology.idle_gpus` | `0`, integer `>=0` | GPUs intentionally left unused. |
| `runtime.ray.gpu_manager.hybrid_toggle.strategy` | `swap_on_toggle`, `dual_resident` | v0.1 examples use `swap_on_toggle`; `dual_resident` is reserved for a stricter future path. |
| `runtime.ray.gpu_manager.hybrid_toggle.rollout_drain_timeout_sec` | integer `>=1` | Time budget to drain rollout before train window. |
| `runtime.ray.gpu_manager.hybrid_toggle.train_drain_timeout_sec` | integer `>=1` | Time budget to drain train before returning to rollout. |
| `runtime.ray.gpu_manager.hybrid_toggle.cuda_quiesce_timeout_sec` | integer `>=1` | Time budget for CUDA work to quiesce before lease switch. |
| `runtime.ray.gpu_manager.hybrid_toggle.offload.trainer_model` | `cpu_pinned`, `cpu` | Trainer model residency after offload. |
| `runtime.ray.gpu_manager.hybrid_toggle.offload.trainer_optimizer` | `cpu_pinned`, `cpu` | Trainer optimizer residency after offload. |
| `runtime.ray.gpu_manager.hybrid_toggle.offload.rollout_engine` | `vllm_sleep`, `teardown_and_reload` | Rollout offload strategy. Use `teardown_and_reload` for mock/HuggingFace examples; `vllm_sleep` requires `vllm_sleep_level`. |
| `runtime.ray.gpu_manager.hybrid_toggle.offload.vllm_sleep_level` | `1`, `2`, required with `vllm_sleep` | vLLM sleep/offload level. |
| `runtime.ray.gpu_manager.hybrid_toggle.offload.cpu_memory_budget_gb` | number `>=1` | CPU memory budget for offloaded state. |
| `runtime.ray.gpu_manager.hybrid_toggle.offload.residual_gpu_memory_budget_mb` | integer `>=0` | Allowed residual GPU memory after offload. |
| `runtime.ray.placement.trainer.num_ranks` | integer `>=1` | Must equal `shared_gpus`. |
| `runtime.ray.placement.trainer.gpus_per_rank` | must be `1` in v0.1 | Logical train GPU resource per rank. Actors still use Ray `num_gpus=0`. |
| `runtime.ray.placement.trainer.cpus_per_rank` | `0`, number `>=0` | CPU resource hint per trainer rank. |
| `runtime.ray.placement.rollout.num_replicas` | integer `>=1` | Must equal `parallel.rollout.data_parallel_size`. |
| `runtime.ray.placement.rollout.gpus_per_replica` | integer `>=1` | Must equal rollout TP size. |
| `runtime.ray.placement.rollout.tensor_parallel_size` | integer `>=1` | Must equal `parallel.rollout.tensor_parallel_size`. |
| `runtime.ray.placement.reward.num_actors` | optional integer `>=1` | Optional reward actor count. |
| `runtime.ray.placement.reward.cpus_per_actor` | optional number `>=0` | CPU resource hint per reward actor. |

### `mode`

| Field | Options | Meaning |
| --- | --- | --- |
| `mode` | `collocated`, `disaggregated` | User-facing topology. `collocated` normalizes to `fully_sync`; `disaggregated` normalizes to `standalone_hybrid`. Do not use internal names in YAML. |

Mode-specific constraints:

- `collocated`: `rollout_only_gpus` must be `0`, `shared_gpus >= 1`, `control.max_policy_lag == 0`, and `rollout.hybrid.policy_pin.lagged_ratio == 0`.
- `disaggregated`: `rollout_only_gpus > 0`, `shared_gpus > 0`, and `control.max_policy_lag >= 1`.

### `parallel`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `parallel.trainer.data_parallel_size` | integer `>=1` | Trainer data parallel size. |
| `parallel.trainer.fsdp_world_size` | integer `>=1` | Must equal `runtime.ray.placement.trainer.num_ranks`. |
| `parallel.trainer.tensor_parallel_size` | `1` in v0.1 | Trainer TP is not supported yet. |
| `parallel.trainer.pipeline_parallel_size` | `1` in v0.1 | Trainer PP is not supported yet. |
| `parallel.rollout.data_parallel_size` | integer `>=1` | Must equal rollout replica count. |
| `parallel.rollout.tensor_parallel_size` | integer `>=1` | Must equal rollout placement TP and GPU count per replica. |

### `model`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `model.model_path` | non-empty string | Model artifact path or model id/path consumed by trainer and rollout backends. |
| `model.tokenizer_path` | defaults to `model.model_path` | Tokenizer path. It is normalized to `model_path` when omitted. |

Model paths are normal artifact paths, not `data.source_type` inputs. Their
existence is checked by the selected backend or optional artifact validation,
not by the storage source whitelist.

### `data`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `data.source_type` | `hdfs_uri`, `hdfs_fuse_path`, `local_csv`, `mock_inline`, `mock_generated`, `mock_jsonl` | Prompt source kind. Must be listed in `runtime.storage.allowed_input_sources`. |
| `data.data_path` | required for `hdfs_uri`, `hdfs_fuse_path`, `local_csv`, `mock_jsonl` | Path/URI to prompt data. `hdfs_uri` must start with `hdfs://`; `hdfs_fuse_path` must be absolute. |
| `data.prompt_column` | `prompt` | Column/key containing prompt text. |
| `data.local_csv.encoding` | `utf-8` | CSV encoding for `local_csv`. |
| `data.mock_inline.prompts` | list of strings, required for `mock_inline` | Inline prompts embedded in YAML. |
| `data.mock_inline.repeat` | `1`, integer `>=1` | Repeat inline prompts N times. |
| `data.mock_generated.count` | integer `>=1`, required for `mock_generated` | Number of deterministic prompts to generate. |
| `data.mock_generated.template` | `prompt-{index}` | Python format string for generated prompts. |
| `data.mock_generated.start_index` | `0` | First index used by the generated source. |
| `data.mock_generated.shuffle` | `false`, `true`, `deterministic` | Whether generated prompt order is shuffled. |
| `data.mock_jsonl.encoding` | `utf-8` | JSONL encoding for `mock_jsonl`. |
| `data.mock_profile.enabled` | `false` | Enables mock prompt length shaping metadata. |
| `data.mock_profile.sleep_enabled` | `false` | Enables mock loader sleep. |
| `data.mock_profile.prompt_length_distribution` | `fixed`, `uniform`, `lognormal`, `chat_mixture` | Synthetic prompt length distribution. |
| `data.mock_profile.min_prompt_tokens` | `1` | Minimum synthetic prompt token count. |
| `data.mock_profile.mean_prompt_tokens` | `64` | Mean synthetic prompt token count; must be `<= max_prompt_tokens`. |
| `data.mock_profile.max_prompt_tokens` | `512` | Maximum synthetic prompt token count. |
| `data.mock_profile.length_jitter` | `0.65` | Jitter for sampled prompt length. |
| `data.mock_profile.pad_prompts` | `false` | Pad prompt text to the sampled token length. |
| `data.mock_profile.load_base_ms` | `0` | Base mock data load latency. |
| `data.mock_profile.load_ms_per_1k_tokens` | `0` | Token-scaled mock load latency. |
| `data.mock_profile.load_jitter_ms` | `0` | Random mock load latency jitter. |
| `data.mock_profile.max_load_ms` | `0` | Maximum mock load sleep. `0` disables sleep cap logic. |

### `algorithm`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `algorithm.name` | `ppo` | Only PPO-like training is represented in v0.1. |
| `algorithm.max_steps` | integer `>=1` | Driver-side maximum training steps. |
| `algorithm.learning_rate` | number `>0` | Passed into trainer backend config. |
| `algorithm.seed` | `0` | Algorithm seed. |

### `trainer`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `trainer.backend` | `fsdp2`, `mock`, `fake` | `fake` is accepted as a deprecated alias and normalizes to `mock`. |
| `trainer.global_batch_size` | integer `>=1` | Global train batch size. |
| `trainer.checkpoint_dir` | optional string | Required for real FSDP2 weight export. Rank 0 writes versioned checkpoints here. |
| `trainer.fsdp2.mixed_precision` | `bf16`, `fp16` | Required when `trainer.backend: fsdp2`. |
| `trainer.fsdp2.sharding` | `full_shard`, `hybrid_shard` | Required when `trainer.backend: fsdp2`. |
| `trainer.fsdp2.rendezvous` | `null` or init method such as `env://` | Optional torch distributed init method. |
| `trainer.fsdp2.store_endpoint` | `null` or `host:port` | Used to build `tcp://host:port` rendezvous for single-machine multi-rank FSDP2. |
| `trainer.fsdp2.dist_backend` | `nccl` | Torch distributed backend. |
| `trainer.fsdp2.trust_remote_code` | `false` | Passed to Hugging Face model/tokenizer loaders inside FSDP2 backend. |
| `trainer.fsdp2.max_length` | `null`, integer `>=1` | Optional tokenizer truncation length for LM training batches. |
| `trainer.mock.initial_loss` | `1.0`, number `>0` | Starting mock loss. |
| `trainer.mock.loss_decay` | `reciprocal`, `constant` | Mock loss schedule. |
| `trainer.mock.export_format` | `vllm_compatible`, `hf`, `safetensors` | Mock exported weight format. |
| `trainer.mock.optimizer_state` | `tracked`, `stateless` | Mock optimizer state behavior. |
| `trainer.mock.require_rank0_export` | `true` | Only rank 0 may publish mock weights when true. |
| `trainer.mock.seed` | `0` | Mock trainer seed. |
| `trainer.mock.num_parameters` | `1024` | Metadata-only mock parameter count. |

### `rollout`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `rollout.backend` | `vllm`, `huggingface`, `mock` | Selects rollout adapter. All adapters enforce rollout lease and policy version checks. |
| `rollout.vllm.dtype` | `null` or vLLM dtype string | Passed to `vllm.LLM`. |
| `rollout.vllm.max_model_len` | `null`, integer `>=1` | vLLM max model length. |
| `rollout.vllm.trust_remote_code` | `false` | Passed to vLLM loader. |
| `rollout.vllm.engine_kwargs` | `{}` | Extra `vllm.LLM` constructor kwargs. |
| `rollout.vllm.sampling_params` | `{}` | kwargs used to build `vllm.SamplingParams`, such as `temperature`, `top_p`, `max_tokens`. |
| `rollout.vllm.weight_sync_backend` | `auto`, `ipc`, `nccl`, `none` | Native vLLM Weight Transfer backend used to sync post-step trainer parameters into the rollout engine. `auto` chooses `ipc` for shared-GPU replicas and `nccl` for rollout-only/separate-GPU replicas. |
| `rollout.vllm.require_weight_sync` | `true` | Fail fast when native vLLM Weight Transfer support is unavailable instead of silently serving stale weights. |
| `rollout.huggingface.dtype` | `null`, `auto`, `bfloat16`, `float16`, `float32`, etc. | Torch dtype for direct Transformers model loading. |
| `rollout.huggingface.device` | `auto` | Device string. `auto` chooses `cuda:0` when CUDA is available, otherwise CPU. |
| `rollout.huggingface.device_map` | `null`, string, or mapping | Optional Transformers `device_map`. |
| `rollout.huggingface.trust_remote_code` | `false` | Passed to `AutoModelForCausalLM` and `AutoTokenizer`. |
| `rollout.huggingface.model_kwargs` | `{}` | Extra model `from_pretrained` kwargs. |
| `rollout.huggingface.tokenizer_kwargs` | `{}` | Extra tokenizer `from_pretrained` kwargs. |
| `rollout.huggingface.generation_kwargs` | `{}` | Extra `model.generate` kwargs, such as `do_sample`, `temperature`, `top_p`, `max_new_tokens`. |
| `rollout.huggingface.skip_special_tokens` | `true` | Passed to tokenizer decode. |
| `rollout.mock.response_template` | `{prompt} :: response@v{policy_version}` | Python format string for deterministic mock responses. |
| `rollout.mock.tokenization` | `sha256_bytes`, `whitespace_hash` | Mock tokenization mode; current backend primarily uses deterministic ids. |
| `rollout.mock.max_response_tokens` | `16`, integer `>=1` | Maximum mock response length. |
| `rollout.mock.min_response_tokens` | `1`, integer `>=1` | Minimum mock response length. |
| `rollout.mock.mean_response_tokens` | `16`, integer `>=1` | Mean response length; must be `<= max_response_tokens`. |
| `rollout.mock.response_length_distribution` | `fixed`, `uniform`, `lognormal`, `chat_mixture` | Mock response length distribution. |
| `rollout.mock.response_length_jitter` | `0.65` | Jitter for mock response length. |
| `rollout.mock.max_sequence_tokens` | `4096`, integer `>=2` | Prompt + response cap; must be greater than `max_response_tokens`. |
| `rollout.mock.prefill_base_ms` | `0` | Base mock prefill latency. |
| `rollout.mock.prefill_ms_per_1k_tokens` | `0` | Prompt-token-scaled prefill latency. |
| `rollout.mock.decode_base_ms` | `0` | Base mock decode latency. |
| `rollout.mock.decode_ms_per_token` | `0` | Response-token-scaled decode latency. |
| `rollout.mock.latency_jitter_ms` | `0` | Mock latency jitter. |
| `rollout.mock.max_sample_sleep_ms` | `20000` | Maximum mock sample sleep. Set `0` to disable sleeping. |
| `rollout.mock.logprob_mode` | `linear`, `constant` | Mock logprob generation mode. |
| `rollout.mock.finish_reason` | `stop` | Finish reason in mock outputs. |
| `rollout.mock.include_policy_segments` | `false` | Include token-level policy segment metadata. Useful for disaggregated/stale sample tests. |
| `rollout.mock.seed` | `0` | Mock rollout seed. |
| `rollout.partial_rollout.enabled` | `false` | Allows partial rollout behavior in disaggregated mode. |
| `rollout.partial_rollout.pause_mode` | `keep` | Only `keep` is accepted. |
| `rollout.partial_rollout.clear_cache` | must be `true` | Required when partial rollout is configured. |
| `rollout.partial_rollout.mixed_policy_samples` | `train`, `drop` | Whether mixed-policy samples can train or should be dropped. |
| `rollout.hybrid.enabled` | `true` | Enables hybrid rollout policy knobs. |
| `rollout.hybrid.precision_mix.bf16` | `1.0`, `0..1` | Target BF16 ratio metadata. |
| `rollout.hybrid.precision_mix.fp8` | `0.0`, `0..1` | Target FP8 ratio metadata. |
| `rollout.hybrid.policy_pin.latest_ratio` | `1.0`, `0..1` | Fraction pinned to latest policy. |
| `rollout.hybrid.policy_pin.lagged_ratio` | `0.0`, `0..1` | Fraction allowed to lag. Must be `0` in `collocated`. |

### `reward`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `reward.backend` | `mock` | Only deterministic mock reward is implemented today. |
| `reward.mock.name` | `deterministic_length_reward` | Reward source name stored on samples. |
| `reward.mock.scale` | `100.0`, number `>0` | Response length scaling denominator. |
| `reward.mock.cap` | `1.0`, number `>0` | Maximum reward. |

### `weight_transfer`

| Field | Options / default | Meaning |
| --- | --- | --- |
| `weight_transfer.method` | `locality_aware_checkpoint`, `objectref` | `locality_aware_checkpoint` keeps large tensors out of Ray object store; `objectref` is for small/debug weights. |
| `weight_transfer.allow_rollout_only_artifact_pull` | `true` | Required for `locality_aware_checkpoint` when `rollout_only_gpus > 0`. |
| `weight_transfer.max_versions_in_flight` | `2`, integer `>=1` | Maximum weight versions being transferred/activated. |
| `weight_transfer.store.backend` | `checkpoint`, `mock_memory`, `mock_filesystem` | Weight payload storage backend. `mock_memory` cannot be used with `run.start_ray_actors: true`. |
| `weight_transfer.store.manifest_dir` | `null` or path | Required when `store.backend: mock_filesystem`. |
| `weight_transfer.store.checksum_mode` | `semantic_sha256` | Current checksum mode. |

### `control`

| Field | Options | Meaning |
| --- | --- | --- |
| `control.max_policy_lag` | integer `>=0` | Maximum allowed sample policy lag. Must be `0` in `collocated`; must be `>=1` in `disaggregated`. |
| `control.sample_ttl_sec` | integer `>=1` | Sample time-to-live. |
| `control.queue_high_watermark` | integer `>=1` | Backpressure threshold for sample queue depth. |
| `control.max_pending_rollout_refs` | integer `>=1` | In-flight rollout request/reference cap. |
| `control.max_pending_train_refs` | integer `>=1` | In-flight trainer reservation/reference cap. |

### Cross-field checklist

- `runtime.local.num_gpus >= rollout_only_gpus + shared_gpus + idle_gpus`.
- `placement.trainer.num_ranks == shared_gpus`.
- `placement.trainer.gpus_per_rank == 1`.
- `placement.rollout.num_replicas == parallel.rollout.data_parallel_size`.
- `placement.rollout.gpus_per_replica == parallel.rollout.tensor_parallel_size`.
- `placement.rollout.num_replicas * placement.rollout.gpus_per_replica == rollout_only_gpus + shared_gpus`.
- `parallel.trainer.fsdp_world_size == placement.trainer.num_ranks`.
- `parallel.trainer.tensor_parallel_size == 1` and `parallel.trainer.pipeline_parallel_size == 1`.
- `rollout_only_gpus` and `shared_gpus` must each be divisible by rollout tensor parallel size.
- `data.source_type` must appear in `runtime.storage.allowed_input_sources`.
- `trainer.backend: fsdp2` requires `trainer.fsdp2`.
- `weight_transfer.store.backend: mock_filesystem` requires `manifest_dir`.

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
- `recipes/collocated_qwen3.yaml`: local HuggingFace-rollout + FSDP2 recipe wired to `huggingface/models/qwen3-0.6b-hf` and `huggingface/data/awesome-chatgpt-prompts/prompts.csv`.

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
