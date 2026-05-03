# Health Events (Draft)

- `ROLLOUT_WORKER_STALE_WEIGHT`
- `ROLLOUT_QUEUE_BACKPRESSURE`
- `ROLLOUT_GENERATION_TIMEOUT`
- `SAMPLE_DROPPED_POLICY_LAG`
- `SAMPLE_DROPPED_TTL`
- `TRAINER_OOM`
- `TRAINER_GROUP_FAILED`
- `WEIGHT_EXPORT_FAILED`
- `WEIGHT_DISTRIBUTION_FAILED`
- `SHARED_GPU_TOGGLE_TIMEOUT`
- `SHARED_GPU_OFFLOAD_FAILED`
- `SHARED_GPU_HYDRATE_FAILED`
- `TRAINER_COMM_GROUP_UNHEALTHY`
- `TRAINER_COMM_GROUP_REBUILT`
- `OBJECT_STORE_PRESSURE`

Minimum event payload:

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
