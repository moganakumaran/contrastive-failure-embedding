# Synthetic Dataset Schema and Generation Design

## Overview

The synthetic dataset contains 60,000 failure episode records distributed uniformly across six failure classes (10,000 records per class). Each record is a 28-dimensional operational signal vector derived from three telemetry categories available in Delta Lake and Apache Iceberg Lakehouse platforms: resource utilization metrics, shuffle and spill statistics, and structured error diagnostics.

Records are split 70 / 15 / 15 into train, validation, and test sets (42,000 / 9,000 / 9,000) using stratified sampling. Hard negative pairs constitute 20% of the test set (1,800 records): pairs sharing an identical error type code but belonging to different failure classes.

All feature values in the CSV files are percentile-relative scores in [0, 1], computed as if derived from a rolling window of historical executions of the same job. This is consistent with the normalization scheme described in Section 2.2 of the manuscript. In a production deployment, these scores would be computed from actual job histories; here they are generated directly to reflect the within-class and between-class distributional structure of each failure type.

---

## Feature Schema

### Group A: Resource Utilization (features 0–7)

| Index | Name | Unit (raw) | Description |
|---|---|---|---|
| 0 | `peak_executor_memory` | GB | Peak heap memory across all executors during job |
| 1 | `mean_executor_memory` | GB | Mean executor memory over job duration |
| 2 | `peak_cpu_utilization` | % | Peak CPU utilization relative to allocated vCPUs |
| 3 | `mean_cpu_utilization` | % | Mean CPU utilization over job duration |
| 4 | `total_input_bytes` | GB | Total bytes read from storage |
| 5 | `total_output_bytes` | GB | Total bytes written to storage |
| 6 | `total_records_read` | millions | Total records read across all stages |
| 7 | `total_records_written` | millions | Total records written to output |

### Group B: Shuffle and Spill Statistics (features 8–19)

| Index | Name | Unit (raw) | Description |
|---|---|---|---|
| 8 | `shuffle_read_bytes` | GB | Total shuffle read volume |
| 9 | `shuffle_write_bytes` | GB | Total shuffle write volume |
| 10 | `spill_to_disk_bytes` | GB | Total bytes spilled to disk during sort/shuffle |
| 11 | `total_partitions` | count | Total number of shuffle partitions |
| 12 | `max_partition_bytes` | MB | Largest shuffle partition size |
| 13 | `min_partition_bytes` | MB | Smallest shuffle partition size |
| 14 | `partition_size_std` | MB | Standard deviation of partition byte sizes (data skew proxy) |
| 15 | `shuffle_fetch_wait_ms` | ms | Mean shuffle fetch wait time (latency proxy) |
| 16 | `sort_buffer_utilization` | ratio | Fraction of sort buffer used at peak (0–1) |
| 17 | `executor_disk_io` | GB | Total executor disk I/O |
| 18 | `network_bytes_transmitted` | GB | Total cross-node bytes transmitted |
| 19 | `task_duration_variance` | s² | Variance of individual task completion times (skew proxy) |

### Group C: Structured Error Diagnostics (features 20–27)

| Index | Name | Type | Description |
|---|---|---|---|
| 20 | `error_type_code` | categorical (0–5) | Error category: 0=skew/partition, 1=OOM, 2=shuffle/executor, 3=spot-preemption, 4=parse/cast, 5=throttle |
| 21 | `affected_subsystem_code` | categorical (0–4) | Subsystem: 0=memory, 1=shuffle, 2=disk, 3=scheduler, 4=parser |
| 22 | `failure_timing_fraction` | ratio (0–1) | Fraction of total job duration elapsed at failure |
| 23 | `retry_attempt_count` | count | Number of task or job retry attempts at failure |
| 24 | `executor_count_at_failure` | count | Active executor count when failure was recorded |
| 25 | `stage_id_at_failure` | count | Stage number (zero-indexed) where failure occurred |
| 26 | `failed_task_count` | count | Number of tasks that failed in the failing stage |
| 27 | `error_severity_code` | categorical (0–3) | Severity: 0=info, 1=warning, 2=error, 3=critical |

---

## Per-Class Distribution Design

Each class is modelled as a multivariate Gaussian in the percentile-relative feature space, with means and standard deviations chosen to reflect the operational signature of each failure type. Correlations between related features (e.g., spill_to_disk and sort_buffer_utilization within Shuffle Spill) are introduced via a class-specific correlation structure.

Feature values are clipped to [0, 1] after sampling. Categorical features (error_type_code, affected_subsystem_code, error_severity_code) are sampled from discrete distributions centered on the class-typical code.

### Class 0: Data Skew

| Feature group | Key signals |
|---|---|
| Resource utilization | Moderate memory and CPU; elevated input bytes |
| Shuffle/spill | Very high partition_size_std and task_duration_variance; high shuffle_read_bytes; low spill |
| Error diagnostics | error_type_code = 0 (skew/partition); timing fraction = 0.65–0.80 (fails late); moderate retry count |

**Discriminating features:** `partition_size_std` and `task_duration_variance` are significantly elevated. `spill_to_disk_bytes` and `sort_buffer_utilization` are low, distinguishing Data Skew from Shuffle Spill.

### Class 1: Memory Saturation

| Feature group | Key signals |
|---|---|
| Resource utilization | Very high peak and mean executor memory (near allocation ceiling); low CPU |
| Shuffle/spill | Low spill; low task_duration_variance (uniform until OOM) |
| Error diagnostics | error_type_code = 1 (OOM); affected_subsystem = 0 (memory); failure_timing_fraction = 0.75–0.90 (fails late); error_severity = 3 (critical) |

**Discriminating features:** `peak_executor_memory` and `mean_executor_memory` are the highest of all classes. `spill_to_disk_bytes` is low, distinguishing Memory Saturation from Shuffle Spill despite both potentially surfacing as executor failures.

**Hard negative pairing:** Memory Saturation records may share `error_type_code = 2` (executor lost) with Shuffle Spill records in the hard negative subset, testing whether the model relies on error code alone.

### Class 2: Shuffle Spill

| Feature group | Key signals |
|---|---|
| Resource utilization | Moderate memory (normal consumption); elevated disk I/O |
| Shuffle/spill | Very high spill_to_disk_bytes; very high sort_buffer_utilization; high shuffle_read and write bytes |
| Error diagnostics | error_type_code = 2 (shuffle/executor); affected_subsystem = 2 (disk); failure_timing_fraction = 0.55–0.75 |

**Discriminating features:** `spill_to_disk_bytes` and `sort_buffer_utilization` are the highest of all classes. `peak_executor_memory` is normal, distinguishing Shuffle Spill from Memory Saturation.

### Class 3: Spot Instance Interruption

| Feature group | Key signals |
|---|---|
| Resource utilization | Low to moderate across all metrics |
| Shuffle/spill | Low shuffle activity and spill; moderate task_duration_variance |
| Error diagnostics | error_type_code = 3 (preemption); failure_timing_fraction = 0.30–0.60 (variable, can occur any time); very high retry_attempt_count; very low executor_count_at_failure |

**Discriminating features:** `retry_attempt_count` is highest of all classes. `executor_count_at_failure` is very low (abrupt executor loss). Spot interruptions have a distinctive combination of low resource pressure with high retry activity.

### Class 4: Schema Drift

| Feature group | Key signals |
|---|---|
| Resource utilization | Very low across all metrics (fails before significant computation) |
| Shuffle/spill | Near-zero shuffle and spill (fails in parse stage) |
| Error diagnostics | error_type_code = 4 (parse/cast); affected_subsystem = 4 (parser); failure_timing_fraction = 0.05–0.20 (VERY EARLY); very low retry count; very low stage_id_at_failure |

**Discriminating features:** `failure_timing_fraction` and `stage_id_at_failure` are lowest of all classes. Near-zero resource utilization across all feature groups. These are the strongest indicators of schema-related early failure.

### Class 5: API Rate Limiting

| Feature group | Key signals |
|---|---|
| Resource utilization | Low to moderate; job does not consume resources heavily before failure |
| Shuffle/spill | Very high shuffle_fetch_wait_ms (latency spikes from retries); low spill |
| Error diagnostics | error_type_code = 5 (throttling); failure_timing_fraction = variable; very high retry_attempt_count; error_severity = 1 (warning/transient) |

**Discriminating features:** `shuffle_fetch_wait_ms` is the highest of all classes. `retry_attempt_count` is very high (alongside Spot Interruption, but distinguished by low executor loss and high fetch wait). Error severity is lower (throttling is transient rather than catastrophic).

---

## Hard Negative Construction

Hard negative pairs are constructed as follows:

1. For each hard-negative record in the test set, a second record is selected from a different class that shares the same `error_type_code` value.
2. The two classes most prone to producing hard negatives are:
   - **Memory Saturation (class 1) and Shuffle Spill (class 2):** Both can surface as `error_type_code = 2` (executor lost or shuffle failure) depending on the specific error propagation path. In the hard negative subset, a fraction of Memory Saturation and Shuffle Spill test records are assigned `error_type_code = 2`.
   - **Data Skew (class 0) and Shuffle Spill (class 2):** Both involve shuffle-related errors and can share `error_type_code = 0` or `2`.
3. All 28 features other than the shared error code retain their class-typical distributions, providing the signal needed for the embedding to discriminate correctly.
4. Hard negatives constitute exactly 20% of the test set (1,800 of 9,000 records).

---

## Calibration to Published Cloud Workload Traces

The per-class distribution parameters for resource utilization and shuffle metrics are calibrated against statistical summaries from:

- Reiss et al. (2012): Google cluster traces. Used to set realistic ranges for CPU utilization, memory consumption, and task duration variance.
- Di et al. (2012): Cloud vs. grid workload characterization. Used to set realistic shuffle volume and partition count distributions.

The calibration maps the coefficient of variation (CV) and skewness reported in those traces to the standard deviation and shape parameters of the per-class Gaussian distributions used here.

---

## CSV Column Order

The generated CSV files contain the following columns in order:

```
peak_executor_memory, mean_executor_memory, peak_cpu_utilization, mean_cpu_utilization,
total_input_bytes, total_output_bytes, total_records_read, total_records_written,
shuffle_read_bytes, shuffle_write_bytes, spill_to_disk_bytes, total_partitions,
max_partition_bytes, min_partition_bytes, partition_size_std, shuffle_fetch_wait_ms,
sort_buffer_utilization, executor_disk_io, network_bytes_transmitted, task_duration_variance,
error_type_code, affected_subsystem_code, failure_timing_fraction, retry_attempt_count,
executor_count_at_failure, stage_id_at_failure, failed_task_count, error_severity_code,
label, is_hard_negative
```

`label` is the integer class index (0–5). `is_hard_negative` is 1 for records participating in a hard negative pair and 0 otherwise (only set in the test split).
