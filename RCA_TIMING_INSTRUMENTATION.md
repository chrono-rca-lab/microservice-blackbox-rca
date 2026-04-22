# RCA Pipeline Timing Instrumentation

## Overview

End-to-end timing instrumentation has been added to the RCA pipeline to track execution time across all major analysis stages. Timing data flows from fault injection through the entire FChain analysis pipeline.

## Architecture

### Entry Points
- **`fault_chain.pinpoint()`** accepts optional `start_time` and `logs` parameters
- If not provided, `start_time` is initialized to current time and `logs` is initialized to empty list
- Both parameters flow through the entire call chain

### Timing Collection
- Each major stage calls `log_stage()` to record timing information
- `log_stage()` is defined in `rca_engine/logger.py`
- Each log entry captures:
  - Stage name (e.g., "LAYER1_CUSUM")
  - Source file
  - Timestamp when stage was logged
  - `since_start_seconds`: time elapsed since pipeline start
  - `duration_seconds`: time spent in this stage

### Output
- Timing data is collected in the `logs` list passed through the pipeline
- Experiment orchestrators (`run_experiment_slo.py`, `run_experiment.py`) capture:
  - `total_time_seconds`: total RCA execution time
  - `timing_logs`: array of stage timing entries
- Output saved to `rca_timing.json` in the experiment directory

## Instrumented Stages

### Core Stages (Always Logged)
1. **START_PINPOINT** - RCA pipeline entry point
2. **LAYER1_CUSUM** - Change-point detection via CUSUM + Bootstrap (change_point.py)
3. **LAYER3_FFT_FILTER** - FFT burst threshold filtering (predictability_filter.py)
4. **LAYER4_ROLLBACK** - Tangent-based onset refinement (tangent_rollback.py)
5. **FINAL_RANKING** - Root cause ranking and propagation filtering

### Implicit Coverage (Part of Above Stages)
- **LAYER2_PREDICTION_ERROR** - Markov model prediction error (within Layer 1 pipeline)
- **LAYER5_AGGREGATION** - Per-metric aggregation (within Layer 1 pipeline)
- **LAYERS 6-8** - Propagation analysis and root cause identification (within FINAL_RANKING)

## Usage

### Basic Usage (No Timing)
```python
from rca_engine import fault_chain
ranked = fault_chain.pinpoint(
    metric_matrix=data,
    baseline_window=(bl_start, bl_end),
    fault_window=(ft_start, ft_end),
)
```

### With Timing Instrumentation
```python
import time
from rca_engine import fault_chain

start_time = time.time()
logs = []

ranked = fault_chain.pinpoint(
    metric_matrix=data,
    baseline_window=(bl_start, bl_end),
    fault_window=(ft_start, ft_end),
    start_time=start_time,
    logs=logs,
)

total_time = time.time() - start_time
print(f"Total RCA time: {total_time:.2f}s")
for log in logs:
    print(f"  {log['stage']}: {log['duration_seconds']:.3f}s")
```

### In Experiment Orchestrators
The updated experiment files automatically:
1. Initialize `start_time` and `logs` at RCA entry
2. Pass both through to `fault_chain.pinpoint()`
3. Collect output with timing data
4. Save to `{experiment_dir}/rca_timing.json`

## Example Output

```json
{
  "ranked_services": [...],
  "total_time_seconds": 2.456,
  "timing_logs": [
    {
      "stage": "START_PINPOINT",
      "file": "fault_chain.py",
      "timestamp": 1234567890.123,
      "since_start_seconds": 0.001,
      "duration_seconds": 0.0
    },
    {
      "stage": "LAYER1_CUSUM",
      "file": "change_point.py",
      "timestamp": 1234567890.234,
      "since_start_seconds": 0.111,
      "duration_seconds": 0.110
    },
    ...
  ]
}
```

## Implementation Details

### Modified Files
- `rca_engine/logger.py` - NEW: Centralized timing utility
- `rca_engine/fault_chain.py` - Added timing parameters to `pinpoint()`, internal layer calls
- `rca_engine/change_point.py` - Added timing to `run_layer1()`
- `rca_engine/predictability_filter.py` - Added timing to `filter_abnormal_change_points()`
- `rca_engine/tangent_rollback.py` - Added timing to `rollback_onset()`
- `eval/run_experiment_slo.py` - Integrated timing collection and output
- `eval/run_experiment.py` - Integrated timing collection and output

### Design Decisions
1. **Optional Parameters**: `start_time` and `logs` are optional with sensible defaults
   - Backward compatible with existing code
   - No breaking changes to API

2. **Minimal Overhead**: Only major pipeline stages are instrumented
   - Avoid logging on every metric analysis (would add excessive noise)
   - Focus on high-level stage boundaries

3. **Centralized Logger**: `log_stage()` in dedicated module
   - Single source of truth for timing collection
   - Easy to modify output format or storage mechanism

4. **Shared Logs List**: Same list passed through call chain
   - No global state
   - Thread-safe collection (mutations happen serially)
   - Easy to test and debug

## Performance Impact

- **Negligible overhead**: Only `time.time()` calls and dictionary appends
- **Memory footprint**: ~5-10 dictionary objects per RCA run
- **No algorithm changes**: Pure observability addition

## Future Enhancements

- Add timing to `smoothing.py` if bottleneck analysis shows it's significant
- Add conditional timing for specific layers based on debug mode
- Export timing data in alternative formats (CSV, Prometheus metrics)
- Add timing assertions/SLAs for each stage
