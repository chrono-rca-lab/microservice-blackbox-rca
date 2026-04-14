# FChain-RCA: Black-Box Root Cause Analysis for Microservice Anomalies

A faithful reimplementation of the **FChain** black-box fault localization algorithm (Nguyen et al., ICDCS 2013) on a modern Kubernetes-hosted microservice system (Google Online Boutique).

This project pinpoints faulty microservices using only system-level metrics (CPU, memory, network, disk) without requiring any application-level instrumentation. The core innovation we reimplement is FChain's **predictability-based abnormal change point selection**, which uses FFT-based burst analysis to dynamically threshold change-point detection on a per-metric basis.

> **Priority note for implementers:** Trace-based augmentation is the LOWEST priority. The metrics-only FChain reimplementation is a complete project on its own. Phase 4 (traces) should only be attempted if Phases 0–3 finish ahead of schedule. See [Scope Reduction Guide](#scope-reduction-guide) at the end.

---

## Table of Contents

1. [Goals](#goals)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Repository Structure](#repository-structure)
5. [Phase 0 — Repository Bootstrap](#phase-0--repository-bootstrap)
6. [Phase 1 — Infrastructure & Telemetry](#phase-1--infrastructure--telemetry)
7. [Phase 2 — Fault Injection Harness](#phase-2--fault-injection-harness)
8. [Phase 2.5 — End-to-End Smoke Test](#phase-25--end-to-end-smoke-test)
9. [Phase 3 — FChain Core RCA Engine](#phase-3--fchain-core-rca-engine)
10. [Phase 4 — Trace Augmentation (LOWEST PRIORITY)](#phase-4--trace-augmentation-lowest-priority)
11. [Phase 5 — Evaluation](#phase-5--evaluation)
12. [Phase 6 — Paper & Presentation](#phase-6--paper--presentation)
13. [Timeline Summary](#timeline-summary)
14. [Scope Reduction Guide](#scope-reduction-guide)

---

## Goals

1. **Faithfully reimplement** FChain's metrics-only RCA pipeline on a real microservice benchmark.
2. **Quantitatively measure** its accuracy (precision, recall, top-k) and diagnosis latency under controlled fault injection.
3. **Compare against baselines** (histogram/KL-divergence and topology-only) to validate the value of FChain's predictability filtering.
4. *(Stretch)* Extend with OpenTelemetry trace evidence and measure if/when traces improve ranking.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Users / Load Generator (locust or shell-based)         │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Online Boutique on kind (11 microservices)             │
│  frontend → cart → catalog → currency → payment ...     │
└────────┬────────────────────────────────┬───────────────┘
         ▼                                ▼
┌──────────────────┐              ┌──────────────────┐
│ Prometheus       │              │ Fault Injector   │
│ + cAdvisor       │              │ (kubectl debug + │
│ (1s scrape)      │              │  stress-ng/tc)   │
└────────┬─────────┘              └──────────────────┘
         ▼
┌─────────────────────────────────────────────────────────┐
│  Python RCA Engine (FChain)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Normal Model │→ │ CUSUM Change │→ │ FFT Predict- │   │
│  │ (Markov)     │  │ Point Detect │  │ ability Filt │   │
│  └──────────────┘  └──────────────┘  └──────┬───────┘   │
│                                              ▼          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Tangent      │→ │ Integrated   │→ │ Dependency   │   │
│  │ Rollback     │  │ Pinpointing  │  │ Filter       │   │
│  └──────────────┘  └──────────────┘  └──────┬───────┘   │
└──────────────────────────────────────────────┼──────────┘
                                               ▼
                                  ┌─────────────────────┐
                                  │ Ranked Root Cause   │
                                  │ JSON Output         │
                                  └─────────────────────┘
```

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Application | Google Online Boutique (11 microservices) | Real, modern, pre-instrumented |
| Orchestration | Kubernetes via `kind` | Local, free, reproducible |
| Metrics | Prometheus + cAdvisor + kube-state-metrics (kube-prometheus-stack Helm chart) | Per-pod metrics at high granularity |
| Fault Injection | `kubectl debug` + custom `fault-injector` Docker image with `stress-ng`, `iproute2`, Python | Simple, deterministic |
| RCA Engine | Python 3.10+ (NumPy, SciPy, pandas, requests) | All FChain math: FFT, CUSUM, Markov |
| Evaluation | Python scripts → JSON + matplotlib | Top-k, precision, recall, latency, plots |
| *(Optional)* Traces | OpenTelemetry Collector → Jaeger | LOWEST PRIORITY — see Phase 4 |

---

## Repository Structure

```
fchain-rca/
├── README.md                    # This file
├── Makefile                     # Top-level commands: setup, deploy, run-experiment, eval
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── infra/
│   ├── kind-cluster.yaml        # 1 control-plane + 3 workers
│   ├── deploy-boutique.sh       # Deploys Online Boutique (load generator DISABLED)
│   ├── deploy-monitoring.sh     # Helm install kube-prometheus-stack with 1s scrape
│   └── fault-injector/
│       ├── Dockerfile           # Image with stress-ng, iproute2, python3
│       └── build.sh
│
├── fault_injection/
│   ├── __init__.py
│   ├── inject.py                # CLI: --fault {cpu,mem,net,disk} --target POD --duration SEC
│   ├── faults/
│   │   ├── cpu_hog.sh           # stress-ng --cpu 2 --timeout {duration}s
│   │   ├── mem_leak.py          # Allocates 10MB/sec, never frees
│   │   ├── net_delay.sh         # tc qdisc add dev eth0 root netem delay 200ms
│   │   └── disk_hog.sh          # stress-ng --hdd 1 --hdd-bytes 512M --timeout {duration}s
│   └── ground_truth.py          # Writes GT labels to JSON
│
├── workload/
│   ├── locustfile.py            # Variable load pattern against frontend
│   └── slo_monitor.py           # Tracks p95 latency, flags SLO violations
│
├── rca_engine/
│   ├── __init__.py
│   ├── metrics_client.py        # Pulls from Prometheus HTTP API → pandas DataFrame
│   ├── normal_model.py          # PRESS-style discrete Markov chain prediction
│   ├── change_point.py          # CUSUM + bootstrap change detection
│   ├── predictability_filter.py # FFT-based abnormal change point selection (CORE INNOVATION)
│   ├── tangent_rollback.py      # Refines onset time of abnormal changes
│   ├── pinpoint.py              # Integrated faulty component pinpointing algorithm
│   ├── dependency.py            # Static dependency graph (built from manifest, NOT discovered)
│   ├── trace_augment.py         # OPTIONAL: Phase 4 trace re-ranking
│   └── pipeline.py              # End-to-end: SLO violation → ranked root causes
│
├── eval/
│   ├── run_experiment.py        # Orchestrates: inject → collect → RCA → score
│   ├── batch_runner.py          # Runs the full experiment matrix
│   ├── smoke_test.py            # Phase 2.5 end-to-end sanity check
│   ├── metrics.py               # Precision, recall, top-k accuracy, diagnosis latency
│   ├── baselines/
│   │   ├── histogram_kl.py      # KL-divergence baseline
│   │   └── topology_only.py     # Topology-only baseline
│   └── plots.py                 # matplotlib figures (PR curves, bar charts, box plots)
│
├── experiments/                 # Raw results per run (gitignored except summaries/)
│   └── summaries/               # Aggregated JSON results (committed)
│
├── paper/                       # LaTeX source (Phase 6)
└── tests/                       # Unit tests for RCA components
    ├── test_normal_model.py
    ├── test_change_point.py
    ├── test_predictability_filter.py
    └── test_pinpoint.py
```

---

## Phase 0 — Repository Bootstrap

**Duration:** Day 1
**Owner:** Whole team

**Tasks:**
1. Create the directory structure above.
2. Initialize git, write `.gitignore` (ignore `experiments/raw/`, `__pycache__/`, `*.pyc`, `.venv/`).
3. Write a top-level `Makefile` with stub targets: `setup`, `cluster-up`, `cluster-down`, `deploy`, `inject`, `run-experiment`, `eval`, `clean`.
4. Write `requirements.txt`:
   ```
   numpy>=1.24
   scipy>=1.10
   pandas>=2.0
   requests>=2.31
   matplotlib>=3.7
   pyyaml>=6.0
   locust>=2.15
   pytest>=7.0
   ```
5. Initial commit: `"Phase 0: repo skeleton, README, directory structure"`

---

## Phase 1 — Infrastructure & Telemetry

**Duration:** Days 2–5
**Owner:** Aryan (microservices, workload, metrics setup)

### 1A. Kind cluster + Online Boutique

Create `infra/kind-cluster.yaml`:
```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
  - role: worker
```

Write `infra/deploy-boutique.sh` to:
1. Create the kind cluster: `kind create cluster --config infra/kind-cluster.yaml`
2. Deploy Online Boutique from the official manifest at `https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml`
3. **CRITICAL: Disable the built-in `loadgenerator` deployment** (`kubectl scale deploy/loadgenerator --replicas=0`). We will use our own controllable load generator.

The 11 services to expect: `frontend`, `cartservice`, `productcatalogservice`, `currencyservice`, `paymentservice`, `shippingservice`, `emailservice`, `checkoutservice`, `recommendationservice`, `adservice`, `redis-cart`.

### 1B. Prometheus + cAdvisor

Write `infra/deploy-monitoring.sh` to install `kube-prometheus-stack` via Helm:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.scrapeInterval=1s
```

**IMPORTANT:** Test 1-second scrape on day 2. If kind on a laptop can't sustain 1s scrape with 11 services, fall back to **5s scrape** — this still gives 24 samples per 120s fault, which is sufficient for CUSUM and FFT.

**Metrics to collect per pod** (from cAdvisor):
- `container_cpu_usage_seconds_total` (CPU)
- `container_memory_working_set_bytes` (Memory)
- `container_network_receive_bytes_total` (Net in)
- `container_network_transmit_bytes_total` (Net out)
- `container_fs_reads_bytes_total` (Disk read)
- `container_fs_writes_bytes_total` (Disk write)

### 1C. `metrics_client.py`

Implement a Python client that queries Prometheus via its HTTP API (`/api/v1/query_range`) and returns a `pandas.DataFrame` indexed by `(timestamp, pod_name, metric_name)` with a `value` column.

API:
```python
class PrometheusClient:
    def __init__(self, url: str = "http://localhost:9090"): ...
    def query_range(self, metric: str, start: datetime, end: datetime, step: str = "1s") -> pd.DataFrame: ...
    def get_pod_metrics(self, pods: list[str], start: datetime, end: datetime) -> pd.DataFrame:
        """Returns long-format DataFrame with all 6 metrics for the given pods."""
```

Use `kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090` for local access.

### 1D. Workload generator

Write `workload/locustfile.py` that hits the Online Boutique frontend with a **variable rate** — either a sine wave (period 60s, amplitude 50% of base RPS) or a step pattern (50→150→100→200 RPS over 5-minute windows). This creates non-trivial "normal" workload variation, which is essential for validating that FChain's predictability filter actually filters out workload-induced change points.

### 1E. SLO definition

Define: **"95th percentile frontend request latency < 500ms over a 10-second sliding window."**

Implement `workload/slo_monitor.py` that:
- Tracks response times from the load generator directly (simplest path).
- Computes a rolling p95 over a 10s window.
- Writes a JSON record `{"violation_time_utc": "...", "p95_ms": 612}` when violated.

Why track from the load generator instead of querying Istio/OTel? It's simpler and avoids deploying a service mesh.

**Commits:**
- `"Phase 1A: kind cluster config + Online Boutique deployment"`
- `"Phase 1B: Prometheus stack + metrics_client.py"`
- `"Phase 1C: workload generator with variable load patterns"`
- `"Phase 1D: SLO monitoring"`

**Milestone check:** `make deploy && make load` → query Prometheus → see per-service metrics in a DataFrame.

---

## Phase 2 — Fault Injection Harness

**Duration:** Days 6–8
**Owner:** Bhavishya (fault injection, orchestration, ground truth)

### 2A. Build the fault-injector image

Create `infra/fault-injector/Dockerfile`:
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    stress-ng iproute2 python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*
COPY faults/ /faults/
ENTRYPOINT ["/bin/bash"]
```

Build and load into kind: `kind load docker-image fault-injector:latest`.

**Pick ONE injection mechanism and stick with it: `kubectl debug`.** Don't waste time evaluating init containers vs ephemeral containers vs sidecars.

```bash
kubectl debug -it <target-pod> \
  --image=fault-injector:latest \
  --target=<container-name> \
  -- /faults/cpu_hog.sh 120
```

### 2B. Fault types

Implement the four faults as scripts in `fault_injection/faults/`:

| Fault | Command | Notes |
|---|---|---|
| `cpu_hog.sh` | `stress-ng --cpu 2 --timeout ${1}s` | Saturates 2 cores |
| `mem_leak.py` | Python loop allocating `bytearray(10*1024*1024)` per second, sleeping 1s | Unbounded growth |
| `net_delay.sh` | `tc qdisc add dev eth0 root netem delay 200ms; sleep ${1}; tc qdisc del dev eth0 root netem` | Adds 200ms latency |
| `disk_hog.sh` | `stress-ng --hdd 1 --hdd-bytes 512M --timeout ${1}s` | Saturates disk I/O |

For **multi-component concurrent faults**, inject the same fault into 2–3 pods simultaneously by launching multiple `kubectl debug` calls in parallel.

### 2C. CLI: `inject.py`

```python
# fault_injection/inject.py
# Usage: python inject.py --fault cpu --target checkoutservice --duration 120
```

This script wraps `kubectl debug` and writes the ground truth JSON automatically.

### 2D. Ground truth labeling

`ground_truth.py` writes:
```json
{
  "run_id": "run_042",
  "fault_type": "cpu_hog",
  "target_pods": ["checkoutservice-7f9b8c-x4kl2"],
  "target_services": ["checkoutservice"],
  "inject_time_utc": "2025-03-15T14:22:00Z",
  "duration_seconds": 120,
  "slo_violation_time_utc": "2025-03-15T14:22:08Z"
}
```

**Commits:**
- `"Phase 2A: fault-injector Docker image"`
- `"Phase 2B: fault scripts (cpu, mem, net, disk)"`
- `"Phase 2C: inject.py CLI"`
- `"Phase 2D: ground truth labeling"`

**Milestone check:** `python inject.py --fault cpu --target checkoutservice --duration 120` → SLO violation fires → ground truth JSON saved.

---

## Phase 2.5 — End-to-End Smoke Test

**Duration:** Day 9
**Owner:** Whole team
**This phase is non-negotiable. Do not skip it.**

Before spending two weeks on the FChain algorithm, validate the **entire pipeline** with a stub RCA that just returns the highest-CPU pod over the fault window. This catches integration bugs (Prometheus not scraping, metrics_client returning empty frames, ground truth timing off, etc.) BEFORE they corrupt your Phase 3 debugging.

**Smoke test script** `eval/smoke_test.py`:
1. Deploy everything fresh.
2. Start load generator.
3. Wait 60s for steady state.
4. Inject CpuHog into `checkoutservice` for 120s.
5. Wait for SLO violation.
6. Pull metrics from `[t_violation - 100s, t_violation]`.
7. Run a stub RCA: `argmax(mean_cpu_usage)` across pods.
8. Assert the stub returns `checkoutservice` as rank 1.
9. Print pipeline timing breakdown.

If this passes, the plumbing works and Phase 3 can focus purely on the algorithm.

**Commit:** `"Phase 2.5: end-to-end smoke test with stub RCA"`

---

## Phase 3 — FChain Core RCA Engine

**Duration:** Days 10–25 (16 days — this is the heart of the project)
**Owner:** Manav (RCA pipeline, evaluation)

This phase implements FChain's algorithm faithfully from the paper (Section II-B and II-C). Each subcomponent has a clear math spec and a unit test target.

### 3A. Normal Fluctuation Modeling (Markov prediction)

Reference: FChain paper Section II-B, citing PRESS [12].

**Algorithm:**
1. Discretize each metric time series into `B` bins (default `B=100`) using equal-width bins computed from a sliding window of recent values (window = 300 seconds).
2. Build a transition matrix `T[i][j] = P(value moves from bin i to bin j)` from the window.
3. At each time step `t`, predicted bin at `t+1` = `argmax_j T[bin(x_t)][j]`. Predicted value = bin center.
4. Prediction error `e_t = |x_t - predicted_t|`.

**API:**
```python
class NormalModel:
    def __init__(self, num_bins: int = 100, window_size: int = 300): ...
    def update(self, value: float, timestamp: float) -> None: ...
    def predict_next(self) -> float: ...
    def prediction_error(self, actual: float) -> float: ...
```

**Test:** Feed a sine wave → after warm-up, prediction error should be small (< 5% of amplitude).

### 3B. CUSUM Change Point Detection

Reference: Paper Section II-B, "CUSUM + Bootstrap" [21].

**Algorithm:**
```
S_t = max(0, S_{t-1} + (x_t - mu_0) - k)
```
Where `mu_0` is the pre-fault mean (computed over the first 30s of the look-back window) and `k` is a sensitivity parameter (default `k = 0.5 * sigma_0`). A change point is flagged when `S_t > h` (default `h = 5 * sigma_0`).

Apply CUSUM to each metric in the look-back window `[t_violation - 100, t_violation]` (where `W = 100` seconds per the paper).

**API:**
```python
def detect_change_points(series: np.ndarray, mu_0: float, sigma_0: float,
                          k_factor: float = 0.5, h_factor: float = 5.0) -> list[int]:
    """Returns indices of change points in the series."""
```

**Test:** A flat series with a step at index 50 should yield exactly one change point near index 50.

### 3C. Predictability-Based Abnormal Change Point Filtering (CORE INNOVATION)

Reference: Paper Section II-B, equations and Figure 4. **This is the part that makes FChain different from naive change detection. Implement it carefully.**

**Algorithm — for each candidate change point `x_t` from 3B:**
1. Get the prediction error `e_t` from the normal model (3A).
2. Extract a window of `2Q+1` samples centered on `t`: `X = [x_{t-Q}, ..., x_{t+Q}]` with `Q = 20`.
3. Apply FFT to `X`: `F = np.fft.fft(X)`.
4. Take the **top 90% frequencies** in the magnitude spectrum as "high frequencies." (Sort `|F|` descending; keep the top 90% by count.)
5. Apply inverse FFT on **only those high-frequency components** (zero out the rest) to reconstruct the burst signal `B`.
6. Compute the **90th percentile** of `|B|` → this is the "expected prediction error" threshold.
7. **If `e_t > expected_error_threshold`, mark `x_t` as ABNORMAL.** Otherwise discard.

**Intuition:** Bursty metrics get a higher threshold (their prediction errors are naturally larger), stable metrics get a lower threshold. Fixed thresholds fail because some metrics are inherently more dynamic than others.

**API:**
```python
def filter_abnormal_change_points(
    series: np.ndarray,
    change_points: list[int],
    prediction_errors: np.ndarray,
    Q: int = 20,
    top_k_freq_pct: float = 0.9,
    burst_percentile: float = 90.0
) -> list[int]:
    """Returns the subset of change_points that are abnormal."""
```

**Test:** A series with steady noise + one large fault-injected spike should have the spike survive filtering, while the noise-driven change points should be filtered out.

### 3D. Tangent-Based Rollback (onset time refinement)

Reference: Paper Section II-B, last paragraph.

The selected abnormal change point may sit in the middle of the fault manifestation, not at its start. Roll back to find the true onset.

**Algorithm:**
Starting from each abnormal change point at index `i`:
1. Compute the local slope (tangent) at `i`: `slope_i = (x[i+1] - x[i-1]) / 2`.
2. Look at index `i-1`: compute `slope_{i-1}`.
3. If `|slope_i - slope_{i-1}| < 0.1`, set `i := i-1` and repeat.
4. Stop when the slopes diverge. The final `i` is the refined onset time.

If a component has multiple metrics with abnormal changes, use the **earliest** onset time as the component's onset.

### 3E. Integrated Faulty Component Pinpointing

Reference: Paper Section II-C.

**Algorithm:**
1. For each service, collect its abnormal change point onset times across all 6 metrics. Take the earliest as the service's onset.
2. Sort all services with abnormal changes by onset time (earliest first).
3. **Pinpoint** the earliest service as a root cause.
4. **Concurrent fault detection:** If the next service's onset is within 2 seconds of the pinpointed one, also pinpoint it (concurrent fault).
5. **External cause detection:** If ALL services in the application show abnormal changes AND they all trend in the same direction (all up or all down), flag the anomaly as **external cause** (workload spike / infra issue) and pinpoint NOTHING.
6. **Dependency filtering:** For each remaining abnormal service `C`, check the dependency graph. If there is a path from any already-pinpointed faulty service to `C`, then `C` is propagation — DO NOT pinpoint. If there is NO path, then `C` is an independent fault — pinpoint it.

**API:**
```python
def pinpoint_faults(
    service_onsets: dict[str, float],     # service → earliest abnormal onset time
    service_trends: dict[str, str],       # service → "up" | "down" | "mixed"
    dependency_graph: dict[str, list[str]],  # adjacency list
    concurrency_threshold_s: float = 2.0
) -> list[str]:
    """Returns ranked list of pinpointed faulty services."""
```

### 3F. Static Dependency Graph (NOT runtime discovery)

**Important deviation from the paper:** Instead of implementing black-box dependency discovery via network correlation (which is noisy and time-consuming), build the Online Boutique dependency graph **statically from the application's known architecture**. This is faithful to the paper's spirit — FChain itself uses an offline tool (Sherlock) for dependency discovery.

Hardcode this in `dependency.py`:
```python
ONLINE_BOUTIQUE_DEPENDENCIES = {
    "frontend": ["productcatalogservice", "currencyservice", "cartservice",
                 "recommendationservice", "shippingservice", "checkoutservice", "adservice"],
    "checkoutservice": ["productcatalogservice", "shippingservice", "paymentservice",
                        "emailservice", "currencyservice", "cartservice"],
    "recommendationservice": ["productcatalogservice"],
    "cartservice": ["redis-cart"],
    "productcatalogservice": [],
    "currencyservice": [],
    "paymentservice": [],
    "shippingservice": [],
    "emailservice": [],
    "adservice": [],
    "redis-cart": [],
}
```

Write a `has_path(graph, src, dst)` helper using BFS.

**Justification for the paper:** "We use a static dependency graph derived from the application architecture, consistent with FChain's approach of discovering dependencies offline. Runtime dependency discovery is orthogonal to FChain's core contribution (predictability-based change point selection) and is not evaluated here."

### 3G. End-to-end pipeline + output format

Wire everything together in `rca_engine/pipeline.py`:

```python
def run_rca(
    violation_time: datetime,
    look_back_seconds: int = 100,
    prom_client: PrometheusClient = ...,
) -> dict:
    # 1. Pull metrics for [t_v - W, t_v] for all 11 services
    # 2. For each (service, metric):
    #    a. Run normal model → prediction errors
    #    b. Run CUSUM → candidate change points
    #    c. Run predictability filter → abnormal change points
    #    d. Run tangent rollback → refined onset times
    # 3. Aggregate per service: earliest onset across metrics
    # 4. Run integrated pinpointing (3E) with static dependency graph (3F)
    # 5. Output ranked JSON
```

**Output format:**
```json
{
  "run_id": "run_042",
  "violation_time_utc": "2025-03-15T14:22:08Z",
  "suspects": [
    {"rank": 1, "service": "checkoutservice", "onset_time": "14:22:03", "confidence": 0.92, "abnormal_metrics": ["cpu_usage"]},
    {"rank": 2, "service": "frontend", "onset_time": "14:22:07", "confidence": 0.45, "abnormal_metrics": ["network_in"]}
  ],
  "diagnosis_latency_seconds": 4.2
}
```

**Commits:**
- `"Phase 3A: normal fluctuation modeling (Markov prediction)"`
- `"Phase 3B: CUSUM change point detection"`
- `"Phase 3C: FFT predictability filter (core FChain innovation)"`
- `"Phase 3D: tangent rollback for onset refinement"`
- `"Phase 3E: integrated pinpointing algorithm"`
- `"Phase 3F: static dependency graph"`
- `"Phase 3G: end-to-end RCA pipeline + JSON output"`

**Milestone check:** Inject CpuHog into `checkoutservice` → run pipeline → `checkoutservice` is rank 1.

---

## Phase 4 — Trace Augmentation (LOWEST PRIORITY)

**Duration:** Days 26–30 IF AND ONLY IF Phase 3 completes early. Otherwise SKIP entirely.
**Owner:** Manav (if undertaken)

**This entire phase is optional and should only be attempted if Phases 0–3 finish ahead of schedule.** The metrics-only FChain reimplementation is a complete project on its own. Traces are a stretch goal.

If undertaken:
1. Deploy OpenTelemetry Collector + Jaeger (Online Boutique is already instrumented).
2. Extract trace evidence per fault window: per-service self-time, per-service error rate, trace-derived call graph.
3. Re-rank FChain's output:
   ```
   final_score(s) = α * fchain_score(s)
                  + β * latency_anomaly_score(s)
                  + γ * error_rate_anomaly_score(s)
                  + δ * trace_dependency_boost(s)
   ```
4. *(If time permits)* Ablation: simulate partial trace coverage (drop X% of traces) to measure how much coverage is needed before traces actually help.

**Do not start this phase unless Phase 3 completes by day 22.** If you start it and fall behind, abandon it and finalize the metrics-only results.

---

## Phase 5 — Evaluation

**Duration:** Days 26–35 (or 31–35 if Phase 4 was attempted)
**Owner:** Whole team, Manav leads

### 5A. Experiment matrix

| Fault Type | Single-Service Targets | Multi-Service (concurrent) |
|---|---|---|
| CpuHog | checkoutservice, productcatalogservice, frontend | checkout + productcatalog |
| MemLeak | checkoutservice, cartservice, frontend | cart + redis-cart |
| NetDelay | frontend, paymentservice | frontend + shipping |
| DiskHog | redis-cart, cartservice | redis + cart |

**~16 cells × 10 runs each = 160 runs.** At ~5 minutes each plus teardown, budget **20+ hours** of wall-clock time. If behind schedule, drop to **5 runs per cell = 80 runs** — still statistically defensible.

### 5B. Metrics

| Metric | Definition |
|---|---|
| Precision | `Ntp / (Ntp + Nfp)` |
| Recall | `Ntp / (Ntp + Nfn)` |
| Top-1 accuracy | Was rank-1 the true root cause? |
| Top-3 accuracy | Was the true root cause in the top 3? |
| Top-5 accuracy | Was the true root cause in the top 5? |
| Diagnosis latency | Seconds from SLO violation to RCA output |

### 5C. Baselines (each baseline = ~1.5 days, budget accordingly)

1. **Histogram (KL divergence)** — `eval/baselines/histogram_kl.py`. For each metric, compute KL divergence between the histogram of recent data (look-back window) and the histogram of all historical data. Rank services by max anomaly score.
2. **Topology-only** — `eval/baselines/topology_only.py`. Detect abnormal services using simple z-score outlier detection (no FFT filtering), then pinpoint based on dependency order alone.
3. **FChain (metrics-only)** — Phase 3 output.
4. *(Optional)* **FChain + Traces** — only if Phase 4 was completed.

### 5D. Ablation study

- FChain **with vs. without predictability filtering** (replace 3C with a fixed threshold) — validates that the FFT filter matters. **This is the most important ablation.**
- FChain **with vs. without dependency information** (skip 3F filtering step).
- Vary look-back window `W ∈ {50, 100, 200, 500}` seconds.
- Vary concurrency threshold `∈ {1, 2, 5}` seconds.

### 5E. Plots

Generate in `eval/plots.py`:
- Precision-Recall curves per fault type (matching FChain paper Figs. 6–10).
- Bar charts for Top-k accuracy across fault types.
- Box plots for diagnosis latency.
- Ablation tables (markdown + LaTeX).

**Commits:**
- `"Phase 5A: experiment batch runner"`
- `"Phase 5B: evaluation metrics"`
- `"Phase 5C: histogram-KL baseline"`
- `"Phase 5C: topology-only baseline"`
- `"Phase 5D: ablation experiment configs"`
- `"Phase 5E: plotting + result tables"`

---

## Phase 6 — Paper & Presentation

**Duration:** Days 36–42

**Paper structure (ACM format):**
1. Introduction — problem, motivation, contributions
2. Background & Related Work — FChain, positioning
3. System Design — architecture, FChain reimplementation details
4. Implementation — Online Boutique setup, telemetry pipeline, fault injection, **explicit note that we use a static dependency graph and omit online validation**
5. Evaluation — experiment setup, results, ablations
6. Discussion — limitations, threats to validity (no production cloud, single benchmark, omitted online validation)
7. Conclusion

**Slides:** Reuse plots. Key slides: problem, FChain algorithm walkthrough (especially the FFT filter), system architecture, results summary, ablation findings.

---

## Timeline Summary

| Phase | Days | What | Milestone |
|---|---|---|---|
| 0 | 1 | Repo setup | Clean skeleton committed |
| 1 | 2–5 | Infra + telemetry | Metrics flowing from live services |
| 2 | 6–8 | Fault injection | Can inject faults + record ground truth |
| 2.5 | 9 | End-to-end smoke test | Stub RCA returns correct service |
| 3 | 10–25 | FChain core (16 days) | Metrics-only RCA pinpoints correctly |
| 4 | (26–30) | Trace augmentation | OPTIONAL — skip if behind |
| 5 | 26–35 | Evaluation | Full matrix + baselines + plots |
| 6 | 36–42 | Paper + presentation | Submission-ready |

**Workload split (3 people):**
- **Aryan:** Phase 1 (microservices, workload, metrics)
- **Bhavishya:** Phase 2 (fault injection, orchestration, ground truth)
- **Manav:** Phase 3 (RCA pipeline) and Phase 5 lead (evaluation)
- All three: Phase 0, Phase 2.5, Phase 6

---

## Quick Start (once built)

```bash
make setup              # Install Python deps, check kubectl/kind/helm
make cluster-up         # Create kind cluster
make deploy             # Deploy Online Boutique + monitoring
make load               # Start workload generator (background)
make smoke-test         # Phase 2.5 sanity check
make run-experiment FAULT=cpu TARGET=checkoutservice DURATION=120
make eval               # Run full experiment matrix
make plots              # Generate figures for paper
make cluster-down       # Tear down
```
