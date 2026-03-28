# microservice-blackbox-rca

Black-box root-cause analysis for microservice failures using metrics, change-point detection, and fault-chain pinpointing.

---

## Prerequisites

Install the following tools before anything else.

**Docker Desktop** — https://www.docker.com/products/docker-desktop
Start it and wait for the whale icon in the menu bar to stop animating.

```bash
brew install kubectl kind helm
pip install -r requirements.txt
mkdir -p /tmp/kind-volumes
```

---

## 1. Deploy Online Boutique

Creates the kind cluster (4 nodes), patches all boutique manifests (removes `readOnlyRootFilesystem`, adds `NET_ADMIN`, mounts emptyDir at `/tmp`), and deploys all microservices. Takes 3–5 min on first run due to image pulls.

```bash
bash infra/deploy-boutique.sh
```

Frontend: **http://localhost:8080**

If the port-forward dies:
```bash
kubectl port-forward -n boutique svc/frontend 8080:80 &
```

---

## 2. Deploy Monitoring (Prometheus + Grafana)

```bash
bash infra/deploy-monitoring.sh
```

| Service    | URL                    | Credentials   |
|------------|------------------------|---------------|
| Prometheus | http://localhost:9090  | —             |
| Grafana    | http://localhost:3000  | admin / admin |

If a port-forward dies:
```bash
kubectl port-forward -n monitoring svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &
kubectl port-forward -n monitoring svc/kube-prom-stack-grafana 3000:80 &
```

---

## 3. Verify Metrics

In Prometheus, run:
```
node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{namespace="boutique"}
```
Should return time-series for each boutique pod.

In Grafana: **Dashboards → Kubernetes / Compute Resources / Pod**, set namespace to `boutique`, time range to last 15 minutes.

---

## 4. Run Load Generator

```bash
python infra/loadgen.py --duration 300 --rps 5 --pattern sine
```

Options: `--pattern` accepts `sine`, `step`, or `constant`. Watch CPU lines move in Grafana.

---

## 5. Fault Injection

### Injectable services (have /bin/sh)

`adservice`, `currencyservice`, `emailservice`, `paymentservice`, `recommendationservice`

The following are distroless (no shell) and cannot be targeted: `cartservice`, `checkoutservice`, `frontend`, `productcatalogservice`, `shippingservice`.

### Inject a single fault manually

```bash
python fault_injection/inject.py --fault cpu_hog   --service recommendationservice --duration 60
python fault_injection/inject.py --fault mem_leak   --service paymentservice        --duration 60
python fault_injection/inject.py --fault disk_hog   --service currencyservice       --duration 60
```

Concurrent faults (two services at once):
```bash
python fault_injection/inject.py --fault cpu_hog --service emailservice --concurrent currencyservice --duration 60
```

Add `--dry-run` to print the command without executing.

---

## 6. Run a Single Experiment (inject → collect → RCA → score)

Each `run_experiment` call:
1. Starts the load generator
2. Waits for a 60 s baseline
3. Injects the fault in a background subprocess
4. Polls for an SLO violation (p95 latency > 500 ms)
5. Collects a Prometheus metrics window
6. Runs the RCA engine
7. Saves artifacts to `experiments/<run_id>/`

### cpu_hog experiments

```bash
python eval/run_experiment.py --fault cpu_hog --service recommendationservice --duration 120
python eval/run_experiment.py --fault cpu_hog --service emailservice           --duration 120
python eval/run_experiment.py --fault cpu_hog --service currencyservice        --duration 120
python eval/run_experiment.py --fault cpu_hog --service adservice              --duration 120
```

### mem_leak experiments

```bash
python eval/run_experiment.py --fault mem_leak --service recommendationservice --duration 120
python eval/run_experiment.py --fault mem_leak --service paymentservice         --duration 120
python eval/run_experiment.py --fault mem_leak --service emailservice           --duration 120
```

### disk_hog experiments

```bash
python eval/run_experiment.py --fault disk_hog --service currencyservice --duration 120
python eval/run_experiment.py --fault disk_hog --service emailservice    --duration 120
python eval/run_experiment.py --fault disk_hog --service paymentservice  --duration 120
```

### Concurrent fault experiments

```bash
python eval/run_experiment.py --fault cpu_hog  --service emailservice          --concurrent currencyservice --duration 120
python eval/run_experiment.py --fault mem_leak --service recommendationservice  --concurrent paymentservice  --duration 120
```

Each experiment takes roughly **5–6 minutes** (60 s baseline + 120 s fault + propagation + recovery). Artifacts land in `experiments/<run_id>/`:

```
experiments/20250328_143201/
├── ground_truth.json   # injected fault + target services
├── timeline.json       # timestamps for each phase
├── metrics.parquet     # per-service metric matrix
└── rca_results.json    # RCA engine output
```

---

## 7. Run the Full Experiment Batch

Once individual experiments look correct, run all 12 back-to-back with a 120 s cooldown between them:

```bash
python eval/run_batch.py --matrix experiments/experiment_matrix.yaml
```

Options:
- `--cooldown 60` — override the cooldown in the YAML
- `--dry-run` — print all commands without running any experiment

The batch runner exits non-zero if any experiment fails and prints a summary at the end.

---

## 8. Fetch Metrics Directly

```bash
python -m rca_engine.metrics_client
```

Prints a per-service, per-metric summary table for the last 5 minutes.

---

## Teardown

```bash
kind delete cluster --name fchain-rca
```
