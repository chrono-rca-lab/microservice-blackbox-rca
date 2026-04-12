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

## 3. Deploy Chaos Mesh (fault injection engine)

```bash
bash infra/deploy-chaos-mesh.sh
```

Installs Chaos Mesh 2.7.0 into the `chaos-mesh` namespace via Helm.

---

## 4. Verify Metrics

In Prometheus, run:
```
node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{namespace="boutique"}
```
Should return time-series for each boutique pod.

In Grafana: **Dashboards → Kubernetes / Compute Resources / Pod**, set namespace to `boutique`, time range to last 15 minutes.

---

## 5. Run Load Generator

```bash
python infra/loadgen.py --duration 300 --rps 5 --pattern sine
```

Options: `--pattern` accepts `sine`, `step`, or `constant`. Watch CPU lines move in Grafana.

---

## 6. Fault Injection

Fault injection is handled by two injectors depending on fault type:

| Injector | Faults | Target services |
|---|---|---|
| `chaos_inject.py` (Chaos Mesh) | `cpu_hog`, `mem_leak`, `net_delay`, `packet_loss` | **All 11 services** — works on distroless Go containers |
| `inject.py` (kubectl exec) | `disk_hog` | Shell-only: `adservice`, `currencyservice`, `emailservice`, `paymentservice`, `recommendationservice` |

`disk_hog` uses the exec path because IOChaos requires FUSE kernel support not available on kind clusters.

### Inject a single fault manually

```bash
# Chaos Mesh — works on any service including distroless
python fault_injection/chaos_inject.py --fault cpu_hog   --service frontend       --duration 60
python fault_injection/chaos_inject.py --fault net_delay --service cartservice     --duration 60
python fault_injection/chaos_inject.py --fault mem_leak  --service recommendationservice --duration 60

# kubectl exec fallback — disk_hog only, shell-capable services only
python fault_injection/inject.py --fault disk_hog --service currencyservice --duration 60
```

Concurrent faults (two services at once):
```bash
python fault_injection/chaos_inject.py --fault cpu_hog --service emailservice --concurrent currencyservice --duration 60
```

---

## 7. Run a Single Experiment (inject → collect → RCA → score)

Each `run_experiment` call:
1. Starts the load generator
2. Waits for a 60 s baseline
3. Injects the fault (`chaos_inject.py` for most faults, `inject.py` for `disk_hog`)
4. Waits the full fault duration (SLO monitor runs in background — metadata only)
5. Collects a Prometheus metrics window anchored to injection time
6. Runs the RCA engine
7. Saves artifacts to `experiments/<run_id>/`

### cpu_hog experiments

```bash
# All services — including distroless Go (frontend, checkoutservice, etc.)
python eval/run_experiment.py --fault cpu_hog --service frontend           --duration 120
python eval/run_experiment.py --fault cpu_hog --service checkoutservice    --duration 120
python eval/run_experiment.py --fault cpu_hog --service adservice          --duration 120
python eval/run_experiment.py --fault cpu_hog --service recommendationservice --duration 120
python eval/run_experiment.py --fault cpu_hog --service currencyservice    --duration 120
python eval/run_experiment.py --fault cpu_hog --service emailservice       --duration 120
```

### mem_leak experiments

```bash
python eval/run_experiment.py --fault mem_leak --service recommendationservice --duration 120
python eval/run_experiment.py --fault mem_leak --service emailservice           --duration 120
python eval/run_experiment.py --fault mem_leak --service frontend               --duration 120
```

### net_delay experiments

```bash
# NetworkChaos works on all services including distroless
python eval/run_experiment.py --fault net_delay --service frontend              --duration 120
python eval/run_experiment.py --fault net_delay --service cartservice           --duration 120
python eval/run_experiment.py --fault net_delay --service checkoutservice       --duration 120
python eval/run_experiment.py --fault net_delay --service productcatalogservice --duration 120
```

### disk_hog experiments (exec fallback, shell-capable services only)

```bash
python eval/run_experiment.py --fault disk_hog --service currencyservice --duration 120
python eval/run_experiment.py --fault disk_hog --service emailservice    --duration 120
```

### Concurrent fault experiments

```bash
python eval/run_experiment.py --fault cpu_hog  --service emailservice         --concurrent currencyservice --duration 120
python eval/run_experiment.py --fault net_delay --service frontend             --concurrent checkoutservice --duration 120
```

Each experiment takes roughly **4 minutes** (60 s baseline + fault duration + recovery). Artifacts land in `experiments/<run_id>/`:

```
experiments/20260412_201315/
├── ground_truth.json   # injected fault + target services
├── timeline.json       # timestamps + SLO violation metadata
├── metrics.parquet     # per-service metric matrix (all 11 services)
└── rca_results.json    # RCA engine output
```

---

## 8. Run the Full Experiment Batch

```bash
python eval/run_batch.py --matrix experiments/experiment_matrix.yaml
```

Options:
- `--cooldown 60` — override the cooldown between experiments
- `--dry-run` — print all commands without running any experiment

The batch runner exits non-zero if any experiment fails and prints a summary at the end.

---

## 9. Fetch Metrics Directly

```bash
python -m rca_engine.metrics_client
```

Prints a per-service, per-metric summary table for the last 5 minutes.

---

## Teardown

```bash
kind delete cluster --name fchain-rca
```
