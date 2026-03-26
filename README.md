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

Creates the kind cluster (4 nodes) and deploys all microservices. Takes 3–5 min on first run due to image pulls.

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

| Service    | URL                        | Credentials  |
|------------|----------------------------|--------------|
| Prometheus | http://localhost:9090      | —            |
| Grafana    | http://localhost:3000      | admin / admin |

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

## Fetch Metrics via Python

```bash
python -m rca_engine.metrics_client
```

Prints a per-service, per-metric summary table for the last 5 minutes.

---

## Teardown

```bash
kind delete cluster --name fchain-rca
```
