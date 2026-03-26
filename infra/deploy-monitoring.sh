#!/usr/bin/env bash
# Deploy kube-prometheus-stack into the monitoring namespace.
set -euo pipefail

NAMESPACE="monitoring"
RELEASE="kube-prom-stack"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# 1. Add / update prometheus-community Helm repo
echo "[monitoring] Adding prometheus-community Helm repo"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 2. Install kube-prometheus-stack with custom values
echo "[monitoring] Creating namespace '${NAMESPACE}'"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "[monitoring] Installing ${RELEASE}"
helm upgrade --install "${RELEASE}" prometheus-community/kube-prometheus-stack \
  --namespace "${NAMESPACE}" \
  --values "${SCRIPT_DIR}/prometheus-values.yaml" \
  --wait

# 3. Port-forward Prometheus → localhost:9090 and Grafana → localhost:3000
echo "[monitoring] Port-forwarding Prometheus → localhost:9090 (background)"
kubectl port-forward -n "${NAMESPACE}" svc/${RELEASE}-kube-prome-prometheus 9090:9090 &

echo "[monitoring] Port-forwarding Grafana → localhost:3000 (background)"
kubectl port-forward -n "${NAMESPACE}" svc/${RELEASE}-grafana 3000:80 &

echo "[monitoring] Done.  Prometheus: http://localhost:9090  Grafana: http://localhost:3000 (admin/admin)"
