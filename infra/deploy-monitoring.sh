#!/usr/bin/env bash
# Deploy kube-prometheus-stack into the monitoring namespace.
#
# Usage:
#   bash infra/deploy-monitoring.sh                          # kind cluster (default)
#   bash infra/deploy-monitoring.sh --k3s                   # k3s/VCL 2-node cluster
#   bash infra/deploy-monitoring.sh --k3s --monitoring-node # k3s/VCL 4-node cluster (pins to Machine 3)
#
# The --k3s flag adds two extra Helm values required for k3s:
#   kubelet.serviceMonitor.https=false
#     k3s exposes kubelet/cAdvisor metrics over plain HTTP, not HTTPS.
#     Without this Prometheus cannot scrape container CPU/memory metrics.
#   prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
#     Allows Prometheus to discover ServiceMonitors from all namespaces,
#     not just those co-installed with this Helm release.
set -euo pipefail

NAMESPACE="monitoring"
RELEASE="kube-prom-stack"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K3S=false
MONITORING_NODE=false

for arg in "$@"; do
  [[ "${arg}" == "--k3s" ]]             && K3S=true
  [[ "${arg}" == "--monitoring-node" ]] && MONITORING_NODE=true
done

# 1. Add / update prometheus-community Helm repo
echo "[monitoring] Adding prometheus-community Helm repo"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 2. Install kube-prometheus-stack with custom values
echo "[monitoring] Creating namespace '${NAMESPACE}'"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

HELM_ARGS=(
  --namespace "${NAMESPACE}"
  --values "${SCRIPT_DIR}/prometheus-values.yaml"
)

if [[ "${K3S}" == true ]]; then
  echo "[monitoring] k3s mode — adding kubelet HTTP scrape overrides"
  HELM_ARGS+=(
    --set kubelet.serviceMonitor.https=false
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
  )
fi

if [[ "${MONITORING_NODE}" == true ]]; then
  echo "[monitoring] monitoring-node mode — pinning components to role=monitoring node"
  HELM_ARGS+=(--values "${SCRIPT_DIR}/prometheus-values-monitoring-node.yaml")
fi

echo "[monitoring] Installing ${RELEASE}"
helm upgrade --install "${RELEASE}" prometheus-community/kube-prometheus-stack \
  "${HELM_ARGS[@]}" \
  --wait

# 3. Port-forward Prometheus → localhost:9090 and Grafana → localhost:3000
echo "[monitoring] Port-forwarding Prometheus → localhost:9090 (background)"
kubectl port-forward -n "${NAMESPACE}" svc/${RELEASE}-kube-prome-prometheus 9090:9090 &

echo "[monitoring] Port-forwarding Grafana → localhost:3000 (background)"
kubectl port-forward -n "${NAMESPACE}" svc/${RELEASE}-grafana 3000:80 &

echo "[monitoring] Done.  Prometheus: http://localhost:9090  Grafana: http://localhost:3000 (admin/admin)"
