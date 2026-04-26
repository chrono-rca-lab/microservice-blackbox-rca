#!/usr/bin/env bash
# Deploy Chaos Mesh via Helm.
#
# Usage:
#   bash infra/deploy-chaos-mesh.sh           # kind cluster (default)
#   bash infra/deploy-chaos-mesh.sh --k3s     # k3s/VCL cluster
#
# kind uses /run/containerd/containerd.sock (verified via docker exec).
# k3s uses /run/k3s/containerd/containerd.sock (no docker exec needed).
set -euo pipefail

CHAOS_MESH_VERSION="2.7.0"
NAMESPACE="chaos-mesh"
K3S=false

for arg in "$@"; do
  [[ "${arg}" == "--k3s" ]] && K3S=true
done

if [[ "${K3S}" == true ]]; then
  SOCKET_PATH="/run/k3s/containerd/containerd.sock"
  echo "[chaos-mesh] k3s mode — using socket ${SOCKET_PATH}"
else
  echo "[chaos-mesh] Verifying containerd socket path on kind nodes …"
  # kind nodes use containerd; the socket is mounted from the host
  SOCKET_PATH="/run/containerd/containerd.sock"
  if docker exec fchain-rca-control-plane ls -la "${SOCKET_PATH}" >/dev/null 2>&1; then
    echo "[chaos-mesh]   socket found at ${SOCKET_PATH}"
  else
    echo "[chaos-mesh] WARNING: socket not found at ${SOCKET_PATH} — adjust SOCKET_PATH if install fails"
  fi
fi

echo "[chaos-mesh] Adding Helm repo …"
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update

echo "[chaos-mesh] Installing Chaos Mesh ${CHAOS_MESH_VERSION} …"
helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace "${NAMESPACE}" --create-namespace \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath="${SOCKET_PATH}" \
  --version "${CHAOS_MESH_VERSION}" \
  --wait --timeout 300s

echo "[chaos-mesh] Waiting for all pods to be ready …"
kubectl wait --for=condition=ready pod --all -n "${NAMESPACE}" --timeout=180s

echo "[chaos-mesh] Chaos Mesh deployed successfully."
echo "  Dashboard: kubectl port-forward -n ${NAMESPACE} svc/chaos-dashboard 2333:2333"
