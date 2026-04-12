#!/usr/bin/env bash
# Deploy Chaos Mesh to the fchain-rca kind cluster via Helm.
set -euo pipefail

CHAOS_MESH_VERSION="2.7.0"
NAMESPACE="chaos-mesh"

echo "[chaos-mesh] Verifying containerd socket path on kind nodes …"
# kind nodes use containerd; the socket is mounted from the host
SOCKET_PATH="/run/containerd/containerd.sock"
if docker exec fchain-rca-control-plane ls -la "${SOCKET_PATH}" >/dev/null 2>&1; then
  echo "[chaos-mesh]   socket found at ${SOCKET_PATH}"
else
  echo "[chaos-mesh] WARNING: socket not found at ${SOCKET_PATH} — adjust SOCKET_PATH if install fails"
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
