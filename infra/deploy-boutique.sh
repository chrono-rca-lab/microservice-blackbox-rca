#!/usr/bin/env bash
# Deploy Google Online Boutique to the fchain-rca kind cluster.
set -euo pipefail

CLUSTER_NAME="fchain-rca"
NAMESPACE="boutique"
MANIFESTS_URL="https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Create kind cluster (skip if it already exists)
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "[boutique] Cluster '${CLUSTER_NAME}' already exists — skipping creation."
else
  echo "[boutique] Creating kind cluster '${CLUSTER_NAME}' …"
  mkdir -p /tmp/kind-volumes
  kind create cluster --config "${SCRIPT_DIR}/kind-cluster.yaml"
fi

# 2. Apply Online Boutique manifests
echo "[boutique] Creating namespace '${NAMESPACE}' …"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "[boutique] Applying Online Boutique manifests …"
kubectl apply -n "${NAMESPACE}" -f "${MANIFESTS_URL}"

# 3. Wait for all pods to be ready (loadgenerator must exist before we scale it down)
echo "[boutique] Waiting for all pods to be ready (timeout 300s) …"
kubectl wait --for=condition=ready pod --all -n "${NAMESPACE}" --timeout=300s

# 4. Disable the built-in load generator (we control load ourselves)
echo "[boutique] Scaling loadgenerator to 0 …"
kubectl scale deployment loadgenerator --replicas=0 -n "${NAMESPACE}"

# 5. Port-forward frontend to localhost:8080 (background)
echo "[boutique] Port-forwarding frontend → localhost:8080 (background) …"
kubectl port-forward -n "${NAMESPACE}" svc/frontend 8080:80 &
echo "[boutique] Done.  Frontend: http://localhost:8080"
