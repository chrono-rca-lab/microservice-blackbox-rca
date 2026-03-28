#!/usr/bin/env bash
# Deploy Google Online Boutique (patched for fault injection) to the fchain-rca kind cluster.
set -euo pipefail

CLUSTER_NAME="fchain-rca"
NAMESPACE="boutique"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_MANIFEST="${SCRIPT_DIR}/boutique-manifests.yaml"
PATCHED_MANIFEST="${SCRIPT_DIR}/boutique-manifests-patched.yaml"

# 1. Create kind cluster (skip if it already exists)
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "[boutique] Cluster '${CLUSTER_NAME}' already exists — skipping creation."
else
  echo "[boutique] Creating kind cluster '${CLUSTER_NAME}' …"
  mkdir -p /tmp/kind-volumes
  kind create cluster --config "${SCRIPT_DIR}/kind-cluster.yaml"
fi

# 2. Download + patch the boutique manifests
echo "[boutique] Patching manifests for fault injection …"
python3 "${SCRIPT_DIR}/patch_manifests.py" \
  --input  "${RAW_MANIFEST}" \
  --output "${PATCHED_MANIFEST}"

# 3. Apply patched manifests
echo "[boutique] Creating namespace '${NAMESPACE}' …"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "[boutique] Applying patched Online Boutique manifests …"
kubectl apply -n "${NAMESPACE}" -f "${PATCHED_MANIFEST}"

# 4. Wait for all pods to be ready
echo "[boutique] Waiting for all pods to be ready (timeout 300s) …"
kubectl wait --for=condition=ready pod --all -n "${NAMESPACE}" --timeout=300s

# 5. Disable the built-in load generator (we control load ourselves)
echo "[boutique] Scaling loadgenerator to 0 …"
kubectl scale deployment loadgenerator --replicas=0 -n "${NAMESPACE}"

# 6. Port-forward frontend to localhost:8080 (background)
echo "[boutique] Port-forwarding frontend → localhost:8080 (background) …"
kubectl port-forward -n "${NAMESPACE}" svc/frontend 8080:80 &
echo "[boutique] Done.  Frontend: http://localhost:8080"
