#!/usr/bin/env bash
# Deploy patched Online Boutique to the kind cluster (creates cluster if needed).
#
# Usage:
#   bash infra/deploy-boutique.sh
#   bash infra/deploy-boutique.sh --fresh   # delete namespace boutique first, then deploy
set -euo pipefail

CLUSTER_NAME="fchain-rca"
NAMESPACE="boutique"
FRESH=false
for arg in "$@"; do
  if [[ "${arg}" == "--fresh" ]]; then
    FRESH=true
  fi
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
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
"${PYTHON}" "${SCRIPT_DIR}/patch_manifests.py" \
  --input  "${RAW_MANIFEST}" \
  --output "${PATCHED_MANIFEST}"

# 3. Apply patched manifests
if [[ "${FRESH}" == true ]]; then
  if kubectl get namespace "${NAMESPACE}" &>/dev/null; then
    phase="$(kubectl get ns "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")"
    if [[ "${phase}" == "Terminating" ]]; then
      echo "[boutique] --fresh: namespace '${NAMESPACE}' stuck Terminating — clearing finalizers …"
      kubectl patch ns "${NAMESPACE}" -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
      if command -v jq &>/dev/null; then
        kubectl get ns "${NAMESPACE}" -o json 2>/dev/null \
          | jq 'del(.metadata.finalizers) | .spec.finalizers = []' \
          | kubectl replace --raw "/api/v1/namespaces/${NAMESPACE}/finalize" -f - >/dev/null 2>&1 || true
      fi
      t0="$(date +%s)"
      while kubectl get namespace "${NAMESPACE}" &>/dev/null; do
        if (( $(date +%s) - t0 > 120 )); then
          echo "[boutique] ERROR: namespace '${NAMESPACE}' still present after finalize wait." >&2
          exit 1
        fi
        sleep 2
      done
      echo "[boutique] --fresh: namespace removed."
    else
      echo "[boutique] --fresh: deleting namespace '${NAMESPACE}' (all workloads) …"
      kubectl delete namespace "${NAMESPACE}" --wait=true
    fi
  else
    echo "[boutique] --fresh: namespace '${NAMESPACE}' does not exist yet — skipping delete."
  fi
fi

echo "[boutique] Creating namespace '${NAMESPACE}' …"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "[boutique] Applying patched Online Boutique manifests …"
kubectl apply -n "${NAMESPACE}" -f "${PATCHED_MANIFEST}"

WAIT_TIMEOUT="600s"
echo "[boutique] Waiting for all pods to be ready (timeout ${WAIT_TIMEOUT}) …"
if ! kubectl wait --for=condition=ready pod --all -n "${NAMESPACE}" --timeout="${WAIT_TIMEOUT}"; then
  echo "[boutique] ERROR: readiness wait timed out." >&2
  kubectl get pods -n "${NAMESPACE}" -o wide >&2 || true
  echo "[boutique] Hint: kubectl rollout restart deployment --all -n ${NAMESPACE}" >&2
  exit 1
fi

echo "[boutique] Scaling loadgenerator to 0 …"
kubectl scale deployment loadgenerator --replicas=0 -n "${NAMESPACE}" || true

echo "[boutique] Port-forwarding frontend → localhost:8080 (background) …"
kubectl port-forward -n "${NAMESPACE}" svc/frontend 8080:80 &
echo "[boutique] Done.  Frontend: http://localhost:8080"
