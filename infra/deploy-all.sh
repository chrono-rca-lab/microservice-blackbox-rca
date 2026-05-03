#!/usr/bin/env bash
# Fresh stack: monitoring → chaos-mesh → boutique (with optional stuck-namespace fix).
#
# Needs: Docker, kubectl, helm, and a kind cluster created by deploy-boutique.sh if absent.
# jq helps if boutique is stuck Terminating while finalizers clear.
#
# Usage:
#   bash infra/deploy-all.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="fchain-rca"

_finalize_ns_if_stuck() {
  local ns=$1
  if ! kubectl get ns "${ns}" &>/dev/null; then
    return 0
  fi
  local phase
  phase=$(kubectl get ns "${ns}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
  if [[ "${phase}" != "Terminating" ]]; then
    return 0
  fi
  echo "[deploy-all] Namespace '${ns}' is Terminating — clearing finalizers …"
  kubectl patch ns "${ns}" -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
  if command -v jq &>/dev/null; then
    kubectl get ns "${ns}" -o json 2>/dev/null \
      | jq 'del(.metadata.finalizers) | .spec.finalizers = []' \
      | kubectl replace --raw "/api/v1/namespaces/${ns}/finalize" -f - >/dev/null 2>&1 || true
  fi
  local t0
  t0=$(date +%s)
  while kubectl get ns "${ns}" &>/dev/null; do
    if (( $(date +%s) - t0 > 120 )); then
      echo "[deploy-all] ERROR: namespace '${ns}' still exists after finalize wait." >&2
      exit 1
    fi
    sleep 2
  done
  echo "[deploy-all] Namespace '${ns}' is gone."
}

_finalize_ns_if_stuck boutique

echo "=============================================="
echo "  [deploy-all] 1/3  Monitoring (Helm + wait)"
echo "=============================================="
bash "${SCRIPT_DIR}/deploy-monitoring.sh"

echo ""
echo "=============================================="
echo "  [deploy-all] 2/3  Chaos Mesh (Helm + wait)"
echo "=============================================="
bash "${SCRIPT_DIR}/deploy-chaos-mesh.sh"

echo ""
echo "=============================================="
echo "  [deploy-all] 3/3  Boutique (fresh namespace)"
echo "=============================================="
_finalize_ns_if_stuck boutique
bash "${SCRIPT_DIR}/deploy-boutique.sh" --fresh

echo ""
echo "[deploy-all] All steps finished."
echo "  Frontend:    http://localhost:8080"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:     http://localhost:3000  (admin/admin)"
echo "  Chaos UI:    kubectl port-forward -n chaos-mesh svc/chaos-dashboard 2333:2333"
