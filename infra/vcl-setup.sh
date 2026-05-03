#!/usr/bin/env bash
# infra/vcl-setup.sh — Deploy microservice-blackbox-rca on a k3s cluster on VCL
#
# 2-node setup (basic):
#   STEP 1 (Machine 1):  bash infra/vcl-setup.sh control-plane
#   STEP 2 (Machine 2):  bash infra/vcl-setup.sh worker <MACHINE1_IP> <TOKEN>
#   STEP 3 (laptop):     bash infra/vcl-setup.sh kubeconfig <MACHINE1_IP> <SSH_USER>
#   STEP 4 (laptop):     bash infra/vcl-setup.sh deploy
#
# 4-node setup (recommended — isolated monitoring + isolated experiment target):
#   Machine 1 (infra-and-upstream):  frontend, productcatalog, adservice, paymentservice,
#                                    shippingservice, redis-cart
#   Machine 2 (fault-targets):       recommendationservice, checkoutservice, currencyservice,
#                                    cartservice, emailservice
#   Machine 3 (monitoring):          Prometheus, Grafana — no resource contention with services
#   Machine 4 (experiment-target):   single service under test, moved here per experiment
#
#   STEP 1 (Machine 1):  bash infra/vcl-setup.sh control-plane
#   STEP 2 (Machine 2):  bash infra/vcl-setup.sh worker <MACHINE1_IP> <TOKEN>
#   STEP 3 (Machine 3):  bash infra/vcl-setup.sh worker <MACHINE1_IP> <TOKEN> role=monitoring
#   STEP 4 (Machine 4):  bash infra/vcl-setup.sh worker <MACHINE1_IP> <TOKEN> role=experiment-target
#   STEP 5 (laptop):     bash infra/vcl-setup.sh kubeconfig <MACHINE1_IP> <SSH_USER>
#   STEP 6 (laptop):     bash infra/vcl-setup.sh deploy --4node
#
# Other:
#   bash infra/vcl-setup.sh verify   → show pod placement across all nodes
#
# Prerequisites (Machine 1 and Machine 2):
#   - Ubuntu 22.04, 4+ cores, 8 GB+ RAM
#   - Port 6443 open between the two machines (k3s API server)
#   - Internet access for image pulls
#
# Prerequisites (your laptop):
#   - kubectl, helm, python3, scp installed
#   - SSH access to both VCL machines
#   - On the NCSU network or VPN so you can reach port 6443 on Machine 1
#
set -euo pipefail

MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# control-plane — run on Machine 1
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "control-plane" ]]; then
  echo "[vcl] Installing k3s control plane…"
  curl -sfL https://get.k3s.io | sh -s - \
    --write-kubeconfig-mode 644 \
    --disable traefik \
    --node-label role=infra-and-upstream

  echo "[vcl] Waiting for node to be Ready…"
  until kubectl --kubeconfig /etc/rancher/k3s/k3s.yaml get nodes 2>/dev/null | grep -q " Ready"; do
    sleep 2
  done

  echo ""
  echo "========================================="
  echo " Machine 1 setup complete."
  echo ""
  echo " Join token (copy this for step 2):"
  cat /var/lib/rancher/k3s/server/node-token
  echo ""
  echo " Machine 1 IP (use this for steps 2 and 3):"
  hostname -I | awk '{print $1}'
  echo "========================================="

# ---------------------------------------------------------------------------
# worker — run on Machine 2, 3, or 4
# Usage: bash infra/vcl-setup.sh worker <MACHINE1_IP> <TOKEN> [role=<label>]
#   role=fault-targets       → Machine 2 (default)
#   role=monitoring          → Machine 3
#   role=experiment-target   → Machine 4
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "worker" ]]; then
  MACHINE1_IP="${2:?Usage: bash infra/vcl-setup.sh worker <MACHINE1_IP> <TOKEN> [role=LABEL]}"
  TOKEN="${3:?Usage: bash infra/vcl-setup.sh worker <MACHINE1_IP> <TOKEN> [role=LABEL]}"
  NODE_LABEL="${4:-role=fault-targets}"

  echo "[vcl] Joining k3s cluster at https://${MACHINE1_IP}:6443 (label: ${NODE_LABEL}) …"
  curl -sfL https://get.k3s.io | \
    K3S_URL="https://${MACHINE1_IP}:6443" \
    K3S_TOKEN="${TOKEN}" \
    sh -s - \
    --node-label "${NODE_LABEL}"

  echo ""
  echo "========================================="
  echo " Worker joined with label: ${NODE_LABEL}"
  echo " Verify on Machine 1: kubectl get nodes -L role"
  echo "========================================="

# ---------------------------------------------------------------------------
# kubeconfig — run on your laptop
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "kubeconfig" ]]; then
  MACHINE1_IP="${2:?Usage: bash infra/vcl-setup.sh kubeconfig <MACHINE1_IP> <SSH_USER>}"
  SSH_USER="${3:?Usage: bash infra/vcl-setup.sh kubeconfig <MACHINE1_IP> <SSH_USER>}"

  echo "[vcl] Copying kubeconfig from ${SSH_USER}@${MACHINE1_IP} …"
  mkdir -p ~/.kube
  scp "${SSH_USER}@${MACHINE1_IP}:/etc/rancher/k3s/k3s.yaml" ~/.kube/vcl-config

  # Replace loopback address with the real Machine 1 IP — works on macOS and Linux
  sed -i.bak "s|https://127.0.0.1:6443|https://${MACHINE1_IP}:6443|g" ~/.kube/vcl-config
  rm -f ~/.kube/vcl-config.bak

  echo "[vcl] Testing connection (KUBECONFIG=~/.kube/vcl-config) …"
  KUBECONFIG=~/.kube/vcl-config kubectl get nodes
  echo ""
  echo "========================================="
  echo " Kubeconfig ready at ~/.kube/vcl-config"
  echo ""
  echo " To activate in your shell:"
  echo "   export KUBECONFIG=~/.kube/vcl-config"
  echo ""
  echo " To persist across sessions, add that line to ~/.zshrc or ~/.bashrc"
  echo "========================================="

# ---------------------------------------------------------------------------
# deploy — run on your laptop (KUBECONFIG must point to the VCL cluster)
# Flags: --4node  use 4-node layout (pins monitoring to role=monitoring node)
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "deploy" ]]; then
  FOURNODE=false
  for arg in "$@"; do
    [[ "${arg}" == "--4node" ]] && FOURNODE=true
  done
  # Confirm we're talking to the right cluster
  echo "[vcl] Active cluster:"
  kubectl cluster-info | head -1
  echo ""

  PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
  [[ ! -x "${PYTHON}" ]] && PYTHON="python3"

  NAMESPACE="boutique"
  RAW_MANIFEST="${SCRIPT_DIR}/boutique-manifests.yaml"
  PATCHED_MANIFEST="${SCRIPT_DIR}/boutique-manifests-patched.yaml"

  # 1. Patch manifests for fault injection (removes readOnlyRootFilesystem, adds NET_ADMIN, /tmp)
  echo "[vcl] Step 1/5 — Patching manifests for fault injection…"
  "${PYTHON}" "${SCRIPT_DIR}/patch_manifests.py" \
    --input  "${RAW_MANIFEST}" \
    --output "${PATCHED_MANIFEST}"

  # 2. Add nodeSelector so services land on the correct physical machine
  echo "[vcl] Step 2/5 — Adding node selectors for 2-node split…"
  "${PYTHON}" "${SCRIPT_DIR}/node-assignment-patch.py" \
    --input  "${PATCHED_MANIFEST}" \
    --output "${PATCHED_MANIFEST}"

  # 3. Deploy Online Boutique
  echo "[vcl] Step 3/5 — Deploying Online Boutique…"
  kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -n "${NAMESPACE}" -f "${PATCHED_MANIFEST}"

  # Scale loadgenerator to 0 immediately — its init container waits for frontend
  # and can block the readiness wait. We run loadgen.py manually for experiments.
  kubectl scale deployment loadgenerator --replicas=0 -n "${NAMESPACE}" || true

  echo "[vcl] Waiting for all pods to be Ready (timeout 600 s)…"
  if ! kubectl wait --for=condition=ready pod --all -n "${NAMESPACE}" --timeout=600s; then
    echo "[vcl] ERROR: readiness wait timed out. Pod status:" >&2
    kubectl get pods -n "${NAMESPACE}" -o wide >&2
    exit 1
  fi

  # 4. Deploy monitoring (k3s mode disables https for kubelet scrape)
  echo "[vcl] Step 4/5 — Deploying Prometheus + Grafana…"
  if [[ "${FOURNODE}" == true ]]; then
    bash "${SCRIPT_DIR}/deploy-monitoring.sh" --k3s --monitoring-node
  else
    bash "${SCRIPT_DIR}/deploy-monitoring.sh" --k3s
  fi

  # 5. Deploy Chaos Mesh (k3s mode uses the k3s containerd socket)
  echo "[vcl] Step 5/5 — Deploying Chaos Mesh…"
  bash "${SCRIPT_DIR}/deploy-chaos-mesh.sh" --k3s

  # Port-forwards so localhost hits the cluster
  echo "[vcl] Starting port-forwards…"
  kubectl port-forward -n "${NAMESPACE}" svc/frontend 8080:80 &
  kubectl port-forward -n monitoring svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &
  kubectl port-forward -n monitoring svc/kube-prom-stack-grafana 3000:80 &

  echo ""
  echo "========================================="
  echo " Deployment complete!"
  echo ""
  echo "  Frontend:   http://localhost:8080"
  echo "  Prometheus: http://localhost:9090"
  echo "  Grafana:    http://localhost:3000  (admin / admin)"
  echo ""
  echo " Run bash infra/vcl-setup.sh verify to confirm pod placement."
  echo "========================================="

# ---------------------------------------------------------------------------
# verify — run on your laptop
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "verify" ]]; then
  echo "=== Nodes ==="
  kubectl get nodes -L role

  echo ""
  echo "=== Boutique pods and their nodes ==="
  kubectl get pods -n boutique -o wide

  echo ""
  echo "=== Monitoring pods ==="
  kubectl get pods -n monitoring -o wide | head -6

  echo ""
  echo "=== Chaos Mesh pods ==="
  kubectl get pods -n chaos-mesh -o wide

  echo ""
  echo "=== Quick Prometheus check ==="
  echo "Expected: metrics for all boutique pods"
  echo "Run in Prometheus UI (http://localhost:9090):"
  echo "  node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{namespace=\"boutique\"}"

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
else
  echo "Usage:"
  echo "  bash infra/vcl-setup.sh control-plane"
  echo "      Run on Machine 1. Installs k3s control plane, prints join token."
  echo ""
  echo "  bash infra/vcl-setup.sh worker <MACHINE1_IP> <JOIN_TOKEN>"
  echo "      Run on Machine 2. Joins the k3s cluster."
  echo ""
  echo "  bash infra/vcl-setup.sh kubeconfig <MACHINE1_IP> <SSH_USER>"
  echo "      Run on your laptop. Downloads and patches kubeconfig."
  echo ""
  echo "  bash infra/vcl-setup.sh deploy"
  echo "      Run on your laptop. Deploys everything (boutique, monitoring, Chaos Mesh)."
  echo ""
  echo "  bash infra/vcl-setup.sh verify"
  echo "      Run on your laptop. Shows pod placement across both nodes."
fi
