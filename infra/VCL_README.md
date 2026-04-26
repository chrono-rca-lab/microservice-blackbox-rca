# VCL Deployment Guide

This guide covers two scenarios:
- **A: Deploy your own cluster** — you want your own isolated 2-node k3s setup on VCL
- **B: Join an existing cluster** — a teammate already has a cluster running and you want to run experiments against it

In both cases, `run_experiment.py`, `run_experiment_slo.py`, and `run_batch.py` run entirely from your laptop. The VCL machines just host the Kubernetes cluster.

---

## Prerequisites (everyone)

```bash
brew install kubectl helm          # macOS
pip install -r requirements.txt   # Python deps
mkdir -p /tmp/kind-volumes
```

You must be on the **NCSU VPN** (or campus network) to reach the VCL machines. Install the Cisco AnyConnect VPN client from [go.ncsu.edu/vpn](https://go.ncsu.edu/vpn) and connect before any of the steps below.

---

## Option B — Join an existing cluster (fastest path)

If a teammate already deployed the cluster, you only need:

1. **Get the kubeconfig** — your teammate runs this on their laptop and sends you `vcl-config`:
   ```bash
   cat ~/.kube/vcl-config
   ```
   Save it to `~/.kube/vcl-config` on your machine.

2. **Activate it:**
   ```bash
   export KUBECONFIG=~/.kube/vcl-config
   echo 'export KUBECONFIG=~/.kube/vcl-config' >> ~/.zshrc
   ```

3. **Verify:**
   ```bash
   kubectl get nodes -L role
   # Should show 2 Ready nodes
   kubectl get pods -n boutique
   # Should show ~11 Running pods
   ```

4. **Start port-forwards** (needed on every new terminal session):
   ```bash
   kubectl port-forward -n boutique svc/frontend 8080:80 &
   kubectl port-forward -n monitoring svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &
   kubectl port-forward -n monitoring svc/kube-prom-stack-grafana 3000:80 &
   ```

5. **Run an experiment:**
   ```bash
   python eval/run_experiment.py --fault cpu_hog --service cartservice --duration 120
   ```

> **Important:** only one person should run an experiment at a time. Concurrent experiments
> inject faults into the same cluster and will corrupt each other's metrics windows.
> Coordinate with your teammates before starting a run.

---

## Option A — Deploy your own cluster

### Step 1 — Request VCL machines (manual, ~2 min)

Go to [vcl.ncsu.edu](https://vcl.ncsu.edu) and request **two reservations**:
- Image: **Ubuntu 22.04**
- CPUs: 4+, RAM: 8 GB+
- Note the IP of each machine as `MACHINE1_IP` and `MACHINE2_IP`

---

### Step 2 — Install k3s on Machine 1 (control plane)

SSH into Machine 1 and paste:

```bash
curl -sfL https://get.k3s.io | sh -s - \
  --write-kubeconfig-mode 644 \
  --disable traefik \
  --node-label role=infra-and-upstream \
  --advertise-address MACHINE1_IP \
  --tls-san MACHINE1_IP

# Add kubelet read-only port so Prometheus can scrape metrics
echo -e "kubelet-arg:\n  - \"read-only-port=10255\"" | sudo tee -a /etc/rancher/k3s/config.yaml
sudo systemctl restart k3s

# Open required ports
sudo iptables -I INPUT -p tcp --dport 6443 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 10255 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 9100 -j ACCEPT
sudo iptables -I INPUT -p udp --dport 8472 -j ACCEPT
sudo iptables -I FORWARD -j ACCEPT

# Wait for k3s to be ready, then print join token
sudo systemctl status k3s
sudo cat /var/lib/rancher/k3s/server/node-token
```

Copy the full token (the entire `K10...::server::...` string).

---

### Step 3 — Install k3s on Machine 2 (worker)

SSH into Machine 2 and paste (replace placeholders):

```bash
curl -sfL https://get.k3s.io | \
  K3S_URL=https://MACHINE1_IP:6443 \
  K3S_TOKEN=FULL_TOKEN_FROM_STEP2 \
  sh -s - \
  --node-label role=fault-targets

# Add kubelet read-only port
sudo mkdir -p /etc/rancher/k3s
echo -e "kubelet-arg:\n  - \"read-only-port=10255\"" | sudo tee /etc/rancher/k3s/config.yaml
sudo systemctl restart k3s-agent

# Open required ports
sudo iptables -I INPUT -p tcp --dport 10255 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 9100 -j ACCEPT
sudo iptables -I INPUT -p udp --dport 8472 -j ACCEPT
sudo iptables -I FORWARD -j ACCEPT
```

Press Ctrl+C if the shell hangs after the install — the agent is running as a systemd service in the background. Check with `sudo systemctl status k3s-agent`.

---

### Step 4 — Configure kubectl on your laptop

```bash
# Download and patch kubeconfig
mkdir -p ~/.kube
scp your_unity_id@MACHINE1_IP:/etc/rancher/k3s/k3s.yaml ~/.kube/vcl-config
sed -i.bak "s|https://127.0.0.1:6443|https://MACHINE1_IP:6443|g" ~/.kube/vcl-config
rm ~/.kube/vcl-config.bak

# Activate
export KUBECONFIG=~/.kube/vcl-config
echo 'export KUBECONFIG=~/.kube/vcl-config' >> ~/.zshrc

# Verify — both nodes should show Ready
kubectl get nodes -L role
```

---

### Step 5 — Deploy everything

```bash
bash infra/vcl-setup.sh deploy
```

This runs in ~10 minutes on first boot (image pulls). It:
1. Patches boutique manifests for fault injection
2. Adds node selectors (Machine 1: frontend/checkout/catalog/shipping/redis, Machine 2: cart/currency/email/payment/recommendation/ad)
3. Deploys Online Boutique and waits for all pods Ready
4. Deploys Prometheus + Grafana
5. Deploys Chaos Mesh
6. Starts port-forwards

If the script fails at the pod readiness wait (loadgenerator sometimes gets stuck), run:
```bash
kubectl scale deployment loadgenerator --replicas=0 -n boutique
bash infra/deploy-monitoring.sh --k3s
bash infra/deploy-chaos-mesh.sh --k3s
kubectl port-forward -n boutique svc/frontend 8080:80 &
kubectl port-forward -n monitoring svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &
kubectl port-forward -n monitoring svc/kube-prom-stack-grafana 3000:80 &
```

---

### Step 6 — Verify everything works

```bash
# Frontend responds
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080   # expect 200

# Prometheus has boutique metrics
python -m rca_engine.metrics_client                            # expect metric table
```

In the Prometheus UI at `http://localhost:9090`, run:
```
node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{namespace="boutique"}
```
Should return ~11 time-series. If empty, wait 2 minutes and retry.

---

## Running experiments

```bash
# Basic fixed-duration experiment (~4 min total)
python eval/run_experiment.py --fault cpu_hog --service cartservice --duration 120

# SLO-triggered (fires RCA as soon as latency spikes)
python eval/run_experiment_slo.py --fault net_delay --service currencyservice --duration 120

# Check results
cat experiments/$(ls -t experiments | head -1)/rca_results.json
```

Artifacts land in `experiments/<run_id>/`:
```
ground_truth.json   — injected fault and target
timeline.json       — timestamps, SLO violation metadata
metrics.parquet     — per-service metric matrix
rca_results.json    — ranked root cause output
```

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Frontend | http://localhost:8080 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## Port-forwards die when the terminal closes

Re-run after opening a new terminal:

```bash
export KUBECONFIG=~/.kube/vcl-config
kubectl port-forward -n boutique svc/frontend 8080:80 &
kubectl port-forward -n monitoring svc/kube-prom-stack-kube-prome-prometheus 9090:9090 &
kubectl port-forward -n monitoring svc/kube-prom-stack-grafana 3000:80 &
```

---

## Teardown

VCL machines are automatically reclaimed when your reservation expires. To tear down manually:
```bash
# Delete from your laptop
kubectl delete namespace boutique monitoring chaos-mesh

# Or just end the VCL reservation in the portal — the VMs are destroyed
```
