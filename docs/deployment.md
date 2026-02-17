# 📁 Complete AI Cluster Project Files (Continued)

## 📂 `docs/` Directory - Documentation

---

## `docs/deployment.md`

```markdown
# Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Single Machine Deployment](#single-machine-deployment)
5. [Multi-Machine Cluster Deployment](#multi-machine-cluster-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Cloud Deployment](#cloud-deployment)
8. [Hybrid Deployment](#hybrid-deployment)
9. [High Availability Setup](#high-availability-setup)
10. [Monitoring Stack Deployment](#monitoring-stack-deployment)
11. [Backup and Recovery](#backup-and-recovery)
12. [Security Hardening](#security-hardening)
13. [Performance Tuning](#performance-tuning)
14. [Scaling Guide](#scaling-guide)
15. [Upgrade Procedures](#upgrade-procedures)
16. [Troubleshooting](#troubleshooting)

---

## Overview

The AI Cluster can be deployed in various configurations, from a single machine with multiple GPUs to a large-scale distributed cluster across data centers. This guide covers all deployment scenarios with step-by-step instructions.

### Deployment Architecture Options

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Deployment Architecture Options                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Single Machine    Multi-Machine      Kubernetes        Cloud/Hybrid │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  │ Coordinator │   │ Coordinator │   │   Pod 1     │   │  AWS/Azure  │
│  │   +         │   │     on      │   │ Coordinator │   │    +        │
│  │  Workers    │   │  Machine 1  │   └─────────────┘   │  On-Prem    │
│  └─────────────┘   └─────────────┘   ┌─────────────┐   └─────────────┘
│                         │            │   Pod 2     │         │
│                    ┌────┴────┐       │  Worker     │    ┌────┴────┐
│                    │ Worker  │       └─────────────┘    │  VPN/    │
│                    │Machine 2│       ┌─────────────┐    │  Direct  │
│                    └─────────┘       │   Pod N     │    │ Connect  │
│                    ┌─────────┐       │  Worker     │    └─────────┘
│                    │ Worker  │       └─────────────┘
│                    │Machine 3│
│                    └─────────┘
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
| Component | Coordinator | Worker (CPU) | Worker (GPU) |
|-----------|------------|--------------|--------------|
| CPU | 4 cores | 8 cores | 8 cores |
| RAM | 8 GB | 32 GB | 32 GB |
| Storage | 50 GB | 100 GB | 100 GB |
| GPU | N/A | N/A | 1x GPU (8GB+) |
| Network | 1 Gbps | 1 Gbps | 10 Gbps |

#### Recommended Production Setup
| Component | Coordinator | Worker (GPU) |
|-----------|------------|--------------|
| CPU | 8-16 cores | 16-32 cores |
| RAM | 16-32 GB | 64-128 GB |
| Storage | 100 GB SSD | 500 GB NVMe |
| GPU | N/A | 4-8x GPU (16GB+ each) |
| Network | 10 Gbps | 25-100 Gbps RDMA |
| Redundancy | Active-Passive | N+1 |

### Software Requirements

#### Base System
- **OS**: Ubuntu 22.04 LTS (recommended), RHEL 9, or Rocky Linux 9
- **Kernel**: 5.15+ with GPU drivers
- **Docker**: 20.10+ (for containerized deployment)
- **Python**: 3.10+
- **Rust**: 1.70+

#### GPU Drivers

**For AMD GPUs:**
```bash
# Install ROCm
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -a -G render,video $USER
```

**For NVIDIA GPUs:**
```bash
# Install CUDA drivers
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
sudo usermod -a -G video $USER
```

### Network Requirements

| Port | Protocol | Service | Direction |
|------|----------|---------|-----------|
| 8000 | TCP | Coordinator API | Inbound |
| 50051-50055 | TCP | Worker gRPC | Inbound |
| 9090 | TCP | Coordinator Metrics | Inbound |
| 9091-9095 | TCP | Worker Metrics | Inbound |
| 50052 | UDP | Broadcast Discovery | Bidirectional |
| 5353 | UDP | mDNS Discovery | Bidirectional |

### Storage Requirements

```bash
# Create directories with proper permissions
sudo mkdir -p /data/{models,downloads,logs,backups}
sudo chown -R $USER:$USER /data
chmod 755 /data/{models,downloads,logs,backups}

# For high-performance storage (NVMe)
sudo mkfs.ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /data
echo "/dev/nvme0n1 /data ext4 defaults 0 0" | sudo tee -a /etc/fstab
```

---

## Quick Start

### 5-Minute Test Deployment

```bash
# 1. Clone repository
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# 2. Run setup script
./scripts/setup_rocm.sh  # For AMD
# or
./scripts/setup_cuda.sh  # For NVIDIA

# 3. Build and start with Docker Compose
docker-compose up -d

# 4. Check status
curl http://localhost:8000/health

# 5. Load a model
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepseek-7b"}'

# 6. Run inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

---

## Single Machine Deployment

### Architecture
```
┌─────────────────────────────────────┐
│         Single Server               │
├─────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐  │
│  │ Coordinator │   │  Worker 1   │  │
│  │  (Process)  │   │  (Process)  │  │
│  └─────────────┘   └─────────────┘  │
│         │               │            │
│  ┌──────▼───────────────▼──────┐    │
│  │        GPU 0    GPU 1       │    │
│  │        GPU 2    GPU 3       │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### Step-by-Step Deployment

#### 1. **Install Dependencies**

```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Install basic tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    htop \
    nvtop \
    python3-pip \
    python3-venv \
    docker.io \
    docker-compose

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### 2. **Configure GPU Access**

```bash
# For AMD GPUs
./scripts/setup_rocm.sh

# For NVIDIA GPUs
./scripts/setup_cuda.sh

# Verify GPU access
rocm-smi  # AMD
nvidia-smi  # NVIDIA
```

#### 3. **Build the Cluster**

```bash
# Build coordinator
cd coordinator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Build worker
cd worker
cargo build --release --features=hip  # AMD
# or
cargo build --release --features=cuda  # NVIDIA
cd ..

# Create model directory
mkdir -p models
```

#### 4. **Configure for Single Machine**

Edit `config/coordinator.yaml`:
```yaml
discovery:
  method: "static"
  static_workers:
    - "localhost:50051"
    - "localhost:50052"
    - "localhost:50053"
    - "localhost:50054"
```

Edit `config/worker.toml` for each worker (copy to worker1.toml, worker2.toml, etc.):
```toml
[worker]
id = "worker-1"  # Change for each worker

[gpu]
device_ids = [0]  # First worker uses GPU 0

[grpc]
port = 50051  # Different port per worker
```

#### 5. **Start Services**

```bash
# Method 1: Using process manager
# Terminal 1 - Coordinator
cd coordinator && uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Worker 1
cd worker && ./target/release/ai-worker --config config/worker1.toml

# Terminal 3 - Worker 2
cd worker && ./target/release/ai-worker --config config/worker2.toml

# Method 2: Using Docker Compose
docker-compose up -d

# Method 3: Using systemd (production)
sudo cp deployment/ai-coordinator.service /etc/systemd/system/
sudo cp deployment/ai-worker@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-coordinator
sudo systemctl enable ai-worker@1 ai-worker@2
sudo systemctl start ai-coordinator
sudo systemctl start ai-worker@1 ai-worker@2
```

#### 6. **Verify Deployment**

```bash
# Check services
curl http://localhost:8000/health
curl http://localhost:8000/v1/workers

# Check GPU usage
rocm-smi  # AMD
nvidia-smi  # NVIDIA

# Check logs
journalctl -u ai-coordinator -f
journalctl -u ai-worker@1 -f
```

---

## Multi-Machine Cluster Deployment

### Architecture
```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Machine 1       │     │  Machine 2       │     │  Machine 3       │
│  (Coordinator)   │     │  (Worker Pool)   │     │  (Worker Pool)   │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ ┌──────────────┐ │     │ ┌──────────────┐ │     │ ┌──────────────┐ │
│ │ Coordinator  │ │     │ │ Worker 2-1   │ │     │ │ Worker 3-1   │ │
│ │   Process    │ │     │ │   GPU 0-3    │ │     │ │   GPU 0-3    │ │
│ └──────────────┘ │     │ └──────────────┘ │     │ └──────────────┘ │
│                  │     │ ┌──────────────┐ │     │ ┌──────────────┐ │
│ ┌──────────────┐ │     │ │ Worker 2-2   │ │     │ │ Worker 3-2   │ │
│ │ Worker 1-1   │ │     │ │   GPU 4-7    │ │     │ │   GPU 4-7    │ │
│ │   GPU 0-3    │ │     │ └──────────────┘ │     │ └──────────────┘ │
│ └──────────────┘ │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │   10GbE Switch  │
                         └─────────────────┘
```

### Prerequisites for Cluster

#### Network Configuration

```bash
# On all machines, set static IPs
sudo vi /etc/netplan/01-netcfg.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eno1:
      addresses:
        - 192.168.1.10/24  # Different IP per machine
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

```bash
# Apply network config
sudo netplan apply

# Update /etc/hosts on all machines
sudo vi /etc/hosts
```

```
192.168.1.10 coordinator
192.168.1.11 worker1
192.168.1.12 worker2
192.168.1.13 worker3
```

#### Passwordless SSH (for management)

```bash
# Generate SSH key on management machine
ssh-keygen -t rsa -b 4096

# Copy to all nodes
ssh-copy-id user@coordinator
ssh-copy-id user@worker1
ssh-copy-id user@worker2
ssh-copy-id user@worker3
```

#### Shared Storage (Optional but Recommended)

```bash
# Option 1: NFS (simpler)
# On coordinator (NFS server)
sudo apt install nfs-kernel-server
sudo mkdir -p /data/models
sudo chown nobody:nogroup /data/models
echo "/data/models 192.168.1.0/24(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -a

# On workers (NFS clients)
sudo apt install nfs-common
sudo mkdir -p /data/models
echo "coordinator:/data/models /data/models nfs defaults 0 0" | sudo tee -a /etc/fstab
sudo mount -a

# Option 2: CephFS (for production)
# Option 3: MinIO + S3FS
```

### Deployment Steps

#### 1. **Prepare Base Image (All Nodes)**

```bash
# Create deployment script
cat > prepare_node.sh << 'EOF'
#!/bin/bash
set -e

# Install common dependencies
sudo apt update
sudo apt install -y \
    docker.io \
    python3-pip \
    build-essential \
    cmake \
    curl \
    git

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA drivers (if GPU node)
if lspci | grep -i nvidia; then
    ./scripts/setup_cuda.sh
fi

# Install ROCm (if AMD GPU node)
if lspci | grep -i amd; then
    ./scripts/setup_rocm.sh
fi

echo "Node preparation complete"
EOF

chmod +x prepare_node.sh

# Run on all nodes
for node in coordinator worker1 worker2 worker3; do
    scp prepare_node.sh user@$node:~/
    ssh user@$node "./prepare_node.sh"
done
```

#### 2. **Deploy Coordinator Node**

```bash
# On coordinator machine
ssh user@coordinator

# Clone repository
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# Configure coordinator
cat > config/coordinator.yaml << 'EOF'
server:
  host: "0.0.0.0"
  port: 8000

discovery:
  method: "static"
  static_workers:
    - "worker1:50051"
    - "worker1:50052"
    - "worker2:50051"
    - "worker2:50052"
    - "worker3:50051"
    - "worker3:50052"

health:
  check_interval_seconds: 30
  max_failures: 3

models:
  cache_dir: "/data/models"
  auto_load_on_startup: true
EOF

# Build and start coordinator
cd coordinator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Create systemd service
sudo tee /etc/systemd/system/ai-coordinator.service << 'EOF'
[Unit]
Description=AI Cluster Coordinator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-cluster/coordinator
ExecStart=/home/ubuntu/ai-cluster/coordinator/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ai-coordinator
sudo systemctl start ai-coordinator
```

#### 3. **Deploy Worker Nodes**

```bash
# On each worker machine
ssh user@worker1

# Clone repository
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# Build worker
cd worker
cargo build --release --features=hip  # or cuda
cd ..

# Create worker configuration for each GPU
for gpu in 0 1 2 3; do
    cat > config/worker-${gpu}.toml << EOF
[worker]
id = "worker1-gpu${gpu}"
environment = "production"

[grpc]
port = $((50051 + gpu))

[gpu]
device_ids = [${gpu}]
memory_fraction = 0.9
enable_peer_access = true

[model_loader]
cache_dir = "/data/models"

[inference]
max_batch_size = 32
enable_continuous_batching = true
EOF

    # Create systemd service for each GPU worker
    sudo tee /etc/systemd/system/ai-worker@${gpu}.service << EOF
[Unit]
Description=AI Worker GPU ${gpu}
After=network.target
Requires=ai-coordinator.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-cluster
ExecStart=/home/ubuntu/ai-cluster/worker/target/release/ai-worker --config /home/ubuntu/ai-cluster/config/worker-${gpu}.toml
Restart=always
RestartSec=10
Environment="RUST_LOG=info"

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl enable ai-worker@${gpu}
    sudo systemctl start ai-worker@${gpu}
done
```

#### 4. **Set Up Monitoring (Optional)**

```bash
# On coordinator, deploy monitoring stack
cd ai-cluster
docker-compose -f docker/docker-compose.monitoring.yml up -d

# Access Grafana at http://coordinator:3000 (admin/admin)
```

#### 5. **Test Cluster**

```bash
# From any machine
curl http://coordinator:8000/v1/workers

# Should show all 12 workers (3 machines × 4 GPUs)
```

---

## Kubernetes Deployment

### Architecture
```
┌─────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐              │
│  │   Namespace  │    │   Namespace  │              │
│  │  ai-cluster  │    │  monitoring  │              │
│  └──────┬───────┘    └──────┬───────┘              │
│         │                   │                       │
│  ┌──────▼───────┐    ┌──────▼───────┐              │
│  │ Coordinator  │    │  Prometheus  │              │
│  │   Deployment │    │   Operator   │              │
│  │    (3 pods)  │    └──────────────┘              │
│  └──────┬───────┘                                   │
│         │                                           │
│  ┌──────▼───────┐    ┌──────────────┐              │
│  │ Worker Daemon│    │    Grafana   │              │
│  │    Set       │    │  Deployment  │              │
│  │  (per node)  │    └──────────────┘              │
│  └──────────────┘                                   │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐              │
│  │   PVC for    │    │   ConfigMap  │              │
│  │    Models    │    │    & Secrets │              │
│  └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────┘
```

### Prerequisites

```bash
# Kubernetes cluster (1.24+)
kubectl version

# Helm 3
helm version

# NVIDIA device plugin (for GPU nodes)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# AMD device plugin (for AMD GPU nodes)
kubectl create -f https://raw.githubusercontent.com/RadeonOpenCompute/k8s-device-plugin/v1.0.0/amd-gpu.yaml

# Cert-manager (for TLS)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.0/deploy/static/provider/cloud/deploy.yaml
```

### Kubernetes Manifests

#### 1. **Namespace and RBAC**

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-cluster
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ai-cluster-sa
  namespace: ai-cluster
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ai-cluster-role
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "update", "create"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ai-cluster-binding
subjects:
- kind: ServiceAccount
  name: ai-cluster-sa
  namespace: ai-cluster
roleRef:
  kind: ClusterRole
  name: ai-cluster-role
  apiGroup: rbac.authorization.k8s.io
```

#### 2. **ConfigMap and Secrets**

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: coordinator-config
  namespace: ai-cluster
data:
  coordinator.yaml: |
    server:
      host: "0.0.0.0"
      port: 8000
    discovery:
      method: "kubernetes"
      kubernetes:
        namespace: "ai-cluster"
        label_selector: "app=ai-worker"
    models:
      cache_dir: "/data/models"
      config_file: "/app/config/models.toml"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: worker-config
  namespace: ai-cluster
data:
  worker.toml: |
    [worker]
    environment = "production"
    
    [grpc]
    port = 50051
    
    [gpu]
    memory_fraction = 0.9
    
    [model_loader]
    cache_dir = "/data/models"
---
apiVersion: v1
kind: Secret
metadata:
  name: ai-cluster-secrets
  namespace: ai-cluster
type: Opaque
stringData:
  api-keys.txt: |
    sk-1234567890abcdef
    sk-0987654321fedcba
  jwt-secret: "your-secret-key-here"
```

#### 3. **Coordinator Deployment**

```yaml
# k8s/coordinator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator
  namespace: ai-cluster
spec:
  replicas: 2
  selector:
    matchLabels:
      app: coordinator
  template:
    metadata:
      labels:
        app: coordinator
    spec:
      serviceAccountName: ai-cluster-sa
      containers:
      - name: coordinator
        image: ai-coordinator:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: COORDINATOR_HOST
          value: "0.0.0.0"
        - name: COORDINATOR_PORT
          value: "8000"
        - name: DISCOVERY_METHOD
          value: "kubernetes"
        - name: RUST_LOG
          value: "info"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-cluster-secrets
              key: database-url
        volumeMounts:
        - name: config
          mountPath: /app/config/coordinator.yaml
          subPath: coordinator.yaml
        - name: models
          mountPath: /data/models
        - name: secrets
          mountPath: /etc/secrets
          readOnly: true
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: coordinator-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: secrets
        secret:
          secretName: ai-cluster-secrets
---
apiVersion: v1
kind: Service
metadata:
  name: coordinator
  namespace: ai-cluster
spec:
  selector:
    app: coordinator
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

#### 4. **Worker DaemonSet**

```yaml
# k8s/worker.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: worker
  namespace: ai-cluster
spec:
  selector:
    matchLabels:
      app: worker
  template:
    metadata:
      labels:
        app: worker
    spec:
      serviceAccountName: ai-cluster-sa
      containers:
      - name: worker
        image: ai-worker:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 50051
          name: grpc
        - containerPort: 9091
          name: metrics
        env:
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: GRPC_PORT
          value: "50051"
        - name: METRICS_PORT
          value: "9091"
        - name: GPU_IDS
          value: "0"  # Will be overridden for multi-GPU nodes
        - name: RUST_LOG
          value: "info"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "0"  # For NVIDIA
        - name: ROCM_VISIBLE_DEVICES
          value: "0"  # For AMD
        volumeMounts:
        - name: config
          mountPath: /app/config/worker.toml
          subPath: worker.toml
        - name: models
          mountPath: /data/models
        - name: dshm
          mountPath: /dev/shm
        resources:
          requests:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1  # For NVIDIA
            amd.com/gpu: 1     # For AMD
          limits:
            memory: "32Gi"
            cpu: "16"
            nvidia.com/gpu: 1
            amd.com/gpu: 1
        securityContext:
          privileged: true
          capabilities:
            add: ["SYS_ADMIN", "SYS_RAWIO"]
      volumes:
      - name: config
        configMap:
          name: worker-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"
      nodeSelector:
        node-type: gpu  # Only schedule on GPU nodes
---
apiVersion: v1
kind: Service
metadata:
  name: worker-headless
  namespace: ai-cluster
spec:
  clusterIP: None
  selector:
    app: worker
  ports:
  - name: grpc
    port: 50051
```

#### 5. **Persistent Storage**

```yaml
# k8s/storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: ai-cluster
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: nfs-client  # Use appropriate storage class
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
  namespace: ai-cluster
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

#### 6. **Ingress for API Access**

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: coordinator-ingress
  namespace: ai-cluster
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.ai-cluster.example.com
    secretName: ai-cluster-tls
  rules:
  - host: api.ai-cluster.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: coordinator
            port:
              number: 8000
```

#### 7. **Horizontal Pod Autoscaling**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: coordinator-hpa
  namespace: ai-cluster
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: coordinator
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 100
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/coordinator.yaml
kubectl apply -f k8s/worker.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get all -n ai-cluster
kubectl get pods -n ai-cluster -w

# Scale workers (if not using DaemonSet)
kubectl scale deployment worker --replicas=10 -n ai-cluster

# View logs
kubectl logs -n ai-cluster -l app=coordinator
kubectl logs -n ai-cluster -l app=worker
```

### GPU Node Labeling

```bash
# Label GPU nodes
kubectl label nodes gpu-node-1 node-type=gpu
kubectl label nodes gpu-node-2 node-type=gpu

# Check GPU availability
kubectl get nodes -l node-type=gpu
kubectl describe node gpu-node-1 | grep -A5 "Capacity:" | grep nvidia
```

---

## Cloud Deployment

### AWS Deployment

#### Architecture
```
┌─────────────────────────────────────────────────────┐
│                    AWS Cloud                         │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐              │
│  │   VPC        │    │   Route 53   │              │
│  │  10.0.0.0/16 │    │   DNS        │              │
│  └──────┬───────┘    └──────────────┘              │
│         │                                           │
│  ┌──────▼───────┐    ┌──────────────┐              │
│  │  Public Subnet│   │  Private Subnet│             │
│  │  10.0.1.0/24 │   │  10.0.2.0/24 │              │
│  │  - NAT Gateway│   │  - Coordinator│             │
│  │  - Bastion    │   │  - Workers    │             │
│  │  - ALB        │   │  - RDS        │             │
│  └──────────────┘    └──────────────┘              │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐              │
│  │  S3 Bucket   │    │  EFS         │              │
│  │  - Models    │    │  - Shared    │              │
│  │  - Backups   │    │    Storage   │              │
│  └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────┘
```

#### Terraform Configuration

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "ai-cluster-vpc"
  }
}

# Subnets
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "ai-cluster-public"
  }
}

resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "${var.aws_region}a"

  tags = {
    Name = "ai-cluster-private"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "ai-cluster-igw"
  }
}

# NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id

  tags = {
    Name = "ai-cluster-nat"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "ai-cluster-public-rt"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "ai-cluster-private-rt"
  }
}

# Security Groups
resource "aws_security_group" "coordinator" {
  name        = "coordinator-sg"
  description = "Security group for coordinator"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP API"
  }

  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Metrics"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "coordinator-sg"
  }
}

resource "aws_security_group" "worker" {
  name        = "worker-sg"
  description = "Security group for workers"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 50051
    to_port     = 50055
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "gRPC"
  }

  ingress {
    from_port   = 9091
    to_port     = 9095
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Metrics"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "worker-sg"
  }
}

# S3 Bucket for models
resource "aws_s3_bucket" "models" {
  bucket = "ai-cluster-models-${random_id.suffix.hex}"
}

resource "random_id" "suffix" {
  byte_length = 4
}

# EFS for shared storage
resource "aws_efs_file_system" "models" {
  creation_token = "ai-cluster-models"

  tags = {
    Name = "ai-cluster-models"
  }
}

resource "aws_efs_mount_target" "models" {
  file_system_id  = aws_efs_file_system.models.id
  subnet_id       = aws_subnet.private.id
  security_groups = [aws_security_group.worker.id]
}

# RDS for metadata (optional)
resource "aws_db_instance" "metadata" {
  identifier     = "ai-cluster-metadata"
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.t3.medium"
  
  db_name  = "aicluster"
  username = "coordinator"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.coordinator.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  storage_encrypted = true
  storage_type      = "gp3"
  allocated_storage = 100

  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "ai-cluster-final-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  tags = {
    Name = "ai-cluster-metadata"
  }
}

resource "random_password" "db_password" {
  length  = 32
  special = false
}

resource "aws_db_subnet_group" "main" {
  name       = "ai-cluster-subnet-group"
  subnet_ids = [aws_subnet.private.id]

  tags = {
    Name = "ai-cluster-db-subnet-group"
  }
}

# EC2 Instances
resource "aws_instance" "coordinator" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "c5.2xlarge"
  
  subnet_id                   = aws_subnet.private.id
  vpc_security_group_ids      = [aws_security_group.coordinator.id]
  associate_public_ip_address = false

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
  }

  user_data = templatefile("user-data-coordinator.sh", {
    database_url = "postgresql://${aws_db_instance.metadata.username}:${random_password.db_password.result}@${aws_db_instance.metadata.endpoint}/aicluster"
    s3_bucket    = aws_s3_bucket.models.bucket
    efs_id       = aws_efs_file_system.models.id
  })

  tags = {
    Name = "ai-cluster-coordinator"
  }
}

resource "aws_instance" "worker" {
  count = var.worker_count
  
  ami           = data.aws_ami.ubuntu.id
  instance_type = "g4dn.xlarge"  # NVIDIA T4
  
  subnet_id                   = aws_subnet.private.id
  vpc_security_group_ids      = [aws_security_group.worker.id]
  associate_public_ip_address = false

  root_block_device {
    volume_type = "gp3"
    volume_size = 200
    encrypted   = true
  }

  user_data = templatefile("user-data-worker.sh", {
    efs_id    = aws_efs_file_system.models.id
    s3_bucket = aws_s3_bucket.models.bucket
  })

  tags = {
    Name = "ai-cluster-worker-${count.index}"
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

# Application Load Balancer
resource "aws_lb" "api" {
  name               = "ai-cluster-api"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.coordinator.id]
  subnets            = [aws_subnet.public.id]

  tags = {
    Name = "ai-cluster-api-alb"
  }
}

resource "aws_lb_target_group" "api" {
  name     = "ai-cluster-api-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
  }

  tags = {
    Name = "ai-cluster-api-tg"
  }
}

resource "aws_lb_listener" "api" {
  load_balancer_arn = aws_lb.api.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# Auto Scaling for workers
resource "aws_autoscaling_group" "workers" {
  name                = "ai-cluster-workers"
  vpc_zone_identifier = [aws_subnet.private.id]
  min_size            = var.worker_count
  max_size            = var.worker_count * 2
  desired_capacity    = var.worker_count

  launch_template {
    id      = aws_launch_template.worker.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "ai-cluster-worker"
    propagate_at_launch = true
  }
}

resource "aws_launch_template" "worker" {
  name_prefix   = "ai-cluster-worker"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = "g4dn.xlarge"

  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size = 200
      volume_type = "gp3"
      encrypted   = true
    }
  }

  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [aws_security_group.worker.id]
  }

  user_data = base64encode(templatefile("user-data-worker.sh", {
    efs_id    = aws_efs_file_system.models.id
    s3_bucket = aws_s3_bucket.models.bucket
  }))
}

# Outputs
output "coordinator_endpoint" {
  value = "https://${aws_lb.api.dns_name}"
}

output "database_endpoint" {
  value     = aws_db_instance.metadata.endpoint
  sensitive = true
}
```

#### User Data Scripts

```bash
# user-data-coordinator.sh
#!/bin/bash
set -e

# Variables passed from Terraform
DATABASE_URL="${database_url}"
S3_BUCKET="${s3_bucket}"
EFS_ID="${efs_id}"

# Mount EFS
apt update
apt install -y nfs-common
mkdir -p /data/models
mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport ${EFS_ID}.efs.${aws_region}.amazonaws.com:/ /data/models
echo "${EFS_ID}.efs.${aws_region}.amazonaws.com:/ /data/models nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport,_netdev 0 0" >> /etc/fstab

# Install AWS CLI
apt install -y awscli

# Download models from S3
aws s3 sync s3://${S3_BUCKET}/models /data/models

# Clone and setup cluster
git clone https://github.com/caestrada1103/ai-cluster.git /opt/ai-cluster
cd /opt/ai-cluster

# Configure coordinator
cat > config/coordinator.yaml << EOF
server:
  host: "0.0.0.0"
  port: 8000

discovery:
  method: "aws"
  aws:
    region: "${aws_region}"
    auto_scaling_group: "ai-cluster-workers"

database:
  type: "postgres"
  url: "${DATABASE_URL}"

models:
  cache_dir: "/data/models"
EOF

# Setup Python
cd coordinator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service
cat > /etc/systemd/system/ai-coordinator.service << EOF
[Unit]
Description=AI Cluster Coordinator
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ai-cluster/coordinator
ExecStart=/opt/ai-cluster/coordinator/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-coordinator
systemctl start ai-coordinator
```

```bash
# user-data-worker.sh
#!/bin/bash
set -e

# Variables from Terraform
EFS_ID="${efs_id}"
S3_BUCKET="${s3_bucket}"

# Install NVIDIA drivers
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update
apt install -y nvidia-container-toolkit nvidia-driver-535
systemctl restart docker

# Mount EFS
apt install -y nfs-common
mkdir -p /data/models
mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport ${EFS_ID}.efs.${aws_region}.amazonaws.com:/ /data/models
echo "${EFS_ID}.efs.${aws_region}.amazonaws.com:/ /data/models nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport,_netdev 0 0" >> /etc/fstab

# Install AWS CLI
apt install -y awscli

# Clone and setup cluster
git clone https://github.com/caestrada1103/ai-cluster.git /opt/ai-cluster
cd /opt/ai-cluster

# Build worker
cd worker
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
cargo build --release --features=cuda

# Configure worker
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
for ((i=0; i<GPU_COUNT; i++)); do
    cat > /opt/ai-cluster/config/worker-${i}.toml << EOF
[worker]
id = "$(hostname)-gpu${i}"
environment = "production"

[grpc]
port = $((50051 + i))

[gpu]
device_ids = [${i}]
memory_fraction = 0.9

[model_loader]
cache_dir = "/data/models"

[inference]
max_batch_size = 32
EOF

    # Create systemd service for each GPU
    cat > /etc/systemd/system/ai-worker@${i}.service << EOF
[Unit]
Description=AI Worker GPU ${i}
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ai-cluster
ExecStart=/opt/ai-cluster/worker/target/release/ai-worker --config /opt/ai-cluster/config/worker-${i}.toml
Restart=always
User=root
Environment="RUST_LOG=info"

[Install]
WantedBy=multi-user.target
EOF

    systemctl enable ai-worker@${i}
    systemctl start ai-worker@${i}
done
```

---

## High Availability Setup

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-AZ Deployment                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    AZ 1      │    │    AZ 2      │    |    AZ 3      │  │
│  │              │    │              │    │              │  │
│  │ Coordinator  │    │ Coordinator  │    │ Coordinator  │  │
│  │   Primary    │───▶│   Standby    │───▶│   Standby    │  │
│  │              │    │              │    │              │  │
│  │ Worker Pool  │    │ Worker Pool  │    │ Worker Pool  │  │
│  │   GPU 0-7    │    │   GPU 0-7    │    │   GPU 0-7    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │  Global Load    │                      │
│                    │    Balancer     │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### Coordinator HA with Leader Election

```yaml
# k8s/coordinator-ha.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: coordinator
  namespace: ai-cluster
spec:
  serviceName: coordinator-headless
  replicas: 3
  selector:
    matchLabels:
      app: coordinator
  template:
    metadata:
      labels:
        app: coordinator
    spec:
      serviceAccountName: ai-cluster-sa
      containers:
      - name: coordinator
        image: ai-coordinator:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: COORDINATOR_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: COORDINATOR_HOST
          value: "0.0.0.0"
        - name: COORDINATOR_PORT
          value: "8000"
        - name: DISCOVERY_METHOD
          value: "kubernetes"
        - name: HA_ENABLED
          value: "true"
        - name: LEADER_ELECTION
          value: "true"
        - name: REDIS_URL
          value: "redis://redis:6379/0"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
      volumes:
      - name: config
        configMap:
          name: coordinator-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: coordinator
  namespace: ai-cluster
spec:
  selector:
    app: coordinator
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: coordinator-headless
  namespace: ai-cluster
spec:
  clusterIP: None
  selector:
    app: coordinator
  ports:
  - port: 8000
    targetPort: 8000
```

### Worker HA with Pod Distribution

```yaml
# k8s/worker-ha.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker
  namespace: ai-cluster
spec:
  replicas: 12
  selector:
    matchLabels:
      app: worker
  template:
    metadata:
      labels:
        app: worker
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: worker
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: worker
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: worker
              topologyKey: kubernetes.io/hostname
      containers:
      - name: worker
        image: ai-worker:latest
        resources:
          limits:
            nvidia.com/gpu: 1
---
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: worker-pdb
  namespace: ai-cluster
spec:
  minAvailable: 80%
  selector:
    matchLabels:
      app: worker
```

### Database HA

```yaml
# k8s/postgres-ha.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres
  namespace: ai-cluster
spec:
  description: "PostgreSQL cluster for AI Cluster"
  instances: 3
  imageName: ghcr.io/cloudnative-pg/postgresql:15
  primaryUpdateStrategy: unsupervised
  
  storage:
    size: 100Gi
    storageClass: standard
  
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
  
  affinity:
    topologyKey: topology.kubernetes.io/zone
    enablePodAntiAffinity: true
  
  postgresql:
    parameters:
      max_connections: "500"
      shared_buffers: "2GB"
      effective_cache_size: "6GB"
      maintenance_work_mem: "512MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
      work_mem: "32768"
      min_wal_size: "1GB"
      max_wal_size: "4GB"
  
  backup:
    barmanObjectStore:
      destinationPath: s3://ai-cluster-backups/postgres
      s3Credentials:
        accessKeyId:
          name: aws-creds
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: aws-creds
          key: SECRET_ACCESS_KEY
      wal:
        compression: gzip
        maxParallel: 8
    retentionPolicy: "30d"
  
  monitoring:
    enablePodMonitor: true
```

### Redis HA for Caching

```yaml
# k8s/redis-ha.yaml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: redis
  namespace: ai-cluster
spec:
  clusterSize: 3
  kubernetesConfig:
    image: quay.io/opstree/redis:v7.0.5
    imagePullPolicy: Always
    resources:
      requests:
        memory: "4Gi"
        cpu: "2"
      limits:
        memory: "8Gi"
        cpu: "4"
  
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 50Gi
        storageClassName: standard
  
  redisExporter:
    enabled: true
    image: quay.io/opstree/redis-exporter:v1.44.0
  
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: redis-cluster
        topologyKey: kubernetes.io/hostname
```

---

## Monitoring Stack Deployment

### Complete Monitoring Setup

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

rule_files:
  - "alerts.yml"

scrape_configs:
- job_name: 'coordinator'
  kubernetes_sd_configs:
  - role: pod
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    action: keep
    regex: coordinator
  - source_labels: [__address__]
    action: replace
    regex: ([^:]+)(?::\d+)?
    replacement: ${1}:9090
    target_label: __address__

- job_name: 'worker'
  kubernetes_sd_configs:
  - role: pod
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    action: keep
    regex: worker
  - source_labels: [__address__]
    action: replace
    regex: ([^:]+)(?::\d+)?
    replacement: ${1}:9091
    target_label: __address__
```

```yaml
# monitoring/grafana-datasources.yaml
apiVersion: 1

datasources:
- name: Prometheus
  type: prometheus
  access: proxy
  url: http://prometheus:9090
  isDefault: true
  editable: false

- name: Loki
  type: loki
  access: proxy
  url: http://loki:3100
  editable: false

- name: Tempo
  type: tempo
  access: proxy
  url: http://tempo:3200
  editable: false
```

```yaml
# monitoring/grafana-dashboards.yaml
apiVersion: 1

providers:
- name: 'AI Cluster'
  orgId: 1
  folder: ''
  type: file
  disableDeletion: false
  editable: true
  options:
    path: /var/lib/grafana/dashboards
```

### Deploy Monitoring Stack

```bash
# Using Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Create namespace
kubectl create namespace monitoring

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.enabled=true \
  --set prometheus.prometheusSpec.scrapeInterval=15s \
  --set alertmanager.enabled=true

# Install Loki for logs
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set grafana.enabled=false \
  --set prometheus.enabled=false \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=50Gi

# Install Tempo for traces
helm install tempo grafana/tempo \
  --namespace monitoring \
  --set tempo.receivers.jaeger.protocols.grpc=true

# Install AI Cluster specific dashboards
kubectl create configmap ai-cluster-dashboards \
  --namespace monitoring \
  --from-file=monitoring/dashboards/

# Configure Grafana to load dashboards
kubectl patch configmap grafana-dashboards-config \
  --namespace monitoring \
  --patch "$(cat monitoring/grafana-dashboards.yaml)"
```

---

## Backup and Recovery

### Automated Backup Script

```bash
# scripts/backup.sh
#!/bin/bash
set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p ${BACKUP_DIR}/{config,models,database,logs}

# Backup configurations
echo "Backing up configurations..."
tar -czf ${BACKUP_DIR}/config/config_${DATE}.tar.gz /opt/ai-cluster/config/

# Backup models (excluding large files if needed)
echo "Backing up models..."
if [ -d "/data/models" ]; then
    tar -czf ${BACKUP_DIR}/models/models_${DATE}.tar.gz /data/models/
fi

# Backup database
echo "Backing up database..."
if command -v pg_dump &> /dev/null; then
    PGPASSWORD=${DB_PASSWORD} pg_dump -h ${DB_HOST} -U ${DB_USER} ${DB_NAME} > \
        ${BACKUP_DIR}/database/db_${DATE}.sql
    gzip ${BACKUP_DIR}/database/db_${DATE}.sql
fi

# Backup logs
echo "Backing up logs..."
tar -czf ${BACKUP_DIR}/logs/logs_${DATE}.tar.gz /var/log/ai-cluster/

# Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 sync ${BACKUP_DIR}/ s3://${S3_BUCKET}/backups/${DATE}/
fi

# Clean old backups
find ${BACKUP_DIR} -type f -mtime +${RETENTION_DAYS} -delete

echo "Backup completed: ${DATE}"
```

### Recovery Procedure

```bash
# scripts/restore.sh
#!/bin/bash
set -e

RESTORE_DATE=$1
BACKUP_DIR="/backups"
S3_BUCKET=${S3_BUCKET:-""}

if [ -z "$RESTORE_DATE" ]; then
    echo "Usage: $0 <date>"
    echo "Available backups:"
    ls ${BACKUP_DIR}/config/
    exit 1
fi

# Download from S3 if needed
if [ -n "$S3_BUCKET" ]; then
    echo "Downloading from S3..."
    aws s3 sync s3://${S3_BUCKET}/backups/${RESTORE_DATE}/ ${BACKUP_DIR}/${RESTORE_DATE}/
    BACKUP_DIR="${BACKUP_DIR}/${RESTORE_DATE}"
fi

# Stop services
echo "Stopping services..."
sudo systemctl stop ai-coordinator
sudo systemctl stop ai-worker@*

# Restore configurations
echo "Restoring configurations..."
tar -xzf ${BACKUP_DIR}/config/config_${RESTORE_DATE}.tar.gz -C /
sudo systemctl daemon-reload

# Restore database
echo "Restoring database..."
if [ -f "${BACKUP_DIR}/database/db_${RESTORE_DATE}.sql.gz" ]; then
    gunzip -c ${BACKUP_DIR}/database/db_${RESTORE_DATE}.sql.gz | \
        PGPASSWORD=${DB_PASSWORD} psql -h ${DB_HOST} -U ${DB_USER} ${DB_NAME}
fi

# Restore models (optional)
read -p "Restore models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Restoring models..."
    tar -xzf ${BACKUP_DIR}/models/models_${RESTORE_DATE}.tar.gz -C /
fi

# Start services
echo "Starting services..."
sudo systemctl start ai-coordinator
sudo systemctl start ai-worker@*

echo "Restore completed from ${RESTORE_DATE}"
```

### Disaster Recovery Plan

```yaml
# disaster-recovery.yaml
plan_name: AI Cluster Disaster Recovery
version: 1.0
last_updated: 2024-01-15

recovery_objectives:
  rpo: 1 hour  # Maximum data loss
  rto: 4 hours  # Maximum downtime

recovery_scenarios:
  single_worker_failure:
    action: auto_restart
    rto: 5 minutes
    steps:
      - Detect worker failure via health checks
      - Automatically restart worker process
      - Redistribute load to remaining workers
      - Notify operations team

  multiple_worker_failure:
    action: scale_up
    rto: 15 minutes
    steps:
      - Detect multiple failures
      - Trigger auto-scaling to replace failed nodes
      - Load models on new workers
      - Update load balancer
      - Notify operations team

  coordinator_failure:
    action: failover
    rto: 2 minutes
    steps:
      - Detect coordinator failure
      - Promote standby coordinator to primary
      - Update DNS records
      - Reconnect workers
      - Resume request processing

  database_failure:
    action: restore_from_backup
    rto: 1 hour
    steps:
      - Detect database corruption
      - Stop all services
      - Restore from latest backup
      - Verify data integrity
      - Restart services
      - Reprocess any failed requests

  full_region_failure:
    action: cross_region_failover
    rto: 4 hours
    steps:
      - Activate disaster recovery plan
      - Spin up infrastructure in secondary region
      - Restore from cross-region backups
      - Update DNS to point to new region
      - Verify all services operational
      - Notify all stakeholders

backup_schedule:
  config_backup: hourly
  database_backup: every 15 minutes (WAL)
  model_backup: daily
  full_backup: weekly

recovery_testing:
  schedule: quarterly
  last_test: 2024-01-01
  next_test: 2024-04-01
  test_procedure: |
    1. Restore backup to test environment
    2. Verify data integrity
    3. Run smoke tests
    4. Measure recovery time
    5. Document results
```

---

## Security Hardening

### Network Security

```bash
# firewall-rules.sh
#!/bin/bash
# Apply firewall rules for AI Cluster nodes

# For coordinator node
setup_coordinator_firewall() {
    # Default deny
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow 22/tcp comment 'SSH'
    
    # Allow API access
    ufw allow 8000/tcp comment 'Coordinator API'
    
    # Allow metrics
    ufw allow 9090/tcp comment 'Metrics'
    
    # Allow internal cluster communication
    ufw allow from 10.0.0.0/8 to any port 50051 proto tcp comment 'Worker gRPC'
    ufw allow from 10.0.0.0/8 to any port 50052 proto udp comment 'Worker Discovery'
    
    # Enable firewall
    ufw --force enable
}

# For worker node
setup_worker_firewall() {
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow 22/tcp comment 'SSH'
    
    # Allow gRPC from coordinator only
    ufw allow from 10.0.1.0/24 to any port 50051 proto tcp comment 'gRPC'
    
    # Allow metrics
    ufw allow from 10.0.1.0/24 to any port 9091 proto tcp comment 'Metrics'
    
    # Enable firewall
    ufw --force enable
}
```

### TLS Configuration

```yaml
# config/tls-config.yaml
tls:
  enabled: true
  min_version: "1.3"
  cipher_suites:
    - "TLS_AES_128_GCM_SHA256"
    - "TLS_AES_256_GCM_SHA384"
    - "TLS_CHACHA20_POLY1305_SHA256"
  
  certificates:
    coordinator:
      cert_file: "/etc/ai-cluster/certs/coordinator.crt"
      key_file: "/etc/ai-cluster/certs/coordinator.key"
      ca_file: "/etc/ai-cluster/certs/ca.crt"
    
    worker:
      cert_file: "/etc/ai-cluster/certs/worker.crt"
      key_file: "/etc/ai-cluster/certs/worker.key"
      ca_file: "/etc/ai-cluster/certs/ca.crt"
  
  mutual_tls:
    enabled: true
    client_auth: "require_and_verify_client_cert"
```

### Secrets Management with Vault

```hcl
# vault/policies/ai-cluster.hcl
path "secret/data/ai-cluster/*" {
  capabilities = ["read", "list"]
}

path "secret/metadata/ai-cluster/*" {
  capabilities = ["list"]
}

path "transit/encrypt/ai-cluster-*" {
  capabilities = ["create", "update"]
}

path "transit/decrypt/ai-cluster-*" {
  capabilities = ["create", "update"]
}
```

```bash
# scripts/vault-setup.sh
#!/bin/bash
# Initialize Vault for AI Cluster

# Enable KV secrets engine
vault secrets enable -version=2 kv

# Store secrets
vault kv put secret/ai-cluster/database \
  host="postgres.cluster.local" \
  port="5432" \
  username="coordinator" \
  password="$(openssl rand -base64 32)"

vault kv put secret/ai-cluster/api-keys \
  admin="$(openssl rand -base64 32)" \
  user1="$(openssl rand -base64 32)" \
  user2="$(openssl rand -base64 32)"

vault kv put secret/ai-cluster/huggingface \
  token="hf_$(openssl rand -hex 20)"

# Enable transit engine for encryption
vault secrets enable transit
vault write -f transit/keys/ai-cluster-models

# Create policy
vault policy write ai-cluster vault/policies/ai-cluster.hcl

# Create token for coordinator
vault token create -policy=ai-cluster -period=24h
```

### Security Context for Kubernetes

```yaml
# k8s/security-context.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker
  namespace: ai-cluster
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        supplementalGroups: [44, 107]  # video, render groups for GPU
      
      containers:
      - name: worker
        securityContext:
          allowPrivilegeEscalation: false
          privileged: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop: ["ALL"]
            add: ["SYS_NICE"]  # For GPU priority
          
          seccompProfile:
            type: RuntimeDefault
          
          seLinuxOptions:
            level: "s0:c123,c456"
```

### Audit Logging

```yaml
# config/audit.yaml
audit:
  enabled: true
  log_path: "/var/log/ai-cluster/audit.log"
  max_size_mb: 100
  max_backups: 10
  max_age_days: 30
  compress: true

  events:
    authentication:
      - login_success
      - login_failure
      - token_refresh
      - logout
    
    authorization:
      - permission_denied
      - role_change
    
    model_management:
      - model_load
      - model_unload
      - model_update
    
    inference:
      - request_start
      - request_complete
      - request_error
    
    configuration:
      - config_change
      - config_reload
    
    system:
      - service_start
      - service_stop
      - service_restart
      - worker_join
      - worker_leave

  format: "json"
  fields:
    timestamp: true
    user_id: true
    request_id: true
    client_ip: true
    event_type: true
    resource: true
    action: true
    status: true
    message: true
    metadata: true
```

---

## Performance Tuning

### System-Level Tuning

```bash
# scripts/tune-system.sh
#!/bin/bash
# System performance tuning for AI Cluster

# Increase system limits
cat >> /etc/security/limits.conf << EOF
* soft nofile 1048576
* hard nofile 1048576
* soft nproc unlimited
* hard nproc unlimited
* soft memlock unlimited
* hard memlock unlimited
EOF

# Network tuning
cat >> /etc/sysctl.conf << EOF
# Network performance
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 30
vm.dirty_background_ratio = 5
vm.max_map_count = 262144

# Kernel scheduling
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0
EOF

sysctl -p

# CPU governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu
done

# Transparent hugepages
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# GPU persistence mode (NVIDIA)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -pm 1
    nvidia-smi -ac 5001,1590  # Max clocks
fi

# GPU compute mode (AMD)
if command -v rocm-smi &> /dev/null; then
    rocm-smi --setperflevel 3  # High performance
    rocm-smi --setpoweroverdriveenable 1
fi
```

### Coordinator Performance Tuning

```yaml
# config/coordinator-performance.yaml
server:
  workers: 8  # Number of worker processes
  backlog: 2048  # Connection backlog
  keep_alive: 65  # Keep-alive timeout
  
performance:
  # Connection pooling
  connection_pool:
    max_size: 1000
    idle_timeout: 60
  
  # Request queuing
  queue:
    max_size: 10000
    priority_levels: 5
    aging_factor: 1.5
  
  # Batching
  batching:
    max_batch_size: 64
    batch_timeout_ms: 25
    max_concurrent_batches: 4
  
  # Caching
  cache:
    response_cache_size: 10000
    response_cache_ttl: 300
    model_info_cache_ttl: 60
    worker_status_cache_ttl: 5
  
  # Rate limiting
  rate_limiting:
    algorithm: "token_bucket"
    refill_rate: 100  # tokens per second
    bucket_size: 1000
  
  # Circuit breakers
  circuit_breakers:
    per_worker:
      failure_threshold: 5
      recovery_timeout: 30
      half_open_requests: 3
    
    per_model:
      failure_threshold: 10
      recovery_timeout: 60
      half_open_requests: 5
```

### Worker Performance Tuning

```toml
# config/worker-performance.toml
[performance]
# Thread pool
thread_pool_size = 32
max_blocking_threads = 16

# Memory
memory_pool_size_gb = 4
prefetch_size_gb = 2
pin_memory = true

# GPU streams
num_compute_streams = 4
num_transfer_streams = 2
stream_priority = 0

[inference]
# Continuous batching
max_batch_size = 64
batch_timeout_ms = 20
max_queued_requests = 2000

# KV cache
kv_cache_size_gb = 16
cache_block_size = 32
enable_prefix_caching = true
enable_compression = true
compression_level = 1

# Speculative decoding
speculative_decoding = true
draft_model_size = "tiny"
num_speculative_tokens = 7
verify_probability = 0.9

[parallelism]
# Pipeline settings
pipeline_num_microbatches = 8
pipeline_interleaved = true

# Tensor parallelism
tensor_parallel_all_gather = true
enable_sequence_parallel = true

[communication]
# NCCL tuning
nccl_min_ctas = 2
nccl_max_ctas = 64
nccl_cga_cluster_size = 4
enable_nvlink = true
enable_infiniband = true

# RCCL tuning
rccl_max_parallel_transfers = 8
enable_xgmi = true
```

### NUMA Optimization

```bash
# scripts/numa-optimize.sh
#!/bin/bash
# NUMA-aware process placement

# Get GPU topology
GPU_TOPOLOGY=$(nvidia-smi topo -m)

# For each GPU, determine closest CPUs
for gpu in $(seq 0 $(nvidia-smi --list-gpus | wc -l)); do
    NUMA_NODE=$(nvidia-smi topo -m | grep "GPU$gpu" | grep -oP "NODE\s+\K\d+")
    
    # Bind worker to specific NUMA node
    cat > /etc/systemd/system/ai-worker@${gpu}.service.d/override.conf << EOF
[Service]
CPUAffinity=$(numactl --hardware | grep "node $NUMA_NODE cpus" | cut -d: -f2)
NUMAMask=${NUMA_NODE}
EOF
    
    # Set environment for CUDA
    echo "CUDA_VISIBLE_DEVICES=${gpu}" >> /etc/systemd/system/ai-worker@${gpu}.service.d/override.conf
done

# For coordinator (no GPU, spread across all NUMA nodes)
cat > /etc/systemd/system/ai-coordinator.service.d/override.conf << EOF
[Service]
CPUAffinity=0-$(nproc --all)
NUMAPolicy=interleave
EOF
```

---

## Scaling Guide

### Horizontal Scaling (More Workers)

```bash
# scripts/scale-horizontal.sh
#!/bin/bash
# Add more worker nodes to cluster

NEW_WORKER_COUNT=$1
CURRENT_WORKERS=$(kubectl get deployment worker -n ai-cluster -o jsonpath='{.spec.replicas}')

if [ -z "$NEW_WORKER_COUNT" ]; then
    echo "Usage: $0 <new_worker_count>"
    echo "Current workers: $CURRENT_WORKERS"
    exit 1
fi

# Scale workers
echo "Scaling workers from $CURRENT_WORKERS to $NEW_WORKER_COUNT..."

# Kubernetes deployment
kubectl scale deployment worker --replicas=$NEW_WORKER_COUNT -n ai-cluster

# Docker Compose
# docker-compose up -d --scale worker=$NEW_WORKER_COUNT

# Wait for workers to be ready
echo "Waiting for workers to be ready..."
kubectl wait --for=condition=ready pod -l app=worker -n ai-cluster --timeout=300s

# Check cluster status
curl -s http://coordinator:8000/v1/workers | jq '. | length'

echo "Scaling complete. New worker count: $NEW_WORKER_COUNT"
```

### Vertical Scaling (More GPUs per Node)

```yaml
# k8s/worker-multi-gpu.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: worker
  namespace: ai-cluster
spec:
  template:
    spec:
      containers:
      - name: worker
        env:
        - name: GPU_IDS
          value: "0,1,2,3,4,5,6,7"  # Use all GPUs
        resources:
          requests:
            nvidia.com/gpu: 8
          limits:
            nvidia.com/gpu: 8
        volumeMounts:
        - name: nvidia-visible-devices
          mountPath: /usr/local/nvidia
      volumes:
      - name: nvidia-visible-devices
        hostPath:
          path: /usr/local/nvidia
```

### Auto-scaling Configuration

```yaml
# k8s/hpa-advanced.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-hpa
  namespace: ai-cluster
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worker
  minReplicas: 2
  maxReplicas: 20
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 50
  - type: Object
    object:
      metric:
        name: queue_size
      describedObject:
        apiVersion: v1
        kind: Service
        name: coordinator
      target:
        type: Value
        averageValue: 100
```

### Cluster Auto-Scaler Configuration (AWS)

```yaml
# terraform/cluster-autoscaler.tf
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  
  scaling_config {
    desired_size = 2
    max_size     = 20
    min_size     = 1
  }
  
  instance_types = ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge"]
  
  tags = {
    "k8s.io/cluster-autoscaler/enabled" = "true"
    "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
  }
}

# Cluster Autoscaler deployment
resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  namespace  = "kube-system"
  
  set {
    name  = "autoDiscovery.clusterName"
    value = var.cluster_name
  }
  
  set {
    name  = "awsRegion"
    value = var.aws_region
  }
  
  set {
    name  = "rbac.create"
    value = "true"
  }
  
  set {
    name  = "scaleDownEnabled"
    value = "true"
  }
  
  set {
    name  = "maxNodeProvisionTime"
    value = "15m"
  }
}
```

---

## Upgrade Procedures

### Rolling Upgrade

```bash
# scripts/rolling-upgrade.sh
#!/bin/bash
# Perform rolling upgrade of cluster components

set -e

COMPONENT=$1
VERSION=$2

if [ -z "$COMPONENT" ] || [ -z "$VERSION" ]; then
    echo "Usage: $0 <coordinator|worker> <version>"
    exit 1
fi

echo "Starting rolling upgrade of $COMPONENT to version $VERSION"

case $COMPONENT in
    coordinator)
        # Update coordinator image
        kubectl set image deployment/coordinator \
            coordinator=ai-coordinator:${VERSION} -n ai-cluster \
            --record
        
        # Rollout status
        kubectl rollout status deployment/coordinator -n ai-cluster
        
        # Verify
        curl -s http://coordinator:8000/version | grep $VERSION
        ;;
    
    worker)
        # Update worker image with rolling update strategy
        kubectl set image deployment/worker \
            worker=ai-worker:${VERSION} -n ai-cluster \
            --record
        
        # Monitor rollout
        kubectl rollout status deployment/worker -n ai-cluster
        
        # Check worker versions
        kubectl get pods -n ai-cluster -l app=worker \
            -o jsonpath='{.items[*].spec.containers[*].image}'
        ;;
    
    *)
        echo "Unknown component: $COMPONENT"
        exit 1
        ;;
esac

echo "Upgrade completed successfully"
```

### Blue-Green Deployment

```yaml
# k8s/blue-green.yaml
apiVersion: v1
kind: Service
metadata:
  name: coordinator
  namespace: ai-cluster
spec:
  selector:
    app: coordinator
    version: blue  # Current active version
  ports:
  - port: 8000
    targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator-blue
  namespace: ai-cluster
spec:
  replicas: 3
  selector:
    matchLabels:
      app: coordinator
      version: blue
  template:
    metadata:
      labels:
        app: coordinator
        version: blue
    spec:
      containers:
      - name: coordinator
        image: ai-coordinator:1.0.0
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator-green
  namespace: ai-cluster
spec:
  replicas: 3
  selector:
    matchLabels:
      app: coordinator
      version: green
  template:
    metadata:
      labels:
        app: coordinator
        version: green
    spec:
      containers:
      - name: coordinator
        image: ai-coordinator:1.1.0  # New version
```

```bash
# scripts/blue-green-switch.sh
#!/bin/bash
# Switch traffic from blue to green

# Deploy new version (green)
kubectl apply -f k8s/coordinator-green.yaml

# Wait for green to be ready
kubectl wait --for=condition=ready pod -l version=green -n ai-cluster --timeout=300s

# Test green deployment
GREEN_POD=$(kubectl get pod -l version=green -n ai-cluster -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward pod/$GREEN_POD 8001:8000 -n ai-cluster &
sleep 2
curl http://localhost:8001/health

# Switch service to green
kubectl patch service coordinator -n ai-cluster -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for errors
sleep 60

# If successful, delete blue
kubectl delete deployment coordinator-blue -n ai-cluster
```

### Canary Deployment

```yaml
# k8s/canary.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator-canary
  namespace: ai-cluster
spec:
  replicas: 1  # 10% of total (assuming 9 blue replicas)
  selector:
    matchLabels:
      app: coordinator
      track: canary
  template:
    metadata:
      labels:
        app: coordinator
        track: canary
    spec:
      containers:
      - name: coordinator
        image: ai-coordinator:1.1.0  # New version
---
apiVersion: v1
kind: Service
metadata:
  name: coordinator
  namespace: ai-cluster
spec:
  selector:
    app: coordinator
  ports:
  - port: 8000
    targetPort: 8000
```

```bash
# scripts/canary-deploy.sh
#!/bin/bash
# Canary deployment with traffic splitting

# Deploy canary
kubectl apply -f k8s/canary.yaml

# Monitor canary metrics
kubectl logs -f -l track=canary -n ai-cluster &

# Check error rate
while true; do
    ERROR_RATE=$(curl -s http://coordinator:9090/metrics | grep "http_requests_total{status=\"5xx\"}" | awk '{print $2}')
    if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
        echo "Error rate too high! Rolling back..."
        kubectl delete deployment coordinator-canary -n ai-cluster
        exit 1
    fi
    
    # Gradually increase canary replicas
    CURRENT=$(kubectl get deployment coordinator-canary -n ai-cluster -o jsonpath='{.spec.replicas}')
    if [ $CURRENT -lt 5 ]; then
        kubectl scale deployment coordinator-canary --replicas=$((CURRENT + 1)) -n ai-cluster
    fi
    
    sleep 300  # Wait 5 minutes between increments
done
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **GPU Not Detected**

```bash
# Check GPU visibility
rocm-smi  # AMD
nvidia-smi  # NVIDIA

# Check drivers
lsmod | grep -E "amdgpu|nvidia"
dmesg | grep -E "amdgpu|nvidia"

# Check permissions
ls -la /dev/dri/*  # AMD
ls -la /dev/nvidia*  # NVIDIA

# Fix permissions
sudo chmod 666 /dev/dri/renderD*
sudo usermod -a -G video,render $USER
```

#### 2. **Worker Connection Failed**

```bash
# Check network connectivity
ping worker-1
telnet worker-1 50051

# Check firewall rules
sudo ufw status
sudo iptables -L -n | grep 50051

# Check worker logs
journalctl -u ai-worker@0 -f

# Verify coordinator discovery
curl http://coordinator:8000/v1/workers
```

#### 3. **Model Load Failed**

```bash
# Check model files
ls -la /data/models/deepseek-7b/

# Verify checksums
md5sum /data/models/deepseek-7b/model.safetensors

# Check disk space
df -h /data

# Check GPU memory
rocm-smi --showmeminfo vram  # AMD
nvidia-smi  # NVIDIA

# Check worker logs for specific error
grep "load model" /var/log/ai-cluster/worker.log
```

#### 4. **High Latency**

```bash
# Check GPU utilization
rocm-smi  # AMD
nvidia-smi  # NVIDIA

# Check network latency
ping -c 10 worker-1

# Check batch sizes
curl http://coordinator:9090/metrics | grep batch_size

# Enable debug logging temporarily
export RUST_LOG=debug
systemctl restart ai-worker@0

# Profile with perf
perf record -F 99 -p $(pgrep ai-worker) -g -- sleep 30
perf report
```

#### 5. **Out of Memory**

```bash
# Check memory usage
free -h
rocm-smi --showmeminfo vram  # AMD
nvidia-smi  # NVIDIA

# Check for memory leaks
while true; do
    rocm-smi --showmeminfo vram | tee -a memory.log
    sleep 60
done

# Reduce model memory usage
# - Use quantization (int8/int4)
# - Reduce batch size
# - Enable KV cache compression
# - Unload unused models

# Restart worker to clear memory
sudo systemctl restart ai-worker@0
```

### Diagnostic Commands

```bash
# scripts/diagnose.sh
#!/bin/bash
# Comprehensive diagnostic script

echo "=== AI Cluster Diagnostic Report ==="
echo "Date: $(date)"
echo

# System info
echo "=== System Information ==="
uname -a
lsb_release -a
echo

# CPU info
echo "=== CPU Information ==="
lscpu | grep "Model name\|CPU(s)\|Thread"
echo

# Memory info
echo "=== Memory Information ==="
free -h
echo

# GPU info
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu --format=csv
elif command -v rocm-smi &> /dev/null; then
    rocm-smi --showallinfo
fi
echo

# Disk info
echo "=== Disk Information ==="
df -h /data
echo

# Network info
echo "=== Network Information ==="
ip addr show
ss -tlnp | grep -E "8000|50051|9090"
echo

# Service status
echo "=== Service Status ==="
systemctl status ai-coordinator --no-pager
systemctl status ai-worker@* --no-pager
echo

# Docker status (if used)
echo "=== Docker Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

# Coordinator API
echo "=== Coordinator Health ==="
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/v1/workers | jq '. | length'
echo

# Logs (last 50 lines)
echo "=== Recent Logs ==="
tail -50 /var/log/ai-cluster/coordinator.log
tail -50 /var/log/ai-cluster/worker.log
echo

echo "=== Diagnostic Complete ==="
```

### Performance Profiling

```bash
# scripts/profile.sh
#!/bin/bash
# Performance profiling for AI Cluster

PROFILE_DIR="/tmp/profiles"
mkdir -p $PROFILE_DIR

# CPU profiling
profile_cpu() {
    local pid=$1
    local duration=${2:-60}
    
    echo "Profiling CPU for PID $pid for ${duration}s..."
    perf record -F 99 -p $pid -g -- sleep $duration
    perf report -i perf.data > $PROFILE_DIR/cpu_${pid}_$(date +%Y%m%d_%H%M%S).txt
    rm perf.data
}

# Memory profiling
profile_memory() {
    local pid=$1
    
    echo "Profiling memory for PID $pid..."
    valgrind --tool=massif --time-unit=ms --massif-out-file=$PROFILE_DIR/memory_${pid}_$(date +%Y%m%d_%H%M%S).out \
        /proc/$pid/exe
}

# GPU profiling (NVIDIA)
profile_gpu_nvidia() {
    local duration=${1:-60}
    
    echo "Profiling NVIDIA GPU for ${duration}s..."
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,温度.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > \
        $PROFILE_DIR/gpu_nvidia_$(date +%Y%m%d_%H%M%S).csv &
    sleep $duration
    kill $!
}

# GPU profiling (AMD)
profile_gpu_amd() {
    local duration=${1:-60}
    
    echo "Profiling AMD GPU for ${duration}s..."
    rocm-smi --showallinfo --loglevel 1 --interval 1 > \
        $PROFILE_DIR/gpu_amd_$(date +%Y%m%d_%H%M%S).log &
    sleep $duration
    kill $!
}

# Network profiling
profile_network() {
    local interface=${1:-eth0}
    local duration=${2:-60}
    
    echo "Profiling network on $interface for ${duration}s..."
    tcpdump -i $interface -w $PROFILE_DIR/network_$(date +%Y%m%d_%H%M%S).pcap &
    sleep $duration
    kill $!
}

# Main
case $1 in
    cpu)
        profile_cpu $2 $3
        ;;
    memory)
        profile_memory $2
        ;;
    gpu)
        if command -v nvidia-smi &> /dev/null; then
            profile_gpu_nvidia $2
        elif command -v rocm-smi &> /dev/null; then
            profile_gpu_amd $2
        fi
        ;;
    network)
        profile_network $2 $3
        ;;
    all)
        profile_cpu $(pgrep ai-worker) 30 &
        profile_gpu_nvidia 30 &
        profile_network eth0 30 &
        wait
        ;;
    *)
        echo "Usage: $0 {cpu|memory|gpu|network|all} [pid] [duration]"
        exit 1
        ;;
esac

echo "Profiling complete. Results in $PROFILE_DIR"
```

---

## Reference

### Quick Reference Cards

#### Deployment Commands

| Action | Command |
|--------|---------|
| Start cluster | `docker-compose up -d` |
| Stop cluster | `docker-compose down` |
| Scale workers | `docker-compose up -d --scale worker=10` |
| View logs | `docker-compose logs -f` |
| Check status | `curl localhost:8000/health` |
| Load model | `curl -X POST localhost:8000/v1/models/load -d '{"model_name":"deepseek-7b"}'` |
| Run inference | `curl -X POST localhost:8000/v1/completions -d '{"model":"deepseek-7b","prompt":"Hello"}'` |

#### Kubernetes Commands

| Action | Command |
|--------|---------|
| Deploy | `kubectl apply -f k8s/` |
| Scale | `kubectl scale deployment worker --replicas=10` |
| Update | `kubectl set image deployment/worker worker=ai-worker:1.1.0` |
| Rollback | `kubectl rollout undo deployment/worker` |
| Logs | `kubectl logs -l app=worker` |
| Port forward | `kubectl port-forward service/coordinator 8000:8000` |

#### Troubleshooting Commands

| Issue | Command |
|-------|---------|
| Check GPU | `nvidia-smi` or `rocm-smi` |
| Check logs | `journalctl -u ai-worker@0 -f` |
| Check metrics | `curl localhost:9090/metrics` |
| Test connectivity | `telnet worker-1 50051` |
| Check config | `python -c "import yaml; yaml.safe_load(open('config/coordinator.yaml'))"` |
| Profile CPU | `perf record -F 99 -p $(pgrep ai-worker) -g` |

### Configuration Templates

#### Minimal Production Config

```yaml
# config/production-minimal.yaml
server:
  host: "0.0.0.0"
  port: 8000

discovery:
  method: "static"
  static_workers:
    - "worker-1:50051"
    - "worker-2:50051"
    - "worker-3:50051"

health:
  check_interval_seconds: 30
  max_failures: 3

models:
  cache_dir: "/data/models"
  auto_load_on_startup: true

security:
  enable_tls: true
  auth:
    enabled: true
    method: "api_key"
```

#### High-Performance Config

```yaml
# config/high-performance.yaml
server:
  workers: 16
  backlog: 4096

performance:
  batching:
    max_batch_size: 64
    batch_timeout_ms: 20
  cache:
    response_cache_size: 100000
  connection_pool:
    max_size: 2000

routing:
  strategy: "least_load"
  queue_size: 5000
  circuit_breaker:
    failure_threshold: 10
```

#### Development Config

```yaml
# config/development.yaml
server:
  host: "127.0.0.1"
  port: 8000
  workers: 1

discovery:
  method: "static"
  static_workers:
    - "localhost:50051"

logging:
  level: "debug"
  format: "text"

security:
  enable_tls: false
  auth:
    enabled: false

advanced:
  debug_mode: true
  enable_pprof: true
```

---

## Conclusion

This deployment guide covers all aspects of deploying the AI Cluster in various environments:

### Key Takeaways

1. **Flexible Deployment Options**: Single machine, multi-machine cluster, Kubernetes, cloud, hybrid
2. **High Availability**: Multi-AZ deployment, leader election, pod distribution, database replication
3. **Security Hardening**: Network policies, TLS, mTLS, Vault integration, audit logging
4. **Performance Tuning**: System-level tuning, NUMA optimization, GPU configuration, batching
5. **Scaling Strategies**: Horizontal scaling, vertical scaling, auto-scaling, cluster auto-scaler
6. **Upgrade Procedures**: Rolling updates, blue-green deployments, canary deployments
7. **Monitoring**: Prometheus, Grafana, Loki, Tempo integration
8. **Backup & Recovery**: Automated backups, point-in-time recovery, disaster recovery plan
9. **Troubleshooting**: Diagnostic tools, profiling, common issues and solutions

### Next Steps

1. Start with the [Quick Start](#quick-start) for a test deployment
2. Move to [Single Machine Deployment](#single-machine-deployment) for production on one server
3. Scale to [Multi-Machine Cluster](#multi-machine-cluster-deployment) for more capacity
4. Implement [High Availability](#high-availability-setup) for production reliability
5. Set up [Monitoring](#monitoring-stack-deployment) and [Backup](#backup-and-recovery)
6. Follow [Security Hardening](#security-hardening) for production security

For more information, see:
- [Architecture Guide](architecture.md)
- [API Reference](api_reference.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting](troubleshooting.md)
```

---

This completes the comprehensive `deployment.md` documentation covering:

1. **Overview** - Deployment architecture options
2. **Prerequisites** - Hardware, software, network requirements
3. **Quick Start** - 5-minute test deployment
4. **Single Machine Deployment** - Step-by-step for one server
5. **Multi-Machine Cluster Deployment** - Distributed across multiple servers
6. **Kubernetes Deployment** - Complete K8s manifests and instructions
7. **Cloud Deployment** - AWS with Terraform, user-data scripts
8. **High Availability Setup** - Multi-AZ, leader election, pod distribution
9. **Monitoring Stack** - Prometheus, Grafana, Loki, Tempo
10. **Backup and Recovery** - Automated scripts, disaster recovery plan
11. **Security Hardening** - Firewall, TLS, Vault, audit logging
12. **Performance Tuning** - System, coordinator, worker, NUMA optimization
13. **Scaling Guide** - Horizontal, vertical, auto-scaling
14. **Upgrade Procedures** - Rolling, blue-green, canary deployments
15. **Troubleshooting** - Common issues, diagnostic commands, profiling
16. **Reference** - Quick reference cards, configuration templates

The documentation is production-ready with practical examples, scripts, and configurations for real-world deployment scenarios.