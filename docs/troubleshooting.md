# Troubleshooting Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Diagnostic Commands](#quick-diagnostic-commands)
3. [Installation Issues](#installation-issues)
4. [GPU Issues](#gpu-issues)
5. [Connection Issues](#connection-issues)
6. [Model Loading Issues](#model-loading-issues)
7. [Inference Issues](#inference-issues)
8. [Performance Issues](#performance-issues)
9. [Memory Issues](#memory-issues)
10. [Network Issues](#network-issues)
11. [Kubernetes Issues](#kubernetes-issues)
12. [Docker Issues](#docker-issues)
13. [Configuration Issues](#configuration-issues)
14. [Security Issues](#security-issues)
15. [Monitoring Issues](#monitoring-issues)
16. [Common Error Messages](#common-error-messages)
17. [Log Analysis](#log-analysis)
18. [Debugging Techniques](#debugging-techniques)
19. [Support Resources](#support-resources)

---

## Overview

This troubleshooting guide helps you diagnose and resolve common issues with the AI Cluster. Each section covers specific problem areas with step-by-step solutions.

### First Response Checklist

When encountering issues, start with this quick checklist:

```bash
# 1. Check system health
./scripts/diagnose.sh

# 2. Check GPU status
nvidia-smi  # or rocm-smi for AMD

# 3. Check service status
systemctl status ai-coordinator
systemctl status ai-worker@0

# 4. Check logs
tail -100 /var/log/ai-cluster/coordinator.log
tail -100 /var/log/ai-cluster/worker.log

# 5. Check network connectivity
curl http://localhost:8000/health
ping worker-1
telnet worker-1 50051
```

---

## Quick Diagnostic Commands

### One-Line Health Check

```bash
# Comprehensive health check
curl -s http://localhost:8000/health && \
curl -s http://localhost:8000/v1/workers | jq '. | length' && \
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv && \
df -h /data && \
free -h
```

### Diagnostic Script Output

```bash
# Run full diagnostic
./scripts/diagnose.sh

=== AI Cluster Diagnostic Report ===
Date: 2024-01-15 10:30:45

=== System Information ===
Linux worker-1 5.15.0-91-generic #101-Ubuntu SMP
Ubuntu 22.04.3 LTS

=== GPU Information ===
name, memory.total [MiB], memory.used [MiB], temperature.gpu, utilization.gpu [%]
"Tesla T4", 15360 MiB, 8923 MiB, 58, 72
"Tesla T4", 15360 MiB, 4215 MiB, 52, 45

=== Service Status ===
● ai-coordinator.service - AI Cluster Coordinator
   Loaded: loaded (/etc/systemd/system/ai-coordinator.service; enabled)
   Active: active (running) since 2024-01-15 09:15:22 UTC

● ai-worker@0.service - AI Worker GPU 0
   Loaded: loaded (/etc/systemd/system/ai-worker@.service; enabled)
   Active: active (running) since 2024-01-15 09:16:34 UTC

=== Coordinator Health ===
{
  "status": "healthy",
  "workers": 4,
  "version": "0.1.0",
  "uptime_seconds": 4513
}

=== Recent Errors (last 50 lines) ===
2024-01-15 10:29:45 [ERROR] worker-0: Failed to load model deepseek-7b: Out of memory
2024-01-15 10:30:12 [WARN] coordinator: Worker worker-2 slow response (2.3s)
```

---

## Installation Issues

### 1. **Rust Installation Fails**

**Symptoms:**
- `cargo` command not found
- Rust compilation errors
- Missing dependencies

**Solutions:**

```bash
# Fix 1: Install Rust correctly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Fix 2: Install build dependencies
sudo apt update
sudo apt install -y build-essential cmake pkg-config libssl-dev

# Fix 3: Check Rust version
rustc --version  # Should be 1.70+

# Fix 4: If behind proxy, configure cargo
export http_proxy=http://proxy:8080
export https_proxy=http://proxy:8080
```

### 2. **Python Dependency Issues**

**Symptoms:**
- `ModuleNotFoundError`
- pip install failures
- Version conflicts

**Solutions:**

```bash
# Fix 1: Use virtual environment
cd coordinator
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Fix 2: Install system dependencies
sudo apt install -y python3-dev build-essential

# Fix 3: Force reinstall
pip install --force-reinstall -r requirements.txt

# Fix 4: Check Python version
python3 --version  # Must be 3.10+
```

### 3. **Docker Installation Issues**

**Symptoms:**
- `docker: command not found`
- Permission denied
- Docker daemon not running

**Solutions:**

```bash
# Fix 1: Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Fix 2: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker  # or log out and back in

# Fix 3: Start Docker daemon
sudo systemctl start docker
sudo systemctl enable docker

# Fix 4: Check Docker version
docker --version
docker-compose --version
```

---

## GPU Issues

### 1. **GPU Not Detected**

**Symptoms:**
- `nvidia-smi` or `rocm-smi` shows no GPUs
- Worker logs: "No GPUs found"
- `gpu_manager.rs` initialization fails

**Diagnostic Steps:**

```bash
# Step 1: Check PCI devices
lspci | grep -E "VGA|3D|Display"

# Step 2: Check drivers loaded
lsmod | grep -E "nvidia|amdgpu"
dmesg | grep -E "nvidia|amdgpu"

# Step 3: Check NVIDIA drivers
nvidia-smi  # Should show GPU info
nvidia-settings -q NvidiaDriverVersion

# Step 4: Check AMD drivers
rocm-smi --showhw
rocminfo | grep "Name:"

# Step 5: Check permissions
ls -la /dev/dri/*  # AMD
ls -la /dev/nvidia*  # NVIDIA
```

**Solutions:**

```bash
# Solution 1: Install NVIDIA drivers
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot

# Solution 2: Install ROCm (AMD)
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo reboot

# Solution 3: Fix permissions
sudo chmod 666 /dev/dri/renderD*
sudo usermod -a -G video,render $USER

# Solution 4: Load kernel modules
sudo modprobe nvidia  # NVIDIA
sudo modprobe amdgpu  # AMD

# Solution 5: Check BIOS settings
# Ensure Above 4G Decoding and Resizable BAR are enabled
```

### 2. **GPU Out of Memory (OOM)**

**Symptoms:**
- Worker logs: "Out of memory"
- Inference fails with OOM error
- GPU memory usage at 100%

**Diagnostic Steps:**

```bash
# Check current memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
rocm-smi --showmeminfo vram

# Monitor memory over time
watch -n 1 nvidia-smi

# Check for memory leaks
for i in {1..10}; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader >> memory.log
    sleep 60
done
```

**Solutions:**

```bash
# Solution 1: Reduce batch size
# In config/worker.toml
[inference]
max_batch_size = 8  # Reduced from 32

# Solution 2: Use quantization
# Load model with INT8
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "deepseek-7b", "quantization": "int8"}'

# Solution 3: Unload unused models
curl -X DELETE http://localhost:8000/v1/models/deepseek-7b

# Solution 4: Clear GPU memory
sudo systemctl restart ai-worker@0

# Solution 5: Reduce model parallelism
# Use fewer GPUs per model
[parallelism]
tensor_parallel_size = 1  # Use single GPU
```

### 3. **GPU Temperature Too High**

**Symptoms:**
- GPU temperature > 85°C
- Performance throttling
- Worker logs temperature warnings

**Solutions:**

```bash
# Solution 1: Check cooling
sensors  # Check system temperatures
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Solution 2: Reduce power limit
sudo nvidia-smi -pl 200  # Set power limit to 200W

# Solution 3: Adjust fan speed (NVIDIA)
sudo nvidia-smi -i 0 -ac 5001,1590
nvidia-settings -a [gpu:0]/GPUFanControlState=1
nvidia-settings -a [fan:0]/GPUTargetFanSpeed=70

# Solution 4: Underclock GPU (AMD)
sudo rocm-smi --setsclk 5  # Set to lower clock level
sudo rocm-smi --setfan 70   # Set fan to 70%

# Solution 5: Improve airflow
# Check server positioning
# Clean dust filters
# Ensure proper rack ventilation
```

### 4. **GPU Utilization Low**

**Symptoms:**
- GPU utilization < 30%
- Low throughput
- CPU bottleneck suspected

**Diagnostic Steps:**

```bash
# Check GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv

# Check CPU usage
top -b -n 1 | grep "Cpu"

# Check I/O wait
iostat -x 1

# Profile with nvtop
nvtop  # Interactive GPU monitoring
```

**Solutions:**

```bash
# Solution 1: Increase batch size
[inference]
max_batch_size = 64  # Increase batch size

# Solution 2: Enable continuous batching
[inference]
enable_continuous_batching = true
batch_timeout_ms = 20

# Solution 3: Use tensor parallelism
[parallelism]
tensor_parallel_size = 2  # Use 2 GPUs

# Solution 4: Check CPU-GPU transfer
# Profile data transfer times
export RUST_LOG=debug
systemctl restart ai-worker@0

# Solution 5: Use faster CPU or NVMe storage
# Move models to faster storage
mv /data/models /fast-nvme/models
ln -s /fast-nvme/models /data/models
```

---

## Connection Issues

### 1. **Worker Cannot Connect to Coordinator**

**Symptoms:**
- Coordinator shows 0 workers
- Worker logs: "Failed to connect to coordinator"
- `curl http://coordinator:8000/v1/workers` returns empty list

**Diagnostic Steps:**

```bash
# Step 1: Check network connectivity
ping coordinator
telnet coordinator 8000

# Step 2: Check coordinator health
curl http://coordinator:8000/health

# Step 3: Check worker logs
journalctl -u ai-worker@0 -f

# Step 4: Check firewall rules
sudo ufw status
sudo iptables -L -n | grep 8000
```

**Solutions:**

```bash
# Solution 1: Update worker configuration
# In config/worker.toml
[worker]
coordinator_url = "http://coordinator:8000"  # Use correct hostname

# Solution 2: Fix firewall rules
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 50051 proto tcp

# Solution 3: Update /etc/hosts
echo "192.168.1.10 coordinator" | sudo tee -a /etc/hosts

# Solution 4: Check DNS resolution
nslookup coordinator
dig coordinator

# Solution 5: Restart services
sudo systemctl restart ai-coordinator
sudo systemctl restart ai-worker@0
```

### 2. **gRPC Connection Failed**

**Symptoms:**
- "rpc error: code = Unavailable"
- "connection refused"
- Worker shows as unhealthy

**Diagnostic Steps:**

```bash
# Step 1: Check gRPC port
ss -tlnp | grep 50051
netstat -an | grep 50051

# Step 2: Test gRPC with grpcurl
grpcurl -plaintext worker-1:50051 list

# Step 3: Check gRPC logs
export GRPC_VERBOSITY=DEBUG
export GRPC_TRACE=all
systemctl restart ai-worker@0
```

**Solutions:**

```bash
# Solution 1: Check worker is running
systemctl status ai-worker@0

# Solution 2: Increase gRPC message size
# In config/worker.toml
[grpc]
max_message_size_mb = 200  # Increase from 100

# Solution 3: Adjust keepalive settings
[grpc]
keepalive_time_ms = 5000
keepalive_timeout_ms = 2000
keepalive_permit_without_calls = true

# Solution 4: Check for port conflicts
sudo lsof -i :50051
kill -9 <PID>  # If another process is using the port

# Solution 5: Restart with clean state
sudo systemctl stop ai-worker@0
sudo pkill ai-worker
sudo systemctl start ai-worker@0
```

### 3. **Discovery Failing**

**Symptoms:**
- Workers not discovered automatically
- Static discovery works but mDNS doesn't
- "No workers found" in logs

**Diagnostic Steps:**

```bash
# Step 1: Check discovery method
grep discovery_method config/coordinator.yaml

# Step 2: Test mDNS
avahi-browse -a -t  # Linux
dns-sd -B _ai-worker._tcp local  # macOS

# Step 3: Check broadcast
tcpdump -i eth0 port 50052

# Step 4: Check Consul
curl http://consul:8500/v1/catalog/services
```

**Solutions:**

```bash
# Solution 1: Switch to static discovery
# config/coordinator.yaml
discovery:
  method: "static"
  static_workers:
    - "192.168.1.11:50051"
    - "192.168.1.12:50051"

# Solution 2: Install mDNS support
sudo apt install -y avahi-daemon avahi-utils
sudo systemctl enable avahi-daemon
sudo systemctl start avahi-daemon

# Solution 3: Check broadcast address
# config/coordinator.yaml
broadcast:
  broadcast_address: "192.168.1.255"  # Correct subnet broadcast

# Solution 4: Consul troubleshooting
consul members
consul info
```

---

## Model Loading Issues

### 1. **Model Download Fails**

**Symptoms:**
- "Failed to download model from HuggingFace"
- Network timeout
- Authentication required

**Diagnostic Steps:**

```bash
# Step 1: Check internet connectivity
ping huggingface.co

# Step 2: Check disk space
df -h /data

# Step 3: Check HuggingFace token
echo $HF_TOKEN
cat ~/.huggingface/token

# Step 4: Test download manually
wget https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/resolve/main/config.json
```

**Solutions:**

```bash
# Solution 1: Set HuggingFace token
export HF_TOKEN="hf_xxxxxxxxxxxx"
# Or in config/worker.toml
[model_loader.huggingface]
token = "hf_xxxxxxxxxxxx"

# Solution 2: Use mirror
export HF_ENDPOINT=https://hf-mirror.com

# Solution 3: Download manually
cd /data/models
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base

# Solution 4: Increase timeout
[model_loader]
download_timeout_seconds = 600  # Increase from 300

# Solution 5: Use offline mode
[model_loader]
offline_mode = true
# Place models manually in /data/models
```

### 2. **Model Conversion Fails**

**Symptoms:**
- "Failed to convert model weights"
- "Unsupported model architecture"
- "CUDA out of memory" during conversion

**Diagnostic Steps:**

```bash
# Step 1: Check model files
ls -la /data/models/deepseek-7b/
file /data/models/deepseek-7b/model.safetensors

# Step 2: Check Python conversion script
python scripts/convert_model.py --model deepseek-7b --dry-run

# Step 3: Check available memory
free -h
nvidia-smi
```

**Solutions:**

```bash
# Solution 1: Convert on CPU
python scripts/convert_model.py \
  --model deepseek-7b \
  --device cpu \
  --quantize int8

# Solution 2: Use smaller shards
python scripts/convert_model.py \
  --model deepseek-7b \
  --max-shard-size 1  # 1GB shards

# Solution 3: Skip verification
python scripts/convert_model.py \
  --model deepseek-7b \
  --no-verify

# Solution 4: Manual conversion with transformers
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-llm-7b-base')
model.save_pretrained('./models/deepseek-7b', safe_serialization=True)
"
```

### 3. **Model Load Fails on Worker**

**Symptoms:**
- Worker logs: "Failed to load model"
- "Error loading model weights"
- Model shows as not loaded

**Diagnostic Steps:**

```bash
# Step 1: Check worker logs
journalctl -u ai-worker@0 -f

# Step 2: Check model path
ls -la /data/models/deepseek-7b/

# Step 3: Check permissions
ls -la /data/models/
sudo -u worker ls -la /data/models/deepseek-7b/

# Step 4: Check GPU memory
nvidia-smi
```

**Solutions:**

```bash
# Solution 1: Fix permissions
sudo chown -R worker:worker /data/models
sudo chmod 755 /data/models
sudo chmod 644 /data/models/**/*

# Solution 2: Load with different quantization
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "deepseek-7b", "quantization": "int8"}'

# Solution 3: Load on specific GPU
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "deepseek-7b", "gpu_ids": [0]}'

# Solution 4: Clear and retry
sudo systemctl stop ai-worker@0
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'  # Clear disk cache
sudo systemctl start ai-worker@0

# Solution 5: Check model integrity
md5sum -c /data/models/deepseek-7b/model.safetensors.md5
```

---

## Inference Issues

### 1. **Slow Inference**

**Symptoms:**
- High latency (> 1 second per token)
- Low throughput (< 10 tokens/sec)
- Users complain about slow responses

**Diagnostic Steps:**

```bash
# Step 1: Measure current performance
curl -w "@curl-format.txt" -X POST http://localhost:8000/v1/completions \
  -d '{"model":"deepseek-7b","prompt":"Hello","max_tokens":100}'

# Step 2: Check GPU utilization
watch -n 1 nvidia-smi

# Step 3: Check batch size
curl http://localhost:9090/metrics | grep batch_size

# Step 4: Profile with perf
perf record -F 99 -p $(pgrep ai-worker) -g -- sleep 30
perf report
```

**Solutions:**

```bash
# Solution 1: Enable continuous batching
[inference]
enable_continuous_batching = true
max_batch_size = 32
batch_timeout_ms = 25

# Solution 2: Use quantization
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "deepseek-7b", "quantization": "int8"}'

# Solution 3: Enable KV cache
[kv_cache]
enabled = true
max_cache_size_gb = 16
enable_prefix_caching = true

# Solution 4: Use tensor parallelism
[parallelism]
tensor_parallel_size = 2
enable_sequence_parallel = true

# Solution 5: Optimize network
# Use RDMA if available
[communication]
enable_infiniband = true
enable_nvlink = true
```

### 2. **Poor Quality Output**

**Symptoms:**
- Nonsensical responses
- Repetitive text
- Incorrect format
- Hallucinations

**Diagnostic Steps:**

```bash
# Step 1: Check temperature setting
curl -X POST http://localhost:8000/v1/completions \
  -d '{"model":"deepseek-7b","prompt":"Hello","temperature":0.1}'

# Step 2: Test with known prompts
curl -X POST http://localhost:8000/v1/completions \
  -d '{"model":"deepseek-7b","prompt":"What is 2+2?","max_tokens":10}'

# Step 3: Check model integrity
# Compare with expected output
```

**Solutions:**

```bash
# Solution 1: Adjust temperature
# Lower temperature for more deterministic output
curl -X POST http://localhost:8000/v1/completions \
  -d '{"model":"deepseek-7b","prompt":"Hello","temperature":0.1}'

# Solution 2: Use top_p sampling
curl -X POST http://localhost:8000/v1/completions \
  -d '{"model":"deepseek-7b","prompt":"Hello","top_p":0.9}'

# Solution 3: Add stop sequences
curl -X POST http://localhost:8000/v1/completions \
  -d '{
    "model":"deepseek-7b",
    "prompt":"Hello",
    "stop": ["\n", "Human:"]
  }'

# Solution 4: Reload model (might be corrupted)
curl -X DELETE http://localhost:8000/v1/models/deepseek-7b
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "deepseek-7b"}'
```

### 3. **Streaming Not Working**

**Symptoms:**
- `stream: true` returns no output
- Chunks arrive late or not at all
- Client times out

**Diagnostic Steps:**

```bash
# Step 1: Test with curl
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-7b","prompt":"Hello","stream":true}'

# Step 2: Check server-sent events support
curl -I http://localhost:8000/v1/completions

# Step 3: Check worker streaming setting
grep enable_streaming config/worker.toml
```

**Solutions:**

```bash
# Solution 1: Enable streaming in worker
[inference]
enable_streaming = true
stream_chunk_size = 1

# Solution 2: Increase buffer size
[grpc]
max_message_size_mb = 100

# Solution 3: Client implementation
# Python
import httpx
async with httpx.AsyncClient() as client:
    async with client.stream("POST", url, json=payload) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                print(line[6:])

# Solution 4: Use WebSocket fallback
ws = websocket.connect("ws://localhost:8000/v1/stream")
ws.send(json.dumps(payload))
for message in ws:
    print(message)
```

---

## Performance Issues

### 1. **High CPU Usage**

**Symptoms:**
- CPU > 90%
- System slow
- Top shows high CPU for worker

**Diagnostic Steps:**

```bash
# Step 1: Identify CPU-intensive processes
top -b -n 1 | head -20
htop

# Step 2: Check per-core usage
mpstat -P ALL 1

# Step 3: Profile CPU
perf top -p $(pgrep ai-worker)

# Step 4: Check system calls
strace -p $(pgrep ai-worker) -c
```

**Solutions:**

```bash
# Solution 1: Limit CPU cores
[resources]
max_cpu_cores = 8  # Limit to 8 cores
cpu_affinity = "0-7"  # Pin to specific cores

# Solution 2: Adjust nice level
[resources]
nice_level = 10  # Lower priority for background tasks

# Solution 3: Optimize tokenization
# Use faster tokenizer
[model_loader]
tokenizer_type = "fast"  # Use Rust-based tokenizer

# Solution 4: Reduce Python overhead
# Increase worker processes in coordinator
[server]
workers = 8  # Match CPU cores
```

### 2. **High Memory Usage**

**Symptoms:**
- System memory > 90%
- OOM killer triggered
- Swap usage high

**Diagnostic Steps:**

```bash
# Step 1: Check memory usage
free -h
vmstat 1 10

# Step 2: Find memory consumers
ps aux --sort=-%mem | head -20

# Step 3: Check for memory leaks
valgrind --leak-check=full ./worker/target/release/ai-worker

# Step 4: Monitor over time
while true; do
    date >> memory.log
    free -h >> memory.log
    sleep 60
done
```

**Solutions:**

```bash
# Solution 1: Limit memory usage
[resources]
max_memory_gb = 32  # Hard limit
[gpu]
memory_fraction = 0.8  # Use 80% of GPU memory

# Solution 2: Reduce cache sizes
[cache]
model_cache_size_gb = 20
kv_cache_size_gb = 8

# Solution 3: Use swap (emergency)
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Solution 4: Restart periodically
# Add to crontab
0 3 * * * /usr/bin/systemctl restart ai-worker@0
```

### 3. **High Disk I/O**

**Symptoms:**
- Disk I/O wait high
- Slow model loading
- iostat shows high utilization

**Diagnostic Steps:**

```bash
# Step 1: Check disk I/O
iostat -x 1
iotop

# Step 2: Find I/O-intensive processes
lsof /data/models

# Step 3: Check disk health
smartctl -a /dev/sda
```

**Solutions:**

```bash
# Solution 1: Use faster storage
# Move to NVMe
mv /data/models /mnt/nvme/models
ln -s /mnt/nvme/models /data/models

# Solution 2: Enable caching
[cache]
enable_mmap = true
prefetch_size_gb = 4

# Solution 3: Use RAM disk for hot models
sudo mkdir /mnt/ramdisk
sudo mount -t tmpfs -o size=32G tmpfs /mnt/ramdisk
cp -r /data/models/hot-model /mnt/ramdisk/

# Solution 4: Adjust I/O scheduler
echo deadline > /sys/block/sda/queue/scheduler
```

---

## Memory Issues

### 1. **CUDA Out of Memory**

**Symptoms:**
- "CUDA out of memory" error
- Inference fails
- GPU memory full

**Diagnostic Steps:**

```bash
# Step 1: Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Step 2: Find memory consumers
nvidia-smi | grep "Process"

# Step 3: Monitor memory over time
watch -n 1 nvidia-smi
```

**Solutions:**

```bash
# Solution 1: Clear GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>  # Kill processes using GPU

# Solution 2: Use memory pooling
[gpu]
enable_memory_pooling = true
memory_pool_size_gb = 4

# Solution 3: Enable memory defragmentation
# In worker code periodically
pub fn defragment_memory() {
    unsafe {
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

# Solution 4: Reduce model size
# Use int4 quantization
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "deepseek-7b", "quantization": "int4"}'
```

### 2. **Memory Fragmentation**

**Symptoms:**
- OOM errors despite free memory
- Allocation failures
- Performance degradation

**Diagnostic Steps:**

```bash
# Step 1: Check memory fragmentation
nvidia-smi -q -d MEMORY | grep "Free"

# Step 2: Monitor allocation patterns
export CUDA_MEMCHECK=1
./worker/target/release/ai-worker
```

**Solutions:**

```bash
# Solution 1: Enable memory defrag
[gpu]
defragment_interval_seconds = 3600

# Solution 2: Use larger block sizes
[kv_cache]
block_size = 32  # Increase from 16

# Solution 3: Restart worker periodically
0 4 * * * /usr/bin/systemctl restart ai-worker@0

# Solution 4: Pre-allocate memory
[model_loader]
prefetch_size_gb = 4
pin_memory = true
```

---

## Network Issues

### 1. **High Network Latency**

**Symptoms:**
- Slow cross-node communication
- High ping times
- gRPC timeouts

**Diagnostic Steps:**

```bash
# Step 1: Measure latency
ping -c 10 worker-1
mtr worker-1

# Step 2: Check bandwidth
iperf3 -c worker-1

# Step 3: Monitor network
iftop
nload
```

**Solutions:**

```bash
# Solution 1: Use RDMA if available
[communication]
enable_infiniband = true
enable_roce = true
roce_interface = "eth0"

# Solution 2: Adjust TCP buffers
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728

# Solution 3: Use compression
[communication]
enable_compression = true
compression_algorithm = "lz4"
compression_level = 1

# Solution 4: Co-locate communicating workers
# Use Kubernetes pod affinity
podAffinity:
  requiredDuringScheduling:
  - labelSelector:
      matchLabels:
        app: worker
    topologyKey: kubernetes.io/hostname
```

### 2. **Packet Loss**

**Symptoms:**
- gRPC retransmissions
- Connection resets
- Incomplete responses

**Diagnostic Steps:**

```bash
# Step 1: Check for packet loss
ping -c 100 -i 0.1 worker-1 | grep loss

# Step 2: Capture packets
tcpdump -i eth0 -w capture.pcap
tshark -r capture.pcap -Y "tcp.analysis.retransmission"

# Step 3: Check network errors
netstat -s | grep -i "loss\|retransmit"
```

**Solutions:**

```bash
# Solution 1: Increase TCP retry
sudo sysctl -w net.ipv4.tcp_retries2=15

# Solution 2: Enable congestion control
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# Solution 3: Use QoS
# Configure switch for QoS
# Mark packets with DSCP
iptables -t mangle -A OUTPUT -p tcp --dport 50051 -j DSCP --set-dscp 46

# Solution 4: Add redundancy
# Use multiple network paths
# Bond interfaces
```

---

## Kubernetes Issues

### 1. **Pod Stuck in Pending**

**Symptoms:**
- Pod shows "Pending" status
- Not scheduling
- `kubectl get pods` shows pending

**Diagnostic Steps:**

```bash
# Step 1: Describe pod
kubectl describe pod worker-xxxxx -n ai-cluster

# Step 2: Check node resources
kubectl get nodes
kubectl describe node gpu-node-1

# Step 3: Check GPU availability
kubectl get nodes -l nvidia.com/gpu.present=true
```

**Solutions:**

```bash
# Solution 1: Add GPU nodes
kubectl label node gpu-node-1 nvidia.com/gpu.present=true

# Solution 2: Check resource requests
# Adjust in deployment
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1

# Solution 3: Remove taints
kubectl taint nodes gpu-node-1 nvidia.com/gpu:NoSchedule-

# Solution 4: Scale node group
eksctl scale nodegroup --cluster=ai-cluster --name=gpu-nodes --nodes=5
```

### 2. **GPU Not Allocatable in Pod**

**Symptoms:**
- Pod requests GPU but gets "0/1 nodes are available"
- `nvidia.com/gpu` not showing in node capacity

**Diagnostic Steps:**

```bash
# Step 1: Check node capacity
kubectl describe node gpu-node-1 | grep -A5 "Capacity"

# Step 2: Check device plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Step 3: Check plugin logs
kubectl logs -n kube-system nvidia-device-plugin-xxxxx
```

**Solutions:**

```bash
# Solution 1: Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Solution 2: Restart device plugin
kubectl delete pods -n kube-system -l name=nvidia-device-plugin-ds

# Solution 3: Check driver installation
kubectl exec -it nvidia-device-plugin-xxxxx -n kube-system -- nvidia-smi

# Solution 4: Manual plugin configuration
# Edit daemonset
kubectl edit daemonset nvidia-device-plugin-daemonset -n kube-system
```

### 3. **Pod CrashLoopBackOff**

**Symptoms:**
- Pod repeatedly crashing
- Container exits immediately
- CrashLoopBackOff status

**Diagnostic Steps:**

```bash
# Step 1: Check logs
kubectl logs worker-xxxxx -n ai-cluster --previous

# Step 2: Check events
kubectl get events -n ai-cluster --sort-by='.lastTimestamp'

# Step 3: Exec into pod (if possible)
kubectl exec -it worker-xxxxx -n ai-cluster -- /bin/sh
```

**Solutions:**

```bash
# Solution 1: Increase startup time
# Add startup probe
startupProbe:
  httpGet:
    path: /health/startup
    port: 9091
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 30

# Solution 2: Adjust resource limits
resources:
  requests:
    memory: "16Gi"
    cpu: "8"
  limits:
    memory: "32Gi"
    cpu: "16"

# Solution 3: Check configuration
kubectl exec worker-xxxxx -n ai-cluster -- cat /app/config/worker.toml

# Solution 4: Debug with sleep
# Temporarily override command
command: ["sleep", "3600"]
```

---

## Docker Issues

### 1. **Container Exits Immediately**

**Symptoms:**
- Container runs then exits
- `docker ps -a` shows Exited
- No logs

**Diagnostic Steps:**

```bash
# Step 1: Check logs
docker logs ai-worker-0

# Step 2: Run interactively
docker run -it --rm ai-worker:latest /bin/bash

# Step 3: Check entrypoint
docker inspect ai-worker-0 | grep -A5 "Entrypoint"
```

**Solutions:**

```bash
# Solution 1: Override command
docker run --rm ai-worker:latest /usr/local/bin/ai-worker --help

# Solution 2: Check config file exists
docker run --rm -v $(pwd)/config:/app/config:ro \
  ai-worker:latest ls -la /app/config/

# Solution 3: Fix permissions
# In Dockerfile
USER worker
RUN chmod +x /usr/local/bin/ai-worker

# Solution 4: Debug with sleep
docker run -d --entrypoint sleep ai-worker:latest infinity
docker exec -it <container> /bin/bash
```

### 2. **GPU Not Accessible in Container**

**Symptoms:**
- `nvidia-smi` not found in container
- Worker can't detect GPU
- `--gpus all` has no effect

**Diagnostic Steps:**

```bash
# Step 1: Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1-runtime nvidia-smi

# Step 2: Check Docker runtime
docker info | grep "Runtimes"

# Step 3: Check container capabilities
docker inspect ai-worker-0 | grep -A10 "HostConfig"
```

**Solutions:**

```bash
# Solution 1: Install NVIDIA container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Solution 2: Set default runtime
# /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Solution 3: Use docker-compose with GPU
# docker-compose.yml
services:
  worker:
    image: ai-worker:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. **Container Networking Issues**

**Symptoms:**
- Can't connect to other containers
- Ports not exposed
- DNS resolution fails

**Diagnostic Steps:**

```bash
# Step 1: Check network
docker network ls
docker inspect ai-cluster-net

# Step 2: Test connectivity
docker exec ai-coordinator ping ai-worker-0

# Step 3: Check port mapping
docker port ai-coordinator
```

**Solutions:**

```bash
# Solution 1: Use custom network
docker network create ai-cluster-net
docker run --network ai-cluster-net --name coordinator -d ai-coordinator
docker run --network ai-cluster-net --name worker-0 -d ai-worker

# Solution 2: Expose ports correctly
# docker-compose.yml
ports:
  - "8000:8000"  # host:container

# Solution 3: Use service discovery
# In coordinator config
discovery:
  method: "docker"
  docker:
    network: "ai-cluster-net"
    service_name: "worker"

# Solution 4: Check firewall
sudo ufw allow from 172.17.0.0/16
```

---

## Configuration Issues

### 1. **Invalid YAML/TOML Syntax**

**Symptoms:**
- Service fails to start
- "Error parsing config file"
- Syntax error messages

**Diagnostic Steps:**

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/coordinator.yaml'))"

# Check TOML syntax
python -c "import toml; toml.load('config/worker.toml')"

# Validate with linter
yamllint config/coordinator.yaml
```

**Solutions:**

```bash
# Fix common YAML issues
# Use online validator
# Check indentation (2 spaces)
# Remove tabs
# Quote strings with special characters

# Fix common TOML issues
# Use = for assignments
# Quote keys with dots
# Use [[array]] for arrays of tables

# Use example config
cp config/coordinator.yaml.example config/coordinator.yaml
cp config/worker.toml.example config/worker.toml
```

### 2. **Missing Configuration Values**

**Symptoms:**
- Service uses defaults unexpectedly
- "Key not found" errors
- Unexpected behavior

**Diagnostic Steps:**

```bash
# Dump configuration on startup
./worker/target/release/ai-worker --dump-config

# Check environment variables
env | grep COORDINATOR_

# Validate with script
python scripts/validate_config.py
```

**Solutions:**

```bash
# Set required values
# config/coordinator.yaml
server:
  host: "0.0.0.0"  # Required
  port: 8000  # Required

# Use environment variables
export COORDINATOR_HOST="0.0.0.0"
export COORDINATOR_PORT="8000"

# Set defaults in code
impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            grpc_port: 50051,
            max_batch_size: 32,
            // ...
        }
    }
}
```

### 3. **Path Resolution Issues**

**Symptoms:**
- "File not found" errors
- Relative paths not working
- Permission denied

**Diagnostic Steps:**

```bash
# Check working directory
pwd
ls -la

# Check absolute paths
realpath config/coordinator.yaml

# Check permissions
ls -la /data/models/
```

**Solutions:**

```bash
# Use absolute paths
# config/coordinator.yaml
models:
  cache_dir: "/absolute/path/to/models"

# Create directories with correct permissions
sudo mkdir -p /data/models
sudo chown -R $USER:$USER /data
chmod 755 /data/models

# Use environment variables for paths
export MODEL_CACHE_DIR="/data/models"
# In config
models:
  cache_dir: ${MODEL_CACHE_DIR:-./models}
```

---

## Security Issues

### 1. **Authentication Failures**

**Symptoms:**
- 401 Unauthorized responses
- "Invalid API key" errors
- Rate limiting not working

**Diagnostic Steps:**

```bash
# Test with correct key
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/v1/models

# Check API key file
cat /etc/ai-cluster/api_keys.txt

# Check auth logs
tail -f /var/log/ai-cluster/audit.log | grep authentication
```

**Solutions:**

```bash
# Generate new API key
openssl rand -base64 32

# Add to keys file
echo "sk-$(openssl rand -hex 20)" >> /etc/ai-cluster/api_keys.txt

# Reload keys without restart
curl -X POST http://localhost:8000/v1/config/reload

# Fix header format
curl -H "X-API-Key: $API_KEY" ...  # Alternative header
```

### 2. **TLS/SSL Issues**

**Symptoms:**
- "certificate expired" errors
- "TLS handshake failed"
- Can't connect with HTTPS

**Diagnostic Steps:**

```bash
# Check certificate
openssl x509 -in /etc/ai-cluster/certs/coordinator.crt -text -noout

# Test TLS connection
openssl s_client -connect localhost:8000 -servername localhost

# Check certificate expiration
openssl x509 -in /etc/ai-cluster/certs/coordinator.crt -noout -enddate
```

**Solutions:**

```bash
# Generate new self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update certificates
sudo cp cert.pem /etc/ai-cluster/certs/
sudo cp key.pem /etc/ai-cluster/certs/

# Configure TLS properly
# config/coordinator.yaml
security:
  enable_tls: true
  tls:
    cert_file: "/etc/ai-cluster/certs/coordinator.crt"
    key_file: "/etc/ai-cluster/certs/coordinator.key"
    ca_file: "/etc/ai-cluster/certs/ca.crt"
    min_version: "1.2"
```

### 3. **Permission Issues**

**Symptoms:**
- "Permission denied" errors
- Can't read/write files
- Can't access GPU

**Diagnostic Steps:**

```bash
# Check process user
ps aux | grep ai-worker

# Check file permissions
ls -la /data/models/
namei -l /data/models/model.safetensors

# Check group membership
groups $USER
groups worker
```

**Solutions:**

```bash
# Fix ownership
sudo chown -R worker:worker /data/models
sudo chown -R worker:worker /var/log/ai-cluster

# Add user to groups
sudo usermod -a -G video,render worker
sudo usermod -a -G docker $USER

# Set proper permissions
chmod 755 /data
chmod 755 /data/models
chmod 644 /data/models/**/*

# Use ACLs for fine-grained control
setfacl -m u:worker:rwx /data/models
```

---

## Common Error Messages

### Coordinator Errors

| Error Message | Likely Cause | Solution |
|--------------|--------------|----------|
| `No workers available` | All workers offline or unhealthy | Check worker status, network connectivity |
| `Model not found: {name}` | Model not in registry | Check models.toml, model name spelling |
| `Failed to load model: OOM` | Insufficient GPU memory | Use quantization, reduce batch size |
| `Request timeout` | Worker too slow or hung | Increase timeout, check worker performance |
| `Circuit breaker open` | Worker failing too often | Check worker logs, restart worker |
| `Rate limit exceeded` | Too many requests | Wait, increase rate limit, get higher tier |
| `Invalid API key` | Authentication failed | Check API key, regenerate if needed |
| `Connection refused` | Worker not running | Start worker, check port |

### Worker Errors

| Error Message | Likely Cause | Solution |
|--------------|--------------|----------|
| `No GPUs found` | GPU drivers not installed | Install drivers, check permissions |
| `CUDA out of memory` | GPU memory full | Reduce batch size, unload models |
| `Failed to load model weights` | Corrupted model files | Redownload model, verify checksums |
| `Unsupported architecture` | Model not supported | Check model family, implement support |
| `gRPC connection failed` | Network issue | Check connectivity, firewall |
| `Thread pool exhausted` | Too many concurrent requests | Increase thread pool, add workers |
| `KV cache full` | Cache size exceeded | Increase cache size, enable compression |
| `Tokenizer not found` | Missing tokenizer files | Download tokenizer, check paths |

### Kubernetes Errors

| Error Message | Likely Cause | Solution |
|--------------|--------------|----------|
| `0/1 nodes are available` | No resources available | Add nodes, reduce requests |
| `Pending` pod status | Scheduling issues | Check node resources, taints |
| `CrashLoopBackOff` | Container keeps crashing | Check logs, increase startup time |
| `ImagePullBackOff` | Can't pull image | Check image name, registry access |
| `Failed to mount volume` | PVC issues | Check PVC status, storage class |
| `Unschedulable` | Node selector mismatch | Check node labels, add matching nodes |
| `OOMKilled` | Out of memory | Increase memory limits |

---

## Log Analysis

### Important Log Locations

| Component | Log Location |
|-----------|-------------|
| Coordinator | `/var/log/ai-cluster/coordinator.log` |
| Worker | `/var/log/ai-cluster/worker.log` |
| Audit | `/var/log/ai-cluster/audit.log` |
| Systemd | `journalctl -u ai-coordinator` |
| Docker | `docker logs ai-coordinator` |
| Kubernetes | `kubectl logs -n ai-cluster pod-name` |

### Log Analysis Script

```bash
#!/bin/bash
# analyze-logs.sh - Analyze logs for common issues

LOG_DIR="/var/log/ai-cluster"
REPORT="log-analysis-$(date +%Y%m%d).txt"

echo "=== Log Analysis Report $(date) ===" > $REPORT

# Count error types
echo -e "\n=== Error Summary ===" >> $REPORT
grep -h "ERROR" $LOG_DIR/*.log | cut -d' ' -f4- | sort | uniq -c | sort -rn >> $REPORT

# Find slow requests (> 5s)
echo -e "\n=== Slow Requests (>5s) ===" >> $REPORT
grep "duration.*[5-9][0-9][0-9][0-9]" $LOG_DIR/coordinator.log >> $REPORT

# Check for OOM events
echo -e "\n=== OOM Events ===" >> $REPORT
grep -i "out of memory\|oom" $LOG_DIR/*.log >> $REPORT

# Check for connection issues
echo -e "\n=== Connection Issues ===" >> $REPORT
grep -i "connection refused\|timeout\|unreachable" $LOG_DIR/*.log >> $REPORT

# Check for model load failures
echo -e "\n=== Model Load Failures ===" >> $REPORT
grep "Failed to load model" $LOG_DIR/worker.log >> $REPORT

# GPU errors
echo -e "\n=== GPU Errors ===" >> $REPORT
grep -i "cuda\|rocm\|gpu" $LOG_DIR/worker.log | grep -i "error\|fail" >> $REPORT

# Rate limiting
echo -e "\n=== Rate Limiting ===" >> $REPORT
grep "rate limit" $LOG_DIR/coordinator.log >> $REPORT

echo -e "\n=== Analysis Complete ===" >> $REPORT
cat $REPORT
```

### Real-time Log Monitoring

```bash
# Monitor all logs
multitail /var/log/ai-cluster/*.log

# Follow with filtering
tail -f /var/log/ai-cluster/coordinator.log | grep -E "ERROR|WARN"

# Use journalctl
journalctl -u ai-coordinator -f -o json | jq '.MESSAGE' | grep -i error

# Docker logs with timestamps
docker logs -f ai-worker-0 --timestamps

# Kubernetes logs with label selector
kubectl logs -f -l app=worker -n ai-cluster --all-containers --max-log-requests=10
```

---

## Debugging Techniques

### 1. **Enable Debug Logging**

```bash
# Coordinator
export LOG_LEVEL=debug
systemctl restart ai-coordinator

# Worker
export RUST_LOG=debug
systemctl restart ai-worker@0

# gRPC debugging
export GRPC_VERBOSITY=DEBUG
export GRPC_TRACE=all,http,api
```

### 2. **Remote Debugging**

```python
# coordinator/debug.py
import debugpy

# Enable remote debugging
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
```

```rust
// worker/src/debug.rs
use tracing::{debug, info, warn, error};

// Add detailed tracing
#[instrument(skip(self))]
async fn process_request(&self, req: Request) -> Result<Response, Error> {
    debug!("Processing request: {:?}", req);
    // ...
}
```

### 3. **Profiling**

```bash
# CPU profiling
perf record -F 99 -p $(pgrep ai-worker) -g -- sleep 60
perf report -g graph --no-children

# Memory profiling
heaptrack ./worker/target/release/ai-worker
heaptrack --analyze heaptrack.ai-worker.*.gz

# GPU profiling
nsys profile -o profile.nsys-rep ./worker/target/release/ai-worker
ncu -o profile.ncu-rep ./worker/target/release/ai-worker
```

### 4. **Tracing with Jaeger**

```yaml
# config/tracing.yaml
tracing:
  enabled: true
  provider: jaeger
  jaeger:
    agent_host: "localhost"
    agent_port: 6831
    service_name: "ai-worker"
    sample_rate: 1.0
```

---

## Support Resources

### Documentation

- [Architecture Guide](architecture.md)
- [API Reference](api_reference.md)
- [Configuration Guide](configuration.md)
- [Deployment Guide](deployment.md)


### Useful Tools

```bash
# Monitoring
htop, nvtop, iotop, iftop

# Profiling
perf, valgrind, heaptrack, nsys

# Networking
tcpdump, wireshark, mtr, iperf3

# Debugging
gdb, lldb, strace, ltrace

# Container
docker, crictl, ctr

# Kubernetes
kubectl, k9s, stern, kube-ps1
```

---

## Quick Reference Card

```bash
# Quick diagnostics
./scripts/diagnose.sh

# Check GPU
nvidia-smi  # or rocm-smi

# Check services
systemctl status ai-*

# Check logs
journalctl -u ai-worker@0 -f

# Test API
curl http://localhost:8000/health

# Check workers
curl http://localhost:8000/v1/workers | jq

# Check models
curl http://localhost:8000/v1/models | jq

# Reload config
curl -X POST http://localhost:8000/v1/config/reload

# Reset worker
sudo systemctl restart ai-worker@0

# Clear GPU memory
sudo nvidia-smi -r
```
