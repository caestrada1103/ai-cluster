#!/bin/bash
# CUDA Setup Script for NVIDIA GPUs
# ================================
# This script installs and configures CUDA for NVIDIA GPUs
# Supports Ubuntu 20.04, 22.04, and 24.04

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CUDA_VERSION="${CUDA_VERSION:-12.1}"  # Default CUDA version
CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_VERSION_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
INSTALL_DRIVER="${INSTALL_DRIVER:-yes}"
INSTALL_DOCKER="${INSTALL_DOCKER:-yes}"
INSTALL_CONTAINER_TOOLKIT="${INSTALL_CONTAINER_TOOLKIT:-yes}"
GPU_IDS="${GPU_IDS:-0}"  # Comma-separated list of GPU IDs to use
SETUP_PERSISTENCE="${SETUP_PERSISTENCE:-yes}"
SETUP_MIG="${SETUP_MIG:-no}"  # Multi-Instance GPU for A100/H100
SETUP_DEBUG="${SETUP_DEBUG:-no}"

# Log file
LOG_FILE="/tmp/cuda-setup-$(date +%Y%m%d-%H%M%S).log"

# Function definitions
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${MAGENTA}⚠ $1${NC}"
}

print_debug() {
    if [ "$SETUP_DEBUG" = "yes" ]; then
        echo -e "${CYAN}🔍 $1${NC}"
    fi
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    print_debug "$1"
}

check_root() {
    if [ "$EUID" -eq 0 ]; then 
        print_warning "Running as root. It's recommended to run as a regular user with sudo access."
    fi
}

detect_os() {
    print_header "Detecting Operating System"
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
        UBUNTU_VERSION=$VERSION_ID
        print_success "Detected OS: $OS $VERSION"
        log "OS: $OS $VERSION"
    else
        print_error "Cannot detect OS"
        exit 1
    fi
    
    # Check if Ubuntu
    if [ "$OS" != "ubuntu" ]; then
        print_error "This script is designed for Ubuntu. Detected: $OS"
        exit 1
    fi
    
    # Check Ubuntu version
    case $UBUNTU_VERSION in
        20.04|22.04|24.04)
            print_success "Ubuntu $UBUNTU_VERSION is supported"
            ;;
        *)
            print_warning "Ubuntu $UBUNTU_VERSION may not be fully tested. Supported versions: 20.04, 22.04, 24.04"
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            ;;
    esac
}

check_nvidia_gpu() {
    print_header "Checking for NVIDIA GPUs"
    
    # Check if lspci is available
    if ! command -v lspci &> /dev/null; then
        print_info "Installing pciutils..."
        sudo apt update
        sudo apt install -y pciutils
    fi
    
    # Look for NVIDIA GPUs
    GPU_COUNT=$(lspci | grep -i nvidia | wc -l)
    
    if [ "$GPU_COUNT" -eq 0 ]; then
        print_error "No NVIDIA GPUs detected!"
        print_info "If you have NVIDIA GPUs, ensure they are properly seated and try:"
        echo "  sudo update-pciids"
        echo "  lspci | grep -i nvidia"
        exit 1
    else
        print_success "Found $GPU_COUNT NVIDIA GPU(s):"
        lspci | grep -i nvidia | while read line; do
            echo "  - $line"
        done
        log "Found $GPU_COUNT NVIDIA GPUs"
    fi
    
    # Show GPU details
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\n${YELLOW}GPU Details (from nvidia-smi):${NC}"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
    fi
}

install_dependencies_ubuntu() {
    print_header "Installing System Dependencies"
    
    log "Updating package list..."
    sudo apt update
    
    log "Installing dependencies..."
    sudo apt install -y \
        wget \
        curl \
        gnupg \
        software-properties-common \
        build-essential \
        cmake \
        git \
        pkg-config \
        libssl-dev \
        libelf-dev \
        libnuma-dev \
        libpciaccess-dev \
        python3-dev \
        python3-pip \
        python3-venv \
        linux-headers-$(uname -r) \
        kmod \
        dkms \
        jq \
        htop \
        nvtop \
        screen \
        tmux \
        vim \
        unzip \
        tar \
        gzip \
        bzip2 \
        linux-tools-common \
        linux-tools-$(uname -r) \
        sysstat \
        lm-sensors \
        || { print_error "Failed to install dependencies"; exit 1; }
    
    print_success "Dependencies installed"
}

install_cuda_ubuntu() {
    print_header "Installing CUDA ${CUDA_VERSION}"
    
    # Map Ubuntu version to CUDA repository name
    case $UBUNTU_VERSION in
        20.04)
            UBUNTU_REPO="ubuntu2004"
            ;;
        22.04)
            UBUNTU_REPO="ubuntu2204"
            ;;
        24.04)
            UBUNTU_REPO="ubuntu2404"
            ;;
        *)
            print_error "Unsupported Ubuntu version: ${UBUNTU_VERSION}"
            exit 1
            ;;
    esac
    
    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        INSTALLED_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1)
        print_warning "CUDA $INSTALLED_VERSION is already installed"
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping CUDA installation"
            return
        fi
    fi
    
    # Remove old CUDA GPG key if exists
    sudo rm -f /usr/share/keyrings/cuda-archive-keyring.gpg
    
    # Download and install CUDA keyring
    log "Adding CUDA repository..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_REPO}/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    
    # Update package list
    sudo apt update
    
    # Install CUDA
    if [ "$INSTALL_DRIVER" = "yes" ]; then
        log "Installing CUDA with drivers..."
        sudo apt install -y cuda-${CUDA_VERSION//./-}
    else
        log "Installing CUDA without drivers..."
        sudo apt install -y cuda-toolkit-${CUDA_VERSION//./-}
    fi
    
    # Add CUDA to PATH
    CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
    if [ -d "$CUDA_PATH" ]; then
        log "Setting up CUDA environment..."
        
        # Add to bashrc
        cat >> ~/.bashrc << EOF

# CUDA ${CUDA_VERSION}
export PATH=${CUDA_PATH}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=${CUDA_PATH}
export CUDA_ROOT=${CUDA_PATH}
EOF
        
        # Also add to current session
        export PATH=${CUDA_PATH}/bin:$PATH
        export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
        export CUDA_HOME=${CUDA_PATH}
        export CUDA_ROOT=${CUDA_PATH}
        
        print_success "CUDA ${CUDA_VERSION} installed"
    else
        print_error "CUDA installation failed - directory not found: $CUDA_PATH"
        exit 1
    fi
    
    # Clean up
    rm -f cuda-keyring*.deb
}

install_nvidia_driver() {
    if [ "$INSTALL_DRIVER" != "yes" ]; then
        print_info "Skipping NVIDIA driver installation as requested"
        return
    fi
    
    print_header "Installing NVIDIA Drivers"
    
    # Check if NVIDIA driver is already loaded
    if lsmod | grep -q nvidia; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        print_warning "NVIDIA driver $DRIVER_VERSION is already loaded"
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping driver installation"
            return
        fi
    fi
    
    # Remove existing NVIDIA drivers
    log "Removing existing NVIDIA drivers..."
    sudo apt remove --purge -y nvidia-* || true
    sudo apt autoremove -y
    
    # Add graphics drivers PPA
    log "Adding graphics drivers PPA..."
    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt update
    
    # Install recommended driver
    log "Installing recommended NVIDIA driver..."
    sudo apt install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    
    # Or install specific driver version
    # sudo apt install -y nvidia-driver-535
    
    print_success "NVIDIA drivers installed"
}

install_cuda_toolkit() {
    print_header "Installing Additional CUDA Tools"
    
    # Install NVIDIA Container Toolkit
    if [ "$INSTALL_CONTAINER_TOOLKIT" = "yes" ]; then
        install_nvidia_container_toolkit
    fi
    
    # Install CUDA samples (optional)
    read -p "Install CUDA samples? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Installing CUDA samples..."
        sudo apt install -y cuda-samples-${CUDA_VERSION//./-}
        
        # Build samples
        if [ -d "/usr/local/cuda-${CUDA_VERSION}/samples" ]; then
            cd "/usr/local/cuda-${CUDA_VERSION}/samples"
            sudo make -j$(nproc)
            print_success "CUDA samples built"
        fi
    fi
    
    # Install CUDA nsight tools
    read -p "Install CUDA profiling tools (nsight)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Installing CUDA profiling tools..."
        sudo apt install -y nvidia-nsight-${CUDA_VERSION//./-}
        print_success "CUDA profiling tools installed"
    fi
}

install_nvidia_container_toolkit() {
    print_header "Installing NVIDIA Container Toolkit"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        if [ "$INSTALL_DOCKER" = "yes" ]; then
            install_docker
        else
            print_warning "Docker not found. Skipping NVIDIA Container Toolkit installation."
            return
        fi
    fi
    
    # Install NVIDIA Container Toolkit
    log "Setting up NVIDIA Container Toolkit repository..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    
    # Configure Docker
    log "Configuring Docker for NVIDIA..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    # Test
    print_info "Testing NVIDIA Container Toolkit..."
    docker run --rm --gpus all nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} nvidia-smi
    
    print_success "NVIDIA Container Toolkit installed"
}

install_docker() {
    print_header "Installing Docker"
    
    if command -v docker &> /dev/null; then
        print_warning "Docker is already installed"
        return
    fi
    
    log "Installing Docker..."
    
    # Remove old versions
    sudo apt remove -y docker docker-engine docker.io containerd runc || true
    
    # Install dependencies
    sudo apt install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Enable and start Docker
    sudo systemctl enable docker
    sudo systemctl start docker
    
    print_success "Docker installed"
    print_info "You may need to log out and back in for docker group changes to take effect"
}

configure_environment() {
    print_header "Configuring Environment"
    
    # Create CUDA environment file
    sudo tee /etc/profile.d/cuda.sh > /dev/null << EOF
#!/bin/bash
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export CUDA_ROOT=/usr/local/cuda-${CUDA_VERSION}
EOF
    
    sudo chmod +x /etc/profile.d/cuda.sh
    
    # Create udev rules for GPU access
    sudo tee /etc/udev/rules.d/99-nvidia.rules > /dev/null << 'EOF'
# NVIDIA GPU devices
SUBSYSTEM=="nvidia", KERNEL=="nvidiactl", GROUP="video", MODE="0660"
SUBSYSTEM=="nvidia", KERNEL=="nvidia-modeset", GROUP="video", MODE="0660"
SUBSYSTEM=="nvidia", KERNEL=="nvidia[0-9]", GROUP="video", MODE="0660"
KERNEL=="nvidia-caps", SUBSYSTEM=="nvidia-caps", GROUP="video", MODE="0660"
EOF
    
    # Reload udev rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    # Add user to video group
    sudo usermod -a -G video $USER
    
    print_success "Environment configured"
}

setup_persistence_mode() {
    if [ "$SETUP_PERSISTENCE" != "yes" ]; then
        return
    fi
    
    print_header "Setting Up GPU Persistence Mode"
    
    # Create systemd service for persistence mode
    sudo tee /etc/systemd/system/nvidia-persistenced.service > /dev/null << 'EOF'
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced
ExecStop=/usr/bin/killall nvidia-persistenced
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
    
    # Create nvidia-persistenced user
    sudo useradd -r -s /sbin/nologin nvidia-persistenced || true
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable nvidia-persistenced.service
    sudo systemctl start nvidia-persistenced.service
    
    # Set persistence mode for all GPUs
    for gpu in $(seq 0 $((GPU_COUNT-1))); do
        sudo nvidia-smi -i $gpu -pm 1
        print_success "GPU $gpu persistence mode enabled"
    done
    
    # Set max performance mode
    sudo nvidia-smi -ac 5001,1590 || print_warning "Could not set max clocks"
    
    print_success "Persistence mode configured"
}

setup_mig() {
    if [ "$SETUP_MIG" != "yes" ]; then
        return
    fi
    
    print_header "Setting Up Multi-Instance GPU (MIG)"
    
    # Check if GPUs support MIG (A100, H100)
    for gpu in $(seq 0 $((GPU_COUNT-1))); do
        MIG_SUPPORT=$(nvidia-smi -i $gpu --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null || echo "N/A")
        
        if [ "$MIG_SUPPORT" != "N/A" ]; then
            print_info "GPU $gpu supports MIG. Configuring..."
            
            # Enable MIG
            sudo nvidia-smi -i $gpu -mig 1
            
            # Create MIG instances (example configuration)
            # 1g.5gb for small models, 2g.10gb for medium, etc.
            sudo nvidia-smi mig -i $gpu -cgi 19,19,19,19  # 4x 1g.5gb instances
            
            print_success "MIG configured on GPU $gpu"
        else
            print_info "GPU $gpu does not support MIG (requires A100/H100)"
        fi
    done
}

setup_nvlink() {
    print_header "Checking NVLink Status"
    
    if nvidia-smi nvlink --status &> /dev/null; then
        print_info "NVLink status:"
        nvidia-smi nvlink --status
    else
        print_info "NVLink not available or not configured"
    fi
}

setup_gpu_monitoring() {
    print_header "Setting Up GPU Monitoring"
    
    # Install nvtop for interactive monitoring
    if ! command -v nvtop &> /dev/null; then
        log "Installing nvtop..."
        sudo apt install -y nvtop
        print_success "nvtop installed"
    fi
    
    # Install DCGM for advanced monitoring
    log "Installing NVIDIA DCGM..."
    sudo apt install -y datacenter-gpu-manager
    sudo systemctl enable nvidia-dcgm
    sudo systemctl start nvidia-dcgm
    
    # Install Prometheus GPU metrics exporter
    read -p "Install Prometheus GPU metrics exporter? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Installing dcgm-exporter..."
        
        # Run DCGM exporter in Docker
        docker run -d \
            --name dcgm-exporter \
            --restart unless-stopped \
            --gpus all \
            -p 9400:9400 \
            nvcr.io/nvidia/k8s/dcgm-exporter:latest
        
        print_success "DCGM exporter running on port 9400"
        print_info "Add to prometheus.yml:"
        echo "  - job_name: 'dcgm'"
        echo "    static_configs:"
        echo "      - targets: ['localhost:9400']"
    fi
}

verify_cuda() {
    print_header "Verifying CUDA Installation"
    
    # Check nvcc
    if command -v nvcc &> /dev/null; then
        echo -e "\n${YELLOW}nvcc version:${NC}"
        nvcc --version | head -3
    else
        print_error "nvcc not found in PATH"
        echo "Try: source ~/.bashrc"
    fi
    
    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\n${YELLOW}nvidia-smi output:${NC}"
        nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,memory.used,temperature.gpu,power.draw --format=csv
    else
        print_error "nvidia-smi not found"
    fi
    
    # Check CUDA samples
    if [ -d "/usr/local/cuda-${CUDA_VERSION}/samples" ]; then
        echo -e "\n${YELLOW}CUDA samples location:${NC}"
        echo "  /usr/local/cuda-${CUDA_VERSION}/samples"
    fi
    
    # Check deviceQuery
    if [ -f "/usr/local/cuda-${CUDA_VERSION}/samples/1_Utilities/deviceQuery/deviceQuery" ]; then
        echo -e "\n${YELLOW}Running deviceQuery:${NC}"
        /usr/local/cuda-${CUDA_VERSION}/samples/1_Utilities/deviceQuery/deviceQuery | grep "Detected" || true
    fi
}

test_pytorch() {
    print_header "Testing PyTorch with CUDA"
    
    # Create virtual environment
    python3 -m venv cuda_test_env
    source cuda_test_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    log "Installing PyTorch for CUDA ${CUDA_VERSION}..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}1
    
    # Test script
    python3 << 'EOF'
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
    
    # Simple tensor test
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✅ Tensor test successful: {z.shape}")
        
        # Benchmark
        import time
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        print(f"✅ 100 matrix multiplications: {elapsed:.2f} ms")
        
    except Exception as e:
        print(f"❌ Tensor test failed: {e}")
else:
    print("❌ CUDA not available in PyTorch")
    print("Check installation: https://pytorch.org/get-started/locally/")
EOF
    
    deactivate
    rm -rf cuda_test_env
    
    print_success "PyTorch test complete"
}

test_tensorflow() {
    print_header "Testing TensorFlow with CUDA"
    
    # Create virtual environment
    python3 -m venv tf_test_env
    source tf_test_env/bin/activate
    
    # Install TensorFlow
    log "Installing TensorFlow..."
    pip install tensorflow[and-cuda]
    
    # Test script
    python3 << 'EOF'
import tensorflow as tf
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {tf.test.is_gpu_available(cuda_only=True)}")
print(f"GPU list: {tf.config.list_physical_devices('GPU')}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
        details = tf.sysconfig.get_build_info()
        print(f"  - CUDA version: {details.get('cuda_version', 'N/A')}")
        print(f"  - cuDNN version: {details.get('cudnn_version', 'N/A')}")
    
    # Simple tensor test
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print(f"✅ Tensor test successful: {c.shape}")
    except Exception as e:
        print(f"❌ Tensor test failed: {e}")
else:
    print("❌ No GPUs found in TensorFlow")
EOF
    
    deactivate
    rm -rf tf_test_env
    
    print_success "TensorFlow test complete"
}

test_jax() {
    print_header "Testing JAX with CUDA"
    
    # Create virtual environment
    python3 -m venv jax_test_env
    source jax_test_env/bin/activate
    
    # Install JAX
    log "Installing JAX..."
    pip install --upgrade pip
    pip install jax[cuda12]
    
    # Test script
    python3 << 'EOF'
import jax
import jax.numpy as jnp
import sys

print(f"Python version: {sys.version}")
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

devices = jax.devices()
if any('gpu' in str(d).lower() for d in devices):
    print("✅ GPU devices found:")
    for d in devices:
        if 'gpu' in str(d).lower():
            print(f"  - {d}")
    
    # Simple test
    try:
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1000, 1000))
        y = jax.random.normal(key, (1000, 1000))
        z = jnp.dot(x, y)
        z.block_until_ready()
        print(f"✅ Tensor test successful: {z.shape}")
    except Exception as e:
        print(f"❌ Tensor test failed: {e}")
else:
    print("❌ No GPU devices found in JAX")
EOF
    
    deactivate
    rm -rf jax_test_env
    
    print_success "JAX test complete"
}

test_burn() {
    print_header "Testing Burn with CUDA"
    
    # Create test project
    mkdir -p burn_test
    cd burn_test
    
    # Create Cargo.toml
    cat > Cargo.toml << 'EOF'
[package]
name = "burn-test"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { version = "0.19", features = ["cuda", "train"] }
burn-tensor = { version = "0.19", features = ["cuda"] }
tokio = { version = "1.35", features = ["full"] }
rand = "0.8"
EOF
    
    # Create source file
    mkdir -p src
    cat > src/main.rs << 'EOF'
use burn::{
    backend::CudaBackend,
    tensor::{Tensor, Distribution},
    module::Module,
    nn::{
        Linear, LinearConfig,
        Dropout, DropoutConfig,
    },
};

type Backend = CudaBackend;

#[derive(Module, Debug)]
struct TestModel {
    linear1: Linear<Backend>,
    linear2: Linear<Backend>,
    dropout: Dropout,
}

impl TestModel {
    fn new(device: &Backend::Device) -> Self {
        Self {
            linear1: LinearConfig::new(512, 256)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(256, 128)
                .with_bias(true)
                .init(device),
            dropout: DropoutConfig::new(0.1).init(),
        }
    }
    
    fn forward(&self, input: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        let x = self.linear1.forward(input);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Burn with CUDA test");
    
    // Get available devices
    let devices = Backend::devices();
    println!("📊 Found {} CUDA device(s)", devices.len());
    
    for (i, device) in devices.iter().enumerate() {
        println!("  GPU {}: {:?}", i, device);
    }
    
    if devices.is_empty() {
        println!("❌ No CUDA devices found!");
        return Ok(());
    }
    
    let device = &devices[0];
    
    // Create tensors
    println!("\n📊 Creating tensors...");
    let tensor1 = Tensor::<Backend, 2>::random([1000, 512], Distribution::Default, device);
    let tensor2 = Tensor::<Backend, 2>::random([512, 1000], Distribution::Default, device);
    
    // Matrix multiplication
    println!("🔄 Running matrix multiplication...");
    let start = std::time::Instant::now();
    let result = tensor1.matmul(tensor2);
    let duration = start.elapsed();
    
    println!("✅ Matrix multiplication successful!");
    println!("   Result shape: {:?}", result.dims());
    println!("   Time: {:?}", duration);
    
    // Test model
    println!("\n🤖 Testing neural network...");
    let model = TestModel::new(device);
    let input = Tensor::<Backend, 2>::random([32, 512], Distribution::Default, device);
    let output = model.forward(input);
    println!("✅ Forward pass successful!");
    println!("   Output shape: {:?}", output.dims());
    
    // Memory info
    if let Ok(used) = Backend::memory_used(device) {
        println!("\n💾 GPU Memory used: {:.2} MB", used as f64 / 1_000_000.0);
    }
    
    println!("\n✅ Burn test completed successfully!");
    Ok(())
}
EOF
    
    # Build and run
    log "Building Burn test..."
    cargo build --release
    
    log "Running Burn test..."
    cargo run --release
    
    cd ..
    rm -rf burn_test
    
    print_success "Burn test complete"
}

setup_cuda_profiling() {
    print_header "Setting Up CUDA Profiling Tools"
    
    # Install NVIDIA NSight Systems
    log "Installing NVIDIA NSight Systems..."
    sudo apt install -y nsight-systems
    
    # Install NVIDIA NSight Compute
    log "Installing NVIDIA NSight Compute..."
    sudo apt install -y nsight-compute
    
    # Install CUPTI (CUDA Profiling Tools Interface)
    log "Installing CUPTI..."
    sudo apt install -y cuda-cupti-${CUDA_VERSION//./-}
    
    # Add CUPTI to library path
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/extras/CUPTI/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    
    # Install nvidia-ml-py for Python monitoring
    pip3 install nvidia-ml-py3
    
    print_success "Profiling tools installed"
    print_info "Usage examples:"
    echo "  nsys profile --gpu-metrics-device=0 ./your_app"
    echo "  ncu --metrics sm__throughput.avg.pct ./your_app"
}

setup_cuda_visualization() {
    print_header "Setting Up Visualization Tools"
    
    # Install CUDA-GDB
    log "Installing CUDA-GDB..."
    sudo apt install -y cuda-gdb-${CUDA_VERSION//./-}
    
    # Install CUDA-MEMCHECK
    log "Installing CUDA-MEMCHECK..."
    sudo apt install -y cuda-memcheck-${CUDA_VERSION//./-}
    
    # Install compute-sanitizer (newer version of memcheck)
    sudo apt install -y cuda-sanitizer-${CUDA_VERSION//./-}
    
    # Install NVIDIA Visual Profiler (deprecated, use nsight instead)
    # sudo apt install -y nvidia-visual-profiler
    
    print_success "Debugging tools installed"
}

configure_cuda_limits() {
    print_header "Configuring System Limits for CUDA"
    
    # Increase locked memory limit
    sudo tee -a /etc/security/limits.conf << EOF

# CUDA/GPU limits
* soft memlock unlimited
* hard memlock unlimited
* soft stack unlimited
* hard stack unlimited
* soft nofile 1048576
* hard nofile 1048576
EOF
    
    # Increase kernel parameters for GPU
    sudo tee -a /etc/sysctl.conf << EOF

# GPU performance settings
vm.max_map_count = 262144
kernel.numa_balancing = 0
EOF
    
    sudo sysctl -p
    
    print_success "System limits configured"
}

create_cuda_swap() {
    print_header "Creating GPU Swap Space (Emergency Only)"
    
    read -p "Create GPU swap space? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi
    
    print_warning "GPU swap will significantly reduce performance!"
    print_warning "Use only as temporary emergency measure."
    
    # Create swap file
    sudo fallocate -l 32G /swapfile_gpu
    sudo chmod 600 /swapfile_gpu
    sudo mkswap /swapfile_gpu
    
    # Enable swap
    sudo swapon /swapfile_gpu
    
    # Add to fstab
    echo "/swapfile_gpu none swap sw 0 0" | sudo tee -a /etc/fstab
    
    # Configure system to use swap
    sudo sysctl vm.swappiness=60
    
    print_success "GPU swap created (use with caution!)"
}

generate_cuda_report() {
    print_header "Generating CUDA System Report"
    
    REPORT_FILE="$HOME/cuda-system-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "========================================="
        echo "CUDA System Report"
        echo "Generated: $(date)"
        echo "========================================="
        echo
        
        echo "System Information:"
        echo "-------------------"
        uname -a
        lsb_release -a 2>/dev/null || true
        echo
        
        echo "CPU Information:"
        echo "----------------"
        lscpu | grep "Model name\|CPU(s)\|Thread(s) per core"
        echo
        
        echo "Memory Information:"
        echo "-------------------"
        free -h
        echo
        
        echo "NVIDIA Driver Information:"
        echo "--------------------------"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
        echo
        
        echo "CUDA Version:"
        echo "-------------"
        nvcc --version | grep "release" || echo "nvcc not found"
        echo
        
        echo "GPU Information:"
        echo "----------------"
        nvidia-smi --query-gpu=index,name,memory.total,compute_cap,power.limit,clocks.max.sm,clocks.max.memory --format=csv
        echo
        
        echo "GPU Details:"
        echo "------------"
        for gpu in $(seq 0 $((GPU_COUNT-1))); do
            echo "GPU $gpu:"
            nvidia-smi -i $gpu -q | grep -E "Product Name|Display Mode|Persistence Mode|MIG Mode|Serial|VBIOS Version|Inforom Version" | sed 's/^/  /'
        done
        echo
        
        echo "NVLink Status:"
        echo "--------------"
        nvidia-smi nvlink --status 2>/dev/null || echo "NVLink not available"
        echo
        
        echo "CUDA Environment:"
        echo "-----------------"
        env | grep -E "CUDA|NVIDIA" | sort
        echo
        
        echo "CUDA Libraries:"
        echo "---------------"
        ldconfig -p | grep -E "libcuda|libcudart|libcublas|libcudnn" | head -20
        echo
        
        echo "Docker Configuration:"
        echo "---------------------"
        docker info | grep -E "Runtimes|Default Runtime" 2>/dev/null || echo "Docker not installed"
        echo
        
        echo "========================================="
        echo "End of Report"
        echo "========================================="
        
    } > "$REPORT_FILE"
    
    print_success "Report generated: $REPORT_FILE"
    cat "$REPORT_FILE"
}

cleanup() {
    print_header "Cleaning Up"
    
    # Remove temporary files
    rm -f cuda-keyring*.deb
    rm -f *.deb
    
    # Clean package cache
    sudo apt autoremove -y
    sudo apt autoclean
    
    print_success "Cleanup complete"
}

print_summary() {
    print_header "Setup Complete! 🎉"
    
    echo -e "${GREEN}CUDA ${CUDA_VERSION} has been installed successfully!${NC}"
    echo ""
    
    echo -e "${YELLOW}📊 Installation Summary:${NC}"
    echo "  • CUDA Version: $CUDA_VERSION"
    echo "  • GPUs Detected: $GPU_COUNT"
    echo "  • Driver Installed: $INSTALL_DRIVER"
    echo "  • Docker Support: $INSTALL_DOCKER"
    echo "  • Container Toolkit: $INSTALL_CONTAINER_TOOLKIT"
    echo "  • Persistence Mode: $SETUP_PERSISTENCE"
    echo "  • MIG Enabled: $SETUP_MIG"
    echo ""
    
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    echo "1. Log out and log back in for group changes to take effect"
    echo "2. Source your bashrc: source ~/.bashrc"
    echo "3. Verify CUDA installation: nvcc --version"
    echo "4. Check GPU status: nvidia-smi"
    echo "5. Run PyTorch test: python3 -c \"import torch; print(torch.cuda.is_available())\""
    echo ""
    
    echo -e "${YELLOW}🛠️  Useful Commands:${NC}"
    echo "  nvidia-smi              - Monitor GPU status"
    echo "  nvtop                   - Interactive GPU monitor"
    echo "  nvidia-smi nvlink --status - Check NVLink status"
    echo "  nvidia-smi -pm 1        - Enable persistence mode"
    echo "  nvidia-smi -ac 5001,1590 - Set max clocks"
    echo "  nvidia-smi mig -i 0 -cgi 19 - Create MIG instance"
    echo ""
    
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo -e "${YELLOW}🔗 Multi-GPU Configuration:${NC}"
        echo "  GPU IDs to use in worker config: $GPU_IDS"
        echo "  Example worker.toml:"
        echo "    [gpu]"
        echo "    device_ids = [0, 1, 2, 3]"
        echo "    enable_peer_access = true"
        echo ""
    fi
    
    echo -e "${YELLOW}📚 Documentation:${NC}"
    echo "  • CUDA Docs: https://docs.nvidia.com/cuda/"
    echo "  • PyTorch: https://pytorch.org/get-started/locally/"
    echo "  • AI Cluster: docs/ directory"
    echo ""
    
    echo -e "${BLUE}Happy AI clustering with NVIDIA GPUs! 🚀${NC}"
}

main() {
    print_header "CUDA Setup Script for AI Cluster"
    echo "Version: 1.0.0"
    echo "Log file: $LOG_FILE"
    echo ""
    
    check_root
    detect_os
    check_nvidia_gpu
    
    # Install dependencies
    install_dependencies_ubuntu
    
    # Install NVIDIA driver
    install_nvidia_driver
    
    # Install CUDA
    install_cuda_ubuntu
    
    # Install additional tools
    install_cuda_toolkit
    
    # Configure environment
    configure_environment
    
    # Setup GPU features
    setup_persistence_mode
    setup_mig
    setup_nvlink
    setup_gpu_monitoring
    
    # Configure system
    configure_cuda_limits
    
    # Setup profiling (optional)
    read -p "Install CUDA profiling tools? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_cuda_profiling
    fi
    
    # Generate report
    generate_cuda_report
    
    # Run tests
    read -p "Run CUDA verification tests? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        verify_cuda
        test_pytorch
        read -p "Run TensorFlow tests? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            test_tensorflow
        fi
        read -p "Run JAX tests? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            test_jax
        fi
        read -p "Run Burn tests? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            test_burn
        fi
    fi
    
    # Create swap (optional, emergency only)
    read -p "Create emergency GPU swap? (NOT RECOMMENDED) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_cuda_swap
    fi
    
    # Cleanup
    cleanup
    
    print_summary
    
    echo -e "\n${YELLOW}⚠️  Important: You may need to log out and back in for all changes to take effect.${NC}"
    echo -e "${YELLOW}📋 Full log saved to: $LOG_FILE${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --no-driver)
            INSTALL_DRIVER="no"
            shift
            ;;
        --no-docker)
            INSTALL_DOCKER="no"
            shift
            ;;
        --no-container-toolkit)
            INSTALL_CONTAINER_TOOLKIT="no"
            shift
            ;;
        --no-persistence)
            SETUP_PERSISTENCE="no"
            shift
            ;;
        --enable-mig)
            SETUP_MIG="yes"
            shift
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --debug)
            SETUP_DEBUG="yes"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda-version VERSION    CUDA version to install (default: 12.1)"
            echo "  --no-driver                Skip NVIDIA driver installation"
            echo "  --no-docker                 Skip Docker installation"
            echo "  --no-container-toolkit      Skip NVIDIA Container Toolkit"
            echo "  --no-persistence            Skip persistence mode setup"
            echo "  --enable-mig                 Enable Multi-Instance GPU (A100/H100)"
            echo "  --gpu-ids IDS                Comma-separated GPU IDs (default: 0)"
            echo "  --debug                      Enable debug output"
            echo "  --help                       Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main