#!/bin/bash
# ROCm Setup Script for AMD GPUs
# ===============================
# This script installs and configures ROCm for AMD GPUs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROCM_VERSION="${ROCM_VERSION:-6.0}"
INSTALL_DIR="${INSTALL_DIR:-/opt/rocm}"
GPU_IDS="${GPU_IDS:-0}"  # Comma-separated list of GPU IDs to use

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

check_root() {
    if [ "$EUID" -eq 0 ]; then 
        print_error "Please don't run this script as root"
        exit 1
    fi
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        print_error "Cannot detect OS"
        exit 1
    fi
}

install_dependencies_ubuntu() {
    print_header "Installing Dependencies for Ubuntu"
    
    sudo apt update
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
        libdrm-dev \
        libnuma-dev \
        libpciaccess-dev \
        python3-dev \
        python3-pip \
        python3-venv \
        linux-headers-$(uname -r) \
        kmod
        
    print_success "Dependencies installed"
}

install_rocm_ubuntu() {
    print_header "Installing ROCm ${ROCM_VERSION}"
    
    # Add ROCm repository
    print_info "Adding ROCm repository..."
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    
    if [ "$VERSION" = "22.04" ]; then
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
    elif [ "$VERSION" = "20.04" ]; then
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} focal main" | sudo tee /etc/apt/sources.list.d/rocm.list
    else
        print_error "Unsupported Ubuntu version: ${VERSION}"
        exit 1
    fi
    
    # Add user to video and render groups
    sudo usermod -a -G video $USER
    sudo usermod -a -G render $USER
    
    # Install ROCm
    print_info "Installing ROCm packages..."
    sudo apt update
    sudo apt install -y \
        rocm-dev \
        rocm-libs \
        rocm-dkms \
        rocm-utils \
        hip-dev \
        hip-runtime-amd \
        miopen-hip \
        rccl \
        rocblas \
        rocrand \
        rocthrust \
        rocprim
        
    print_success "ROCm installed"
}

install_docker_support() {
    print_header "Installing Docker Support"
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        print_info "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
    
    # Install ROCm Docker runtime
    sudo apt install -y rocm-docker
    
    # Pull ROCm Docker image
    docker pull rocm/dev-ubuntu-22.04:${ROCM_VERSION}
    
    print_success "Docker support installed"
}

configure_environment() {
    print_header "Configuring Environment"
    
    # Add ROCm to PATH
    ROCM_LINE="export PATH=$INSTALL_DIR/bin:$INSTALL_DIR/llvm/bin:\$PATH"
    LD_LIBRARY_LINE="export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$INSTALL_DIR/lib64:\$LD_LIBRARY_PATH"
    
    if ! grep -q "$INSTALL_DIR/bin" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# ROCm" >> ~/.bashrc
        echo "$ROCM_LINE" >> ~/.bashrc
        echo "$LD_LIBRARY_LINE" >> ~/.bashrc
    fi
    
    # Create device persistence script
    cat > ~/enable_rocm_persistence.sh << 'EOF'
#!/bin/bash
# Enable ROCm persistence for all GPUs
for gpu in /sys/class/kfd/kfd/topology/nodes/*/gpu_id; do
    if [ -f "$gpu" ]; then
        gpu_id=$(cat $gpu)
        echo "manual" | sudo tee /sys/class/kfd/kfd/topology/nodes/$gpu_id/power_dpm_force_performance_level
    fi
done
EOF
    
    chmod +x ~/enable_rocm_persistence.sh
    
    print_success "Environment configured"
}

verify_gpus() {
    print_header "Verifying GPU Detection"
    
    # Check if ROCm is in PATH
    export PATH=$INSTALL_DIR/bin:$PATH
    
    # Run rocminfo
    if command -v rocminfo &> /dev/null; then
        echo -e "\n${YELLOW}rocminfo output:${NC}"
        rocminfo | grep "Name:" | head -5
    else
        print_error "rocminfo not found"
    fi
    
    # Run rocm-smi
    if command -v rocm-smi &> /dev/null; then
        echo -e "\n${YELLOW}rocm-smi output:${NC}"
        rocm-smi --showallinfo
    else
        print_error "rocm-smi not found"
    fi
    
    # Check GPU count
    GPU_COUNT=$(rocm-smi --showid | grep "GPU\[" | wc -l)
    print_info "Detected ${GPU_COUNT} GPU(s)"
    
    # Check specified GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    for gpu in "${GPU_ARRAY[@]}"; do
        if [ "$gpu" -ge "$GPU_COUNT" ]; then
            print_error "GPU $gpu specified but only $GPU_COUNT GPUs available"
            exit 1
        fi
    done
    
    print_success "GPU verification complete"
}

test_pytorch() {
    print_header "Testing PyTorch with ROCm"
    
    # Create virtual environment
    python3 -m venv rocm_test_env
    source rocm_test_env/bin/activate
    
    # Install PyTorch for ROCm
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION//./}
    
    # Test script
    python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Simple tensor test
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"Tensor test successful: {z.shape}")
EOF
    
    deactivate
    rm -rf rocm_test_env
    
    print_success "PyTorch test complete"
}

test_burn() {
    print_header "Testing Burn with ROCm"
    
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
burn = { version = "0.19", features = ["hip"] }
burn-tensor = { version = "0.19", features = ["hip"] }
tokio = { version = "1.35", features = ["full"] }
EOF
    
    # Create source file
    mkdir -p src
    cat > src/main.rs << 'EOF'
use burn::tensor::{Tensor, backend::Backend};
use burn::backend::HipBackend;

type Backend = HipBackend;

fn main() {
    println!("Burn with ROCm test");
    
    let device = Backend::device(0);
    println!("Using device: {:?}", device);
    
    // Create tensors
    let tensor1 = Tensor::<Backend, 2>::random([100, 100], burn::tensor::Distribution::Default, &device);
    let tensor2 = Tensor::<Backend, 2>::random([100, 100], burn::tensor::Distribution::Default, &device);
    
    // Matrix multiplication
    let result = tensor1.matmul(tensor2);
    println!("Tensor shape: {:?}", result.dims());
    
    println!("Burn test successful!");
}
EOF
    
    # Build and run
    cargo run --release
    
    cd ..
    rm -rf burn_test
    
    print_success "Burn test complete"
}

setup_persistence() {
    print_header "Setting Up GPU Persistence"
    
    # Create systemd service for GPU persistence
    sudo tee /etc/systemd/system/rocm-persistence.service > /dev/null << 'EOF'
[Unit]
Description=ROCm GPU Persistence
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/rocm-smi -d 0 --setpoweroverdriveenable 1
ExecStart=/usr/bin/rocm-smi -d 0 --setprofile 0
User=root

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable rocm-persistence.service
    sudo systemctl start rocm-persistence.service
    
    print_success "Persistence service configured"
}

setup_firewall() {
    print_header "Configuring Firewall"
    
    # Open ports for cluster communication
    if command -v ufw &> /dev/null; then
        sudo ufw allow 50051/tcp comment "gRPC worker communication"
        sudo ufw allow 9091/tcp comment "Worker metrics"
        sudo ufw allow 8000/tcp comment "Coordinator API"
        sudo ufw allow 9090/tcp comment "Coordinator metrics"
        print_success "Firewall rules added"
    else
        print_info "UFW not installed, skipping firewall configuration"
    fi
}

print_summary() {
    print_header "Setup Complete! 🎉"
    
    echo -e "${GREEN}ROCm ${ROCM_VERSION} has been installed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Log out and log back in for group changes to take effect"
    echo "2. Verify GPU access: rocm-smi"
    echo "3. Run the test scripts to verify everything works"
    echo ""
    echo -e "${YELLOW}Useful commands:${NC}"
    echo "  rocm-smi           - Monitor GPU status"
    echo "  rocminfo           - Display ROCm info"
    echo "  hipconfig          - Show HIP configuration"
    echo "  ~/enable_rocm_persistence.sh - Enable GPU persistence"
    echo ""
    echo -e "${YELLOW}GPU IDs to use in worker config: ${GPU_IDS}${NC}"
    echo ""
    echo -e "${BLUE}Happy AI clustering! 🚀${NC}"
}

main() {
    print_header "ROCm Setup Script for AI Cluster"
    
    check_root
    detect_os
    
    case $OS in
        ubuntu)
            install_dependencies_ubuntu
            install_rocm_ubuntu
            ;;
        *)
            print_error "Unsupported OS: $OS"
            exit 1
            ;;
    esac
    
    install_docker_support
    configure_environment
    verify_gpus
    setup_persistence
    setup_firewall
    
    # Export ROCm path for current session
    export PATH=$INSTALL_DIR/bin:$PATH
    
    # Run tests
    test_pytorch
    test_burn
    
    print_summary
    
    echo -e "\n${YELLOW}Note: You may need to log out and back in for all changes to take effect.${NC}"
}

# Run main function
main