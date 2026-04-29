# Agent Lightning POC Setup Guide

Complete setup guide for integrating Microsoft Agent Lightning with Infosys Agentic Foundry.

**Hardware Requirements:** 4x NVIDIA A100 80GB GPUs ✅

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Agent Lightning Installation](#agent-lightning-installation)
4. [MongoDB Setup](#mongodb-setup)
5. [GPU Configuration](#gpu-configuration)
6. [POC Installation](#poc-installation)
7. [Running First Training](#running-first-training)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

---

## 1. Prerequisites

### System Requirements

```bash
# Operating System
Ubuntu 20.04+ or Rocky Linux 8+

# Python
Python 3.10 or 3.11 (required)

# CUDA
CUDA 12.1+ (for A100 support)
cuDNN 8.9+

# GPU
4x NVIDIA A100 80GB (confirmed ✅)
NVIDIA Driver 535+ (for CUDA 12.1)
```

### Verify GPU Setup

```bash
# Check GPUs are detected
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.XX       Driver Version: 535.XX       CUDA Version: 12.1    |
# |-------------------------------+----------------------+----------------------+
# |   0  NVIDIA A100-SXM... Off  |   ...                | 00000000:00:07.0 Off |
# |   1  NVIDIA A100-SXM... Off  |   ...                | 00000000:00:08.0 Off |
# |   2  NVIDIA A100-SXM... Off  |   ...                | 00000000:00:09.0 Off |
# |   3  NVIDIA A100-SXM... Off  |   ...                | 00000000:00:0A.0 Off |
# +-----------------------------------------------------------------------------+

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Expected: CUDA available: True, GPUs: 4
```

---

## 2. Environment Setup

### Create Python Virtual Environment

```bash
# Navigate to Foundry repository
cd /path/to/Infosys-Agentic-Foundry

# Create virtual environment
python3.10 -m venv venv_agent_lightning

# Activate environment
source venv_agent_lightning/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Set Environment Variables

```bash
# Add to ~/.bashrc or create .env file
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# PyTorch CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# NCCL for multi-GPU communication
export NCCL_DEBUG=INFO  # Set to WARN in production
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available

# Agent Lightning specific
export AGENTLIGHTNING_STORE_TYPE=mongo
export AGENTLIGHTNING_MONGO_URI=mongodb://localhost:27017

# Reload
source ~/.bashrc
```

---

## 3. Agent Lightning Installation

### Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.3+ with CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Expected: PyTorch: 2.3.0+cu121, CUDA: 12.1
```

### Install Agent Lightning

```bash
# Clone Agent Lightning repository
cd /home/user
git clone https://github.com/microsoft/agent-lightning.git
cd agent-lightning

# Install with VERL support (includes vLLM, flash-attn)
pip install -e ".[verl]"

# This installs:
# - agentlightning core
# - verl (RL algorithms)
# - vllm (fast inference)
# - flash-attn (memory-efficient attention)
# - transformers, accelerate, etc.

# Verify installation
python -c "from agentlightning import Trainer; from agentlightning.algorithm.verl import VERL; print('Agent Lightning installed successfully!')"
```

### Install Additional Dependencies

```bash
# MongoDB driver
pip install pymongo motor

# LangChain (for Foundry integration)
pip install langchain langchain-openai langchain-community

# Monitoring and visualization
pip install wandb tensorboard

# Development tools
pip install pytest black flake8 mypy
```

---

## 4. MongoDB Setup

Agent Lightning uses MongoDB for persistent storage of rollouts, spans, and checkpoints.

### Install MongoDB

**Option A: Docker (Recommended for Development)**

```bash
# Pull MongoDB image
docker pull mongo:7.0

# Run MongoDB container
docker run -d \
  --name agent-lightning-mongo \
  -p 27017:27017 \
  -v /data/mongodb:/data/db \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=your_secure_password \
  mongo:7.0

# Verify MongoDB is running
docker ps | grep agent-lightning-mongo

# Test connection
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017'); print(f'MongoDB connected: {client.server_info()}')"
```

**Option B: Native Installation**

```bash
# On Ubuntu
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify
sudo systemctl status mongod
```

### Create Database and User

```bash
# Connect to MongoDB
mongosh

# Create database and user
use agent_lightning_training

db.createUser({
  user: "agent_trainer",
  pwd: "your_secure_password",
  roles: [
    { role: "readWrite", db: "agent_lightning_training" }
  ]
})

# Exit mongosh
exit
```

### Update MongoDB URI

```bash
# Add to .env or ~/.bashrc
export AGENTLIGHTNING_MONGO_URI="mongodb://agent_trainer:your_secure_password@localhost:27017/agent_lightning_training"
```

---

## 5. GPU Configuration

### Optimize NVIDIA Settings

```bash
# Enable persistence mode (prevents driver unload)
sudo nvidia-smi -pm 1

# Set power limit (optional, for power efficiency)
# A100 max is 400W, can reduce to 300W for training
sudo nvidia-smi -pl 300  # Watts

# Enable Multi-Instance GPU (MIG) mode if needed
# (Not recommended for RL training - use full GPUs)
sudo nvidia-smi -mig 0  # Disable MIG

# Verify settings
nvidia-smi
```

### Test Multi-GPU Training

```bash
# Create test script
cat > test_multi_gpu.py << 'EOF'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def test_multi_gpu():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"✅ Found {n_gpus} GPUs")

        # Test each GPU
        for i in range(n_gpus):
            device = torch.device(f"cuda:{i}")
            x = torch.randn(1000, 1000, device=device)
            y = torch.matmul(x, x)
            print(f"✅ GPU {i}: Matrix multiplication successful")

        # Test NCCL backend
        if n_gpus > 1:
            print("\nTesting NCCL backend...")
            try:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                dist.init_process_group(backend='nccl', world_size=1, rank=0)
                print("✅ NCCL backend initialized successfully")
                dist.destroy_process_group()
            except Exception as e:
                print(f"❌ NCCL test failed: {e}")

        print("\n✅ All GPU tests passed!")
    else:
        print("❌ CUDA not available")

if __name__ == "__main__":
    test_multi_gpu()
EOF

python test_multi_gpu.py
```

---

## 6. POC Installation

### Install POC Package

```bash
# Navigate to POC directory
cd /home/user/Infosys-Agentic-Foundry/poc_agent_lightning

# Install in editable mode
pip install -e .

# Verify installation
python -c "from src.foundry_lit_agent_adapter import FoundryLitAgentAdapter; print('POC installed successfully!')"
```

### Verify Foundry Integration

```bash
# Test Foundry services are accessible
python << 'EOF'
import sys
sys.path.append('/home/user/Infosys-Agentic-Foundry/Infosys-Agentic-Foundry-Backend')

from src.api.dependencies import ServiceProvider
from src.database.services import AgentService

# This should work if Foundry backend is running
# If not, start Foundry backend first

print("✅ Foundry integration ready")
EOF
```

---

## 7. Running First Training

### Step 1: Prepare Training Dataset

```bash
# Create example dataset for SQL agent
cat > example_dataset.json << 'EOF'
[
  {
    "query": "Find all customers in California",
    "expected_sql": "SELECT * FROM customers WHERE state = 'CA'",
    "db_schema": {"customers": ["id", "name", "state", "email"]}
  },
  {
    "query": "Count orders by customer",
    "expected_sql": "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id",
    "db_schema": {"orders": ["order_id", "customer_id", "total"]}
  },
  {
    "query": "Get top 10 products by sales",
    "expected_sql": "SELECT product_id, SUM(quantity * price) as total_sales FROM order_items GROUP BY product_id ORDER BY total_sales DESC LIMIT 10",
    "db_schema": {"order_items": ["order_id", "product_id", "quantity", "price"]}
  }
]
EOF
```

### Step 2: Train with APO (No GPU Required)

```bash
# Run APO training (quick validation)
python examples/train_react_agent_apo.py \
  --agent-id <your_agent_id> \
  --dataset-file example_dataset.json \
  --reward-function sql_correctness \
  --n-iterations 3 \
  --beam-width 2 \
  --n-rollouts-per-prompt 5 \
  --n-runners 4

# Expected runtime: 30-60 minutes for small dataset
# Output: Optimized prompts in ./apo_output/
```

### Step 3: Train with VERL (GPU Required)

```bash
# Run VERL training on 4x A100
python examples/train_react_agent_verl.py \
  --agent-id <your_agent_id> \
  --dataset-file example_dataset.json \
  --reward-function sql_correctness \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --total-epochs 5 \
  --train-batch-size 64 \
  --n-runners 8

# Expected runtime: 2-4 hours for small dataset, 8-12 hours for full training
# Output: Model checkpoints in ./verl_output/
```

### Step 4: Monitor Training

```bash
# Terminal 1: Watch GPU utilization
watch -n 1 nvidia-smi

# Terminal 2: TensorBoard (if enabled)
tensorboard --logdir ./verl_output/logs

# Terminal 3: W&B (if configured)
wandb login
# Then access dashboard at wandb.ai

# Terminal 4: MongoDB logs
docker logs -f agent-lightning-mongo
```

---

## 8. Troubleshooting

### Issue: CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

```bash
# 1. Reduce batch size
# Edit configs/verl_4xa100_config.yaml:
data:
  train_batch_size: 32  # Reduce from 64

# 2. Enable gradient checkpointing
optimizations:
  gradient_checkpointing: true

# 3. Reduce sequence lengths
data:
  max_prompt_length: 2048  # Reduce from 4096
  max_response_length: 1024  # Reduce from 2048

# 4. Use smaller model
actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-1.5B-Instruct"  # Instead of 7B
```

### Issue: MongoDB Connection Failed

**Symptoms:**
```
pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused
```

**Solutions:**

```bash
# Check MongoDB status
sudo systemctl status mongod

# Restart MongoDB
sudo systemctl restart mongod

# Check connection
mongosh --eval "db.serverStatus()"

# Verify environment variable
echo $AGENTLIGHTNING_MONGO_URI
```

### Issue: NCCL Initialization Failed

**Symptoms:**
```
RuntimeError: NCCL error in: /path/to/file.cpp:XXX, invalid usage
```

**Solutions:**

```bash
# 1. Check NCCL installation
python -c "import torch; print(torch.cuda.nccl.version())"

# 2. Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Or your network interface
export NCCL_IB_DISABLE=1  # If no InfiniBand

# 3. Test NCCL
python -m torch.distributed.run --nproc_per_node=4 test_multi_gpu.py
```

### Issue: vLLM Startup Slow

**Symptoms:**
Training hangs at "Initializing vLLM..."

**Solutions:**

```bash
# 1. Pre-download model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"

# 2. Increase vLLM timeout
# Edit config:
actor_rollout_ref:
  rollout:
    timeout: 600  # Increase timeout

# 3. Check vLLM logs
export VLLM_LOGGING_LEVEL=DEBUG
```

### Issue: Slow Training Speed

**Checklist:**

```bash
# 1. Verify GPU utilization >80%
nvidia-smi dmon -s u

# 2. Check if Flash Attention is enabled
python -c "from flash_attn import flash_attn_func; print('Flash Attention: OK')"

# 3. Enable torch.compile
optimizations:
  torch_compile:
    enabled: true

# 4. Use trajectory aggregation
agentlightning:
  trace_aggregator:
    level: trajectory

# 5. Increase batch size (if memory allows)
data:
  train_batch_size: 128  # From 64
```

---

## 9. Next Steps

### Phase 1: Validation (Week 1-2)

- [ ] Train 2-3 agents with APO
- [ ] Measure performance improvement
- [ ] Validate reward functions
- [ ] Document learnings

### Phase 2: GPU Training (Week 3-4)

- [ ] Train 1-2 agents with VERL
- [ ] Optimize GPU utilization
- [ ] Fine-tune hyperparameters
- [ ] A/B test trained vs. baseline

### Phase 3: Production Integration (Week 5-8)

- [ ] Add training endpoints to Foundry API
- [ ] Build training UI
- [ ] Integrate with evaluation service
- [ ] Deploy to production

### Recommended Resources

**Documentation:**
- Agent Lightning: https://github.com/microsoft/agent-lightning
- VERL: https://github.com/volcengine/verl
- vLLM: https://docs.vllm.ai

**Monitoring:**
- Weights & Biases: https://wandb.ai
- TensorBoard: https://www.tensorflow.org/tensorboard
- NVIDIA DCGM: https://developer.nvidia.com/dcgm

**Community:**
- Agent Lightning Discussions: https://github.com/microsoft/agent-lightning/discussions
- Foundry Internal: [Your internal channels]

---

## Quick Reference

### Essential Commands

```bash
# Activate environment
source venv_agent_lightning/bin/activate

# Start MongoDB
docker start agent-lightning-mongo

# Check GPUs
nvidia-smi

# Run APO training
python examples/train_react_agent_apo.py --agent-id <id> --dataset-file data.json

# Run VERL training
python examples/train_react_agent_verl.py --agent-id <id> --dataset-file data.json

# Monitor training
tensorboard --logdir ./verl_output/logs

# Stop all training
pkill -f train_react_agent
```

### Key File Locations

```
/home/user/Infosys-Agentic-Foundry/
├── poc_agent_lightning/
│   ├── src/
│   │   ├── foundry_lit_agent_adapter.py  # Core adapter
│   │   └── reward_functions.py           # Reward registry
│   ├── examples/
│   │   ├── train_react_agent_apo.py      # APO training
│   │   └── train_react_agent_verl.py     # VERL training
│   ├── configs/
│   │   └── verl_4xa100_config.yaml       # GPU config
│   └── SETUP_GUIDE.md                    # This file
├── INTEGRATION_PLAN_AGENT_LIGHTNING.md   # Integration plan
└── AGENT_TEMPLATES_UNDERSTANDING.md      # Templates docs
```

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review Agent Lightning [docs](https://github.com/microsoft/agent-lightning)
3. Ask in Foundry internal channels
4. File issue on GitHub

**POC Version:** 1.0
**Last Updated:** 2025-12-29
**Hardware:** 4x NVIDIA A100 80GB ✅
