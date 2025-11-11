# Athena Setup Guide

Complete step-by-step guide for setting up Athena Recursive AI on your dual 7900 XT system.

## Prerequisites

### Hardware Requirements

- **Dual AMD Radeon RX 7900 XT GPUs** (20GB VRAM each, 40GB total)
- Minimum 32GB system RAM (64GB recommended)
- Modern multi-core CPU (Ryzen 7/9 or equivalent)
- Fast SSD with at least 100GB free space (for models)

### Software Requirements

- **Operating System**: Linux (Ubuntu 22.04+ recommended) or Windows 11
- **Python**: 3.10 or higher
- **GPU Drivers**:
  - Linux: ROCm 5.7+ ([installation guide](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.7/page/Overview_of_ROCm_Installation_Methods.html))
  - Windows: AMD Adrenalin drivers (latest version)
- **LM Studio**: Latest version ([download](https://lmstudio.ai/))

## Step 1: Install System Dependencies

### Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3.10 python3.10-venv python3-pip git curl -y

# Install ROCm (for AMD GPU support)
# Follow official AMD ROCm installation guide for your distro
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*_all.deb
sudo apt install ./amdgpu-install_*_all.deb
sudo amdgpu-install --usecase=rocm
```

### Windows

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Install Git from [git-scm.com](https://git-scm.com/)
3. Install AMD Adrenalin drivers from [amd.com](https://www.amd.com/en/support)
4. Ensure AMD GPUs are properly detected in Device Manager

## Step 2: Install LM Studio

1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Install and launch LM Studio
3. Verify GPU detection:
   - Go to Settings → Hardware
   - Ensure both 7900 XT GPUs are detected
   - Note the GPU IDs (typically 0 and 1)

## Step 3: Download Models

Download the following models in LM Studio (use the Search tab):

### GPU 1 (Device 0) - Logical/Context Pipeline (~18GB)

**Orchestrator:**
- **Model**: `Qwen/Qwen2.5-14B-Instruct-GGUF`
- **Quantization**: `Q5_K_M` (~10GB)
- **Port**: 1234
- **GPU**: Assign to GPU 0

**Reasoning Expert:**
- **Model**: `microsoft/Phi-3.5-mini-instruct-gguf`
- **Quantization**: `Q4_K_M` (~3GB)
- **Port**: 1235
- **GPU**: Assign to GPU 0

**Memory Expert:**
- **Model**: `meta-llama/Llama-3.1-8B-Instruct-GGUF`
- **Quantization**: `Q4_K_M` (~5GB)
- **Port**: 1236
- **GPU**: Assign to GPU 0

### GPU 2 (Device 1) - Creative/Implementation Pipeline (~9GB)

**Creative Expert:**
- **Model**: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- **Quantization**: `Q4_K_M` (~4.5GB)
- **Port**: 1237
- **GPU**: Assign to GPU 1

**Technical Expert:**
- **Model**: `Qwen/CodeQwen1.5-7B-Chat-GGUF`
- **Quantization**: `Q4_K_M` (~4.5GB)
- **Port**: 1238
- **GPU**: Assign to GPU 1

**Performance Benefits**: This optimized distribution enables true parallelism when consulting experts from different GPUs simultaneously, achieving 1.5-2x throughput gains during multi-expert queries.

## Step 4: Configure LM Studio Instances

LM Studio doesn't natively support multiple simultaneous servers, so you'll need to:

### Option A: Use Multiple LM Studio Instances (Recommended)

1. **Install LM Studio in multiple locations:**

   ```bash
   # Linux/Mac
   cp -r ~/Applications/LM\ Studio.app ~/Applications/LM\ Studio-1.app
   cp -r ~/Applications/LM\ Studio.app ~/Applications/LM\ Studio-2.app
   # ... repeat for 3, 4, 5
   ```

   Windows: Copy the LM Studio installation folder 5 times

2. **Launch instances with optimized GPU distribution:**

   **GPU 1 (Device 0) Instances:**

   Instance 1 (Orchestrator):
   - Port: 1234
   - Model: Qwen2.5-14B-Instruct-Q5_K_M
   - GPU: 0

   Instance 2 (Reasoning Expert):
   - Port: 1235
   - Model: Phi-3.5-mini-instruct-Q4_K_M
   - GPU: 0

   Instance 3 (Memory Expert):
   - Port: 1236
   - Model: Llama-3.1-8B-Instruct-Q4_K_M
   - GPU: 0

   **GPU 2 (Device 1) Instances:**

   Instance 4 (Creative Expert):
   - Port: 1237
   - Model: Mistral-7B-Instruct-Q4_K_M
   - GPU: 1

   Instance 5 (Technical Expert):
   - Port: 1238
   - Model: CodeQwen-7B-Q4_K_M
   - GPU: 1

   **Why this distribution?** Grouping Reasoning + Memory with the Orchestrator on GPU 1 enables seamless logical and context processing, while Creative + Technical on GPU 2 allows parallel creative/code generation without GPU contention.

### Option B: Use Alternative Model Servers

Instead of multiple LM Studio instances, you can use:

- **llama.cpp server** with multiple instances
- **vLLM** with tensor parallelism
- **Text-generation-webui** with multiple instances

Example with llama.cpp:

```bash
# Install llama.cpp with ROCm support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_HIPBLAS=1

# Launch servers (in separate terminals)
./server -m models/qwen-14b-q5.gguf --port 1234 --gpu-id 0
./server -m models/phi-3.5-mini-q4.gguf --port 1235 --gpu-id 1
# ... etc for other models
```

## Step 5: Install Athena

```bash
# Clone repository
git clone https://github.com/zoadrazorro/Athena-Recursive-AI.git
cd Athena-Recursive-AI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Athena
pip install -e .

# Or for development:
pip install -e ".[dev]"
```

## Step 6: Configure Athena

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

Update the `.env` file with your endpoints:

```bash
# LM Studio Endpoints (verify these match your setup)
ORCHESTRATOR_URL=http://localhost:1234/v1
REASONING_EXPERT_URL=http://localhost:1235/v1
CREATIVE_EXPERT_URL=http://localhost:1236/v1
TECHNICAL_EXPERT_URL=http://localhost:1237/v1
MEMORY_EXPERT_URL=http://localhost:1238/v1

# Model names (should match what's loaded in LM Studio)
ORCHESTRATOR_MODEL=qwen2.5-14b-instruct
REASONING_MODEL=phi-3.5-mini-instruct
CREATIVE_MODEL=mistral-7b-instruct
TECHNICAL_MODEL=codeqwen-7b-instruct
MEMORY_MODEL=llama-3.1-8b-instruct

# GPU mapping
ORCHESTRATOR_GPU=0
EXPERTS_GPU=1

# Tuning parameters (adjust based on your needs)
MAX_CONTEXT_LENGTH=8192
TEMPERATURE=0.7
GWT_ATTENTION_THRESHOLD=0.6
ENABLE_PARALLEL_CONSULTATION=true
```

## Step 7: Verify Installation

```bash
# Run health check
athena health
```

Expected output:

```
System Health Check
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Component        ┃ Status     ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ orchestrator     │ ✓ Healthy  │
│ reasoning        │ ✓ Healthy  │
│ creative         │ ✓ Healthy  │
│ technical        │ ✓ Healthy  │
│ memory           │ ✓ Healthy  │
└──────────────────┴────────────┘

All systems operational!
```

If any component shows as unavailable:
1. Check that LM Studio instance is running
2. Verify the model is loaded
3. Confirm the port matches your configuration
4. Check GPU assignment

## Step 8: Test Basic Functionality

```bash
# Try a simple query
athena query "Explain recursion in simple terms"

# Or launch interactive mode
athena interactive
```

In interactive mode, try:

```
You: What is 2 + 2?
# Should get quick direct response

You: Write a Python function to calculate fibonacci numbers
# Should route to Technical Expert

You: What are the philosophical implications of AI consciousness?
# Should route to Creative Expert

/stats
# Show system statistics
```

## Performance Tuning

### If You Experience VRAM Issues

1. **Reduce quantization:**
   - Use Q4_K_M instead of Q5_K_M for orchestrator
   - Use Q3_K_M for experts (quality trade-off)

2. **Reduce context length:**
   ```bash
   MAX_CONTEXT_LENGTH=4096
   ```

3. **Disable parallel consultation:**
   ```bash
   ENABLE_PARALLEL_CONSULTATION=false
   ```

4. **Reduce workspace size:**
   ```bash
   GWT_WORKSPACE_SIZE=3
   ```

### If You Experience Slow Response Times

1. **Check GPU utilization:**
   ```bash
   # Linux
   rocm-smi

   # Windows
   # Use GPU-Z or AMD Software
   ```

2. **Verify correct GPU assignment:**
   - Ensure expert models are on GPU 1
   - Orchestrator on GPU 0

3. **Adjust temperature for faster sampling:**
   ```bash
   TEMPERATURE=0.5
   ```

## Monitoring

### View Logs

```bash
# Real-time logs
tail -f athena.log

# With filtering
tail -f athena.log | grep "ERROR\|WARNING"
```

### Monitor GPU Usage

```bash
# Linux - continuous monitoring
watch -n 1 rocm-smi

# Check VRAM usage
rocm-smi --showmeminfo VRAM
```

### System Statistics

In interactive mode, use `/stats` to see:
- Query counts
- Expert usage
- Average confidence scores
- Workspace state

## Troubleshooting

### "Cannot connect to endpoint" Error

**Cause**: LM Studio instance not running or wrong port

**Solution**:
1. Check LM Studio is running: `curl http://localhost:1234/v1/models`
2. Verify port in .env matches LM Studio
3. Check firewall settings

### "Out of VRAM" Error

**Cause**: Models too large for GPU memory

**Solution**:
1. Use lower quantization (Q4 or Q3)
2. Reduce context length
3. Close other GPU-intensive applications
4. Ensure only designated models on each GPU

### "Model not responding" Error

**Cause**: Model crashed or hung

**Solution**:
1. Restart LM Studio instance
2. Reload the model
3. Check GPU temperature (overheating?)
4. Reduce batch size in LM Studio settings

### Slow Performance

**Cause**: GPU not being utilized or wrong GPU assignment

**Solution**:
1. Verify GPU assignment in LM Studio
2. Check GPU drivers are up to date
3. Monitor GPU utilization with rocm-smi
4. Ensure PCI Express link is at full speed

## Next Steps

Once installation is verified:

1. **Explore examples:**
   ```bash
   python examples/basic_query.py
   python examples/multi_turn_conversation.py
   python examples/workspace_inspection.py
   ```

2. **Read the architecture docs:**
   - Understanding expert specializations
   - Global Workspace mechanics
   - Routing strategies

3. **Experiment with configuration:**
   - Tune GWT parameters
   - Adjust expert weights
   - Customize system prompts

4. **Integrate with your projects:**
   - Use the Python API
   - Create custom experts
   - Build domain-specific applications

## Support

If you encounter issues not covered here:

1. Check existing GitHub issues
2. Review logs for error details
3. Run `athena health` for diagnostics
4. Open a new issue with:
   - Error messages
   - System specs
   - Configuration files
   - Log excerpts

Happy reasoning! 🧠✨
