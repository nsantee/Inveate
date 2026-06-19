# Installation Guide

## Quick Start (Choose Your Hardware Profile)

### Option A: NVIDIA GPU Users (Recommended for RAG)
For best performance with local LLM inference and vector embedding, use the CUDA-optimized stack.

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install core dependencies (Hardware Agnostic)
pip install -r requirements-core.txt

# 3. Install NVIDIA CUDA acceleration stack
pip install -r requirements-cuda.txt
```

### Option B: CPU / AMD ROCm / Intel ARC Users
If you do not have an NVIDIA GPU, the system will run on CPU. Performance for embedding generation will be slower (approx. 10x).

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install core dependencies only (No CUDA)
pip install -r requirements-core.txt
```

### Option C: Apple Silicon (M1/M2/M3)
For Mac users, PyTorch will automatically utilize the Metal Performance Shaders (MPS).

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install core dependencies only (No CUDA)
pip install -r requirements-core.txt
```

## Hardware Specific Notes
- **CUDA 12.6**: The `requirements-cuda.txt` file assumes CUDA 12.x (or onward) drivers are installed on your system.
