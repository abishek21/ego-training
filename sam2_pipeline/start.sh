#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SAM 2 Setup Script for RunPod / GPU Machine
# Installs SAM 2, downloads the large checkpoint, sets up env.
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────

set -e

echo "=========================================="
echo "  SAM 2 GPU Pipeline Setup"
echo "=========================================="

# --- Check GPU ---
echo ""
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "❌ No GPU detected. This script requires CUDA."
    exit 1
}

# --- Python environment ---
echo ""
echo "🐍 Setting up Python environment..."
pip install --upgrade pip

# --- Install PyTorch with CUDA (adjust cu121 to your CUDA version) ---
echo ""
echo "🔥 Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- Install SAM 2 from Meta's repo ---
echo ""
echo "📦 Installing SAM 2..."
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git
fi
cd sam2
pip install -e .
cd ..

# --- Download SAM 2 Large checkpoint ---
echo ""
echo "⬇️  Downloading SAM 2 Large checkpoint (~900MB)..."
mkdir -p checkpoints
if [ ! -f "checkpoints/sam2.1_hiera_large.pt" ]; then
    wget -O checkpoints/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

# --- Extra deps for our scripts ---
echo ""
echo "📦 Installing pipeline dependencies..."
pip install opencv-python numpy pillow tqdm

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload your video + prompts.json"
echo "  2. Run: python segment.py --video ego_press.mp4 --prompts prompts.json --output masks/"
echo ""
