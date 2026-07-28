#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# WiLoR GPU setup for the stereo-3D hand pipeline (RunPod / GPU pod)
# One-shot: installs WiLoR-mini + deps into the current Python env.
# Usage:  bash setup_wilor.sh
# ─────────────────────────────────────────────────────────────
set -e

echo "=========================================="
echo "  WiLoR-mini setup"
echo "=========================================="

echo ""
echo "🔍 GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "❌ No GPU detected."; exit 1; }

echo ""
echo "🐍 Python:"; python --version

echo ""
echo "📦 Installing pipeline deps (opencv, numpy, pyyaml)..."
pip install -q opencv-python numpy pyyaml

echo ""
echo "📦 Installing WiLoR-mini (auto-downloads weights on first run)..."
pip install -q git+https://github.com/warmshao/WiLoR-mini

echo ""
echo "✅ Verifying import + CUDA..."
python - <<'PY'
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("WiLoR-mini import OK")
PY

echo ""
echo "=========================================="
echo "✅ Setup complete."
echo "=========================================="
echo ""
echo "Next (from stereo_3d/orbbec):"
echo "  python test_wilor_gpu.py --data <DATA_DIR> --frames 300 --viz-frame 75"
echo "  python process_clip.py  --data <DATA_DIR> --output ../out_wilor --max-frames 300 --detector wilor"
