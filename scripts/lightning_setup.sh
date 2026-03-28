#!/usr/bin/env bash
# Lightning AI Studio setup script for ArticleTagging benchmark
# Usage: bash scripts/lightning_setup.sh
set -euo pipefail

echo "=== ArticleTagging Lightning AI Setup ==="

# ── Check GPU ─────────────────────────────────────────────────────────
echo ""
echo "--- GPU Info ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found — no GPU detected"
fi
python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB' if torch.cuda.is_available() else 'No CUDA')" 2>/dev/null || echo "PyTorch not yet installed"

# ── Install Unsloth (must be first for correct torch/triton versions) ─
echo ""
echo "--- Installing Unsloth ---"
pip install --upgrade --no-cache-dir unsloth unsloth_zoo 2>&1 | tail -3

# ── Install project with training + serving extras ───────────────────
echo ""
echo "--- Installing ArticleTagging ---"
cd "$(dirname "$0")/.."
pip install -e ".[training,serving]" 2>&1 | tail -3

# ── Install Claude Code ───────────────────────────────────────────────
echo ""
echo "--- Installing Claude Code ---"
if command -v claude &>/dev/null; then
    echo "  Claude Code already installed: $(claude --version)"
else
    npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
    echo "  Installed: $(claude --version 2>/dev/null || echo 'run: claude to authenticate')"
fi

# ── Verify imports ────────────────────────────────────────────────────
echo ""
echo "--- Verifying imports ---"
python3 -c "
from unsloth import FastVisionModel
from article_tagging.training.model import load_model
from article_tagging.training.trainer import run_training
from article_tagging.evaluation.metrics import compute_metrics
print('All imports OK')
"

# ── Check data files ──────────────────────────────────────────────────
echo ""
echo "--- Checking data files ---"
REQUIRED_FILES=(
    "data/processed/fashion/train.jsonl"
    "data/processed/fashion/val.jsonl"
    "data/processed/fashion/test_50_seed42.jsonl"
    "configs/schemas/fashion.yaml"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done

# Check for images
if [ -d "data/raw/fashion/images" ]; then
    IMG_COUNT=$(find data/raw/fashion/images -name "*.jpg" -o -name "*.png" | head -100 | wc -l)
    echo "  OK: data/raw/fashion/images/ ($IMG_COUNT+ images found)"
else
    echo "  MISSING: data/raw/fashion/images/"
    MISSING=$((MISSING + 1))
fi

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING required files/dirs missing."
    echo "Upload data to Lightning AI Studio before running the benchmark."
    echo ""
    echo "Required structure:"
    echo "  data/raw/fashion/images/*.jpg    (product images)"
    echo "  data/processed/fashion/train.jsonl"
    echo "  data/processed/fashion/val.jsonl"
    echo "  data/processed/fashion/test_50_seed42.jsonl"
else
    echo ""
    echo "All data files present."
fi

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo "Next: python scripts/benchmark_qwen3vl_30b.py --phase all"
echo "  or: python scripts/benchmark_qwen3vl_30b.py --phase train"
echo "  or: python scripts/benchmark_qwen3vl_30b.py --phase eval"
echo "  or: python scripts/benchmark_qwen3vl_30b.py --phase latency"
