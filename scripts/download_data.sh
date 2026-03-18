#!/bin/bash
# =============================================================================
# Download FineWeb dataset for Parameter Golf
#
# Run this on a login node inside apptainer shell (compute nodes have no internet):
#   apptainer shell --bind /tmpdir,/work --nv <container.sif>
#   bash scripts/download_data.sh
#
# Or directly (if login node has the deps installed):
#   bash scripts/download_data.sh
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/tmpdir/m24047brmn/pgolf/data}"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=========================================="
echo "Parameter Golf - Data Download"
echo "Project dir:  ${PROJECT_DIR}"
echo "Data root:    ${DATA_ROOT}"
echo "Variant:      ${VARIANT}"
echo "Train shards: ${TRAIN_SHARDS}"
echo "=========================================="

# --- Redirect HF cache to tmpdir (avoid filling home directory quota) ---------
export HF_HOME="${DATA_ROOT}/.hf_cache"
mkdir -p "${HF_HOME}"

# --- Create target directories ------------------------------------------------
mkdir -p "${DATA_ROOT}/datasets"
mkdir -p "${DATA_ROOT}/tokenizers"

# Symlink project data dirs -> tmpdir so cached_challenge_fineweb.py writes
# directly to fast storage. Back up any existing dirs first.
for subdir in datasets tokenizers; do
    target="${PROJECT_DIR}/data/${subdir}"
    if [ -d "${target}" ] && [ ! -L "${target}" ]; then
        echo "Backing up existing ${target} -> ${target}.bak"
        mv "${target}" "${target}.bak"
    fi
    ln -sfn "${DATA_ROOT}/${subdir}" "${target}"
    echo "Symlinked ${target} -> ${DATA_ROOT}/${subdir}"
done

# --- Install deps if missing -------------------------------------------------
python3 -c "import huggingface_hub" 2>/dev/null || {
    echo "Installing download dependencies..."
    pip install --quiet huggingface-hub sentencepiece datasets tqdm numpy
}

# --- Download -----------------------------------------------------------------
echo ""
echo "Downloading ${VARIANT} with ${TRAIN_SHARDS} training shards..."
python3 "${PROJECT_DIR}/data/cached_challenge_fineweb.py" \
    --variant "${VARIANT}" \
    --train-shards "${TRAIN_SHARDS}"

# --- Verify -------------------------------------------------------------------
echo ""
echo "Download complete. Contents:"
echo "Datasets:"
ls -lh "${DATA_ROOT}/datasets/" 2>/dev/null || echo "  (empty)"
echo "Tokenizers:"
ls -lh "${DATA_ROOT}/tokenizers/" 2>/dev/null || echo "  (empty)"

TRAIN_COUNT=$(ls "${DATA_ROOT}/datasets/fineweb10B_${VARIANT}"/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls "${DATA_ROOT}/datasets/fineweb10B_${VARIANT}"/fineweb_val_*.bin 2>/dev/null | wc -l)
echo ""
echo "Summary: ${TRAIN_COUNT} train shards, ${VAL_COUNT} val shards"
echo "=========================================="
