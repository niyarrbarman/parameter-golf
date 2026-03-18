#!/bin/bash
# =============================================================================
# Parameter Golf - Baseline Training on 4 nodes x 2 A100-80GB
#
# Usage:
#   sbatch scripts/train_slurm.sh
#
# Override defaults:
#   RUN_ID=my_experiment sbatch scripts/train_slurm.sh
# =============================================================================
#SBATCH -J pgolf_baseline
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# --- Paths -------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/work/m24047/m24047brmn/parameter-golf}"
DATA_ROOT="${DATA_ROOT:-/tmpdir/m24047brmn/pgolf/data}"
CONTAINER="${CONTAINER:-/work/conteneurs/calmip/nemo_25.04.03_arm.sif}"

# --- Training config ---------------------------------------------------------
RUN_ID="${RUN_ID:-baseline_$(date +%Y%m%d_%H%M%S)}"
VARIANT="${VARIANT:-sp1024}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
ITERATIONS="${ITERATIONS:-20000}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
SEED="${SEED:-1337}"

# --- Model config (baseline defaults) ----------------------------------------
NUM_LAYERS="${NUM_LAYERS:-9}"
MODEL_DIM="${MODEL_DIM:-512}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
MLP_MULT="${MLP_MULT:-2}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"

# --- Derived paths ------------------------------------------------------------
DATA_PATH="${DATA_ROOT}/datasets/fineweb10B_${VARIANT}"
TOKENIZER_PATH="${DATA_ROOT}/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model"

# --- Multi-node coordination --------------------------------------------------
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 50000 + 10001" | bc)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

GPUS_PER_NODE=2
NNODES=${SLURM_NNODES:-4}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "=========================================="
echo "Parameter Golf - Training"
echo "Project:     ${PROJECT_DIR}"
echo "Container:   ${CONTAINER}"
echo "Data:        ${DATA_PATH}"
echo "Tokenizer:   ${TOKENIZER_PATH}"
echo "Run ID:      ${RUN_ID}"
echo "Nodes:       ${NNODES}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo "World size:  ${WORLD_SIZE}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Wallclock:   ${MAX_WALLCLOCK_SECONDS}s"
echo "Iterations:  ${ITERATIONS}"
echo "Batch tokens:${TRAIN_BATCH_TOKENS}"
echo "Model:       ${NUM_LAYERS}L ${MODEL_DIM}D ${NUM_HEADS}H ${NUM_KV_HEADS}KV"
echo "=========================================="

# --- Verify data exists -------------------------------------------------------
if [ ! -d "${DATA_PATH}" ]; then
    echo "ERROR: Data not found at ${DATA_PATH}"
    echo "Run 'bash scripts/download_data.sh' first."
    exit 1
fi
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "ERROR: Tokenizer not found at ${TOKENIZER_PATH}"
    exit 1
fi

# --- Launch -------------------------------------------------------------------
srun apptainer exec \
    --env "MASTER_ADDR=${MASTER_ADDR}" \
    --env "MASTER_PORT=${MASTER_PORT}" \
    --env "RUN_ID=${RUN_ID}" \
    --env "DATA_PATH=${DATA_PATH}" \
    --env "TOKENIZER_PATH=${TOKENIZER_PATH}" \
    --env "VOCAB_SIZE=${VOCAB_SIZE}" \
    --env "ITERATIONS=${ITERATIONS}" \
    --env "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}" \
    --env "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS}" \
    --env "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN}" \
    --env "VAL_LOSS_EVERY=${VAL_LOSS_EVERY}" \
    --env "TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY}" \
    --env "SEED=${SEED}" \
    --env "NUM_LAYERS=${NUM_LAYERS}" \
    --env "MODEL_DIM=${MODEL_DIM}" \
    --env "NUM_HEADS=${NUM_HEADS}" \
    --env "NUM_KV_HEADS=${NUM_KV_HEADS}" \
    --env "MLP_MULT=${MLP_MULT}" \
    --env "TIE_EMBEDDINGS=${TIE_EMBEDDINGS}" \
    --bind /tmpdir,/work --nv "${CONTAINER}" \
    torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${PROJECT_DIR}/train_gpt.py"

status=$?
echo "=========================================="
echo "Training finished with status ${status}"
echo "=========================================="
exit ${status}
