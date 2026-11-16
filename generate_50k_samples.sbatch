#!/bin/bash
#SBATCH --job-name=gen_simu_50k
#SBATCH --partition=gpu-general-pool
#SBATCH --account=public-blaufer_v2
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch200/yotams2/logs/gen_simu_50k_%j.out
#SBATCH --error=/scratch200/yotams2/logs/gen_simu_50k_%j.err

echo "=== Job started on $(hostname) at $(date) ==="

# 1. Ensure logs and data directories exist on /scratch200
mkdir -p /scratch200/yotams2/logs
mkdir -p /scratch200/yotams2/data/MicSig/simu

# 2. Load necessary cluster modules
module purge
module load python/python-3.11_torch-2.2.2
module load CUDA/CUDA-12.2

# 3. Activate the virtual environment located on /scratch200
source /scratch200/yotams2/venvs/sac-sar-ssl/bin/activate

# 4. Navigate to data generation script directory
cd /a/home/cc/students/enginer/yotams2/SAC-SAR-SSL/code/data_generation

# 5. Define paths and parameters
# NOTE: Set SRC_DIR to where your clean speech dataset (WSJ0) is stored.
# gen_simu.py automatically looks inside $SRC_DIR/tr for the pretrain stage.
SRC_DIR="/scratch200/yotams2/data/SrcSig/wsj0"
SAVE_TO="/scratch200/yotams2/data/MicSig/simu"
DATA_NUM=50000

echo "Using source speech from: $SRC_DIR"
echo "Saving generated samples to: $SAVE_TO"
echo "Number of samples: $DATA_NUM"

# 6. Execute data generation
python gen_simu.py \
    --mode sig \
    --stage pretrain \
    --data_num $DATA_NUM \
    --src_dir "$SRC_DIR" \
    --save_to "$SAVE_TO" \
    --gpus [0]

echo "=== Job finished at $(date) ==="
