#!/bin/bash
#SBATCH --job-name=ADCT_Grid
#SBATCH --output=logs/cnn_grid_%j.log
#SBATCH --error=errors/cnn_grid_error_%j.err
#SBATCH --mem=48g               # Increased RAM for grid search overhead
#SBATCH --cpus-per-task=32      # More cores for MKL speedup
#SBATCH --time=24:00:00         # Grid search takes a long time on CPU
#SBATCH --qos=normal

module load python/3.10.19
source activate pytorch         
conda activate torch_env
echo "Starting CPU-only task on node: $SLURMD_NODENAME"

python -u CNN_gridsearch.py