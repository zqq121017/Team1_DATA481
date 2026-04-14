#!/bin/bash

# --- Job Metadata ---
#SBATCH --job-name=ADCT_CPU_Run
#SBATCH --output=logs/cnn_test_output_%j.log
#SBATCH --error=errors/cnn_test_error_%j.err

# --- Resource Allocation ---
# Omit --partition to let SLURM find an idle node faster
#SBATCH --qos=normal           # Standard QOS for CPU jobs
#SBATCH --ntasks=1             # Run 1 instance of the task
#SBATCH --cpus-per-task=32      # 8 cores to handle multimodal data
#SBATCH --mem=32g              # 32GB RAM to prevent memory crashes
#SBATCH --time=04:00:00        # 4 hour time limit
#SBATCH --nodes=4

# --- Environment Setup ---
module load python/3.10.19      
source activate pytorch         
conda activate torch_env
# --- Execution ---
echo "Starting CPU-only task on node: $SLURMD_NODENAME"

# Use -u for unbuffered output to see Epochs in the log immediately
python -u CNN-test.py   
        

echo "Job Finished."