#!/bin/bash

# --- Job Metadata ---
#SBATCH --job-name=ADCT_CNN_Kfold_Run
#SBATCH --output=logs/cnn_kfold_output_%j.log
#SBATCH --error=errors/cnn_kfold_error_%j.err

# --- Resource Allocation ---
# Omit --partition to let SLURM find an idle node faster
#SBATCH --qos=normal           # Standard QOS for CPU jobs
#SBATCH --ntasks=1             # Run 1 instance of the task
#SBATCH --cpus-per-task=64      #  cores to handle multimodal data
#SBATCH --mem=32g              # 32GB RAM to prevent memory crashes
#SBATCH --time=10:00:00        # 4 hour time limit
#SBATCH --nodes=1


# --- Environment Setup ---
module load python/3.10.19      
source activate pytorch         
conda activate torch_env
# --- Execution ---
echo "Starting CPU-only task on node: $SLURMD_NODENAME"

# Use -u for unbuffered output to see Epochs in the log immediately
python -u CNN_model_val.py          
        

echo "Job Finished."