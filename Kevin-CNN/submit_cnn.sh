#!/bin/bash

#1. Job Metadata
#SBATCH --job-name=ADCT_CNN_Research
#SBATCH --output=cnn_output_%j.log
#SBATCH --error=cnn_error_%j.err

#2. Resource Allocation
#SBATCH --partition=gpu        # Request a GPU partition (volt or a100)
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --nodes=1               # Ensure all resources are on one node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Request 8 CPU cores for data loading
#SBATCH --mem=32G               # Request 32GB RAM to prevent segmentation faults
#SBATCH --time=04:00:00         # Request 4 hours of walltime

#3. Environment Setup
module load python/3.10.19      # Match your local conda version
module load cuda/12.1           # Load CUDA drivers for GPU support
source activate pytorch         # Activate your environment

#4. Execution
echo "Starting K-Fold Cross Validation..."
python CNN-kfold.py

echo "Starting Final Test Script..."
python CNN-test.py

echo "Job Finished."