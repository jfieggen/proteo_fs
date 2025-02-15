#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gpus=1
#SBATCH --job-name=xgb_train
#SBATCH -A clifton.prj
#SBATCH -o outputs/slurm_logs/xgboost.out
#SBATCH -e outputs/slurm_logs/xgboost.err

echo "------------------------------------------------" 
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate the virtual environment
source /well/clifton/users/ncu080/proteo_fs/proteo_fs_venv/bin/activate

# Load the libffi module as required
module load libffi/3.4.4-GCCcore-12.3.0

# Run the XGBoost feature selection / training script
python /well/clifton/users/ncu080/proteo_fs/python/xgb_feature_selector.py

echo "Job finished at: $(date)"
