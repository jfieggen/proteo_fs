#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=cox_lasso2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH -A clifton.prj
#SBATCH -o outputs/slurm_logs/cox_lasso2.out
#SBATCH -e outputs/slurm_logs/cox_lasso2.err

echo "------------------------------------------------" 
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"

export PYTHONUNBUFFERED=1

echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate the virtual environment
source /well/clifton/users/ncu080/proteo_fs/proteo_fs_venv/bin/activate

# Load the libffi module as required
module load libffi/3.4.4-GCCcore-12.3.0

# Run the script
python /well/clifton/users/ncu080/proteo_fs/python/cox_lasso.py

echo "Job finished at: $(date)"
