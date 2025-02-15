#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=process
#SBATCH --mem=64G
#SBATCH -A clifton.prj
#SBATCH -o outputs/slurm_logs/process.out
#SBATCH -e outputs/slurm_logs/process.err

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

# Run the script
python /well/clifton/users/ncu080/proteo_fs/python/preprocess_data.py

echo "Job finished at: $(date)"
