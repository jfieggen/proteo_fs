#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=R_lasso
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH -A clifton.prj
#SBATCH -o /well/clifton/users/ncu080/proteo_fs/outputs/slurm_logs/R_lasso.out
#SBATCH -e /well/clifton/users/ncu080/proteo_fs/outputs/slurm_logs/R_lasso.err

echo "------------------------------------------------" 
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Load the correct R module
module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2

# Run the R script
Rscript /well/clifton/users/ncu080/proteo_fs/R/lasso.R

echo "Job finished at: $(date)"
