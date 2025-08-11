#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=uni_multi_cox
#SBATCH --mem=32G
#SBATCH -A clifton.prj
#SBATCH -o /well/clifton/users/ncu080/proteo_fs/outputs/slurm_logs/uni_multi_cox.out
#SBATCH -e /well/clifton/users/ncu080/proteo_fs/outputs/slurm_logs/uni_multi_cox.err

echo "------------------------------------------------" 
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"

echo "SLURM Job ID: $SLURM_JOB_ID"

# Load the correct R module
module load R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2

# Run the R script and capture output
Rscript /well/clifton/users/ncu080/proteo_fs/R/uni_multi_cox.R

echo "Job finished at: $(date)"
