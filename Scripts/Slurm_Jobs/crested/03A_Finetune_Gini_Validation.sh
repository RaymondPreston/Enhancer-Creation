#!/bin/bash
#SBATCH --job-name=gini_qc
#SBATCH --output=logs/gini_qc_%j.out
#SBATCH --error=logs/gini_qc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=c64-m512

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Gini-QC analysis..."

# Run the R script using the Crested_QC environment
conda run -n Crested_QC Rscript Scripts/03_CREsted_Pipeline/03A_Finetune_Gini_Validation.R

echo "Gini-QC analysis complete. Check output/Gini_QC/ for results."
