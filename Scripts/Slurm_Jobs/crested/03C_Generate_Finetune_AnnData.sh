#!/bin/bash
#SBATCH --job-name=gen_finetune_adata
#SBATCH --output=logs/gen_finetune_adata_%j.out
#SBATCH --error=logs/gen_finetune_adata_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=c64-m512

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Finetune AnnData Generation..."

# Run the Python script using the Crested environment
# Note: Ensure bedtools is installed in this environment or available in PATH
conda run -n Crested python Scripts/03_CREsted_Pipeline/03C_Generate_Finetune_AnnData.py

echo "Finetune AnnData generation complete."
