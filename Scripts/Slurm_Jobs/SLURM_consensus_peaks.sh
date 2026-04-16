#!/bin/bash
#SBATCH --job-name=consensus_peaks
#SBATCH --output=logs/consensus_peaks_%j.out
#SBATCH --error=logs/consensus_peaks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=c128-m1024

# 2. Run the Consensus Peaks R script
echo "Running Generate_Consensus_Peaks.R..."
conda run -n Enhancer-Creation Rscript Scripts/Generate_Consensus_Peaks.R

echo "Done."
