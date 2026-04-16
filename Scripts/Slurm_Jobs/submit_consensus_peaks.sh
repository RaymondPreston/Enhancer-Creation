#!/bin/bash
#SBATCH --job-name=consensus_peaks
#SBATCH --output=logs/consensus_peaks_%j.out
#SBATCH --error=logs/consensus_peaks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=c64-m512 

# 1. Install missing R packages (if needed)
echo "Checking and installing required R packages..."
conda run -n Enhancer-Creation Rscript Scripts/install_r_pkgs.R

# 2. Run the Consensus Peaks R script
echo "Running Generate_Consensus_Peaks.R..."
conda run -n Enhancer-Creation Rscript Scripts/Generate_Consensus_Peaks.R

echo "Done."
