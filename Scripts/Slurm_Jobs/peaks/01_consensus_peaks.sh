#!/bin/bash
#SBATCH --job-name=consensus_peaks
#SBATCH --output=logs/consensus_peaks_%j.out
#SBATCH --error=logs/consensus_peaks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=c128-m1024

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate peak_analysis

# 2. Run the Consensus Peaks R script
echo "Running Generate_Consensus_Peaks.R..."
Rscript /scratch/rprest2/Enhancer-Creation/Scripts/02_Peak_Analysis/01_Generate_Consensus_Peaks.R

echo "Done."
