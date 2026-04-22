#!/bin/bash
#SBATCH --job-name=crested_analysis
#SBATCH --output=logs/crested_analysis_%j.out
#SBATCH --error=logs/crested_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=c64-m512

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted target parameter analysis..."
echo "Comparing: max, mean, sum"

conda init bash > /dev/null 2>&1
source ~/.bashrc  
conda activate Crested

python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/02_Crested_PP_Target_Analysis.py

echo "Analysis complete. Check the output/ directory for results."
