#!/bin/bash
#SBATCH --job-name=TF_Motif
#SBATCH --output=logs/TF_Motif%j.out
#SBATCH --error=logs/TF_Motif%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --gpus=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted TF Motif Analysis..."

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate Crested

# Run the Python script
python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/04B_TF_Analysis.py

echo "Model evaluation complete. Check output folder for results."
