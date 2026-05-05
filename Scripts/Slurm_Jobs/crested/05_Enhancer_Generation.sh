#!/bin/bash
#SBATCH --job-name=enhancer_gen
#SBATCH --output=logs/enhancer_gen_%j.out
#SBATCH --error=logs/enhancer_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=l4-4-gm96-c48-m192
#SBATCH --gpus=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Enhancer Generation (In-Silico Evolution)..."

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate Crested

# Run the Python script
python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/05D_Enhancer_Generation.py

echo "Enhancer generation complete."
