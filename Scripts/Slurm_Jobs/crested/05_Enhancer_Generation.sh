#!/bin/bash
#SBATCH --job-name=enhancer_gen
#SBATCH --output=logs/enhancer_gen_%j.out
#SBATCH --error=logs/enhancer_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --gpus=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Enhancer Generation (In-Silico Evolution)..."

# Run the Python script using the Crested environment
conda run -n Crested python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/05b_Enhancer_Generation.py

echo "Enhancer generation complete."
