#!/bin/bash
#SBATCH --job-name=crested_evaluation
#SBATCH --output=logs/crested_evaluation_%j.out
#SBATCH --error=logs/crested_evaluation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=a100-8-gm320-c96-m1152
#SBATCH --gpus=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted preprocessing..."

# Run the Python script using the Crested environment
conda run -n Crested python /scratch/rprest2/Enhancer-Creation/Scripts/Crested_Evaluation.py

echo "Model evaluation complete. Check output folder for results."
