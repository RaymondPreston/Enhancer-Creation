#!/bin/bash
#SBATCH --job-name=crested_evaluation
#SBATCH --output=logs/crested_evaluation_%j.out
#SBATCH --error=logs/crested_evaluation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=rp6b-1-gm96-c8-m64
#SBATCH --gpus=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted preprocessing..."

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate Crested

# Run the Python script
python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/04_Evaluation.py

echo "Model evaluation complete. Check output folder for results."
