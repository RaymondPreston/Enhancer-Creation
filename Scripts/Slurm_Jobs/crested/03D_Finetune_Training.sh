#!/bin/bash
#SBATCH --job-name=Finetune_DA
#SBATCH --output=logs/da_finetune_%j.out
#SBATCH --error=logs/da_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --gpus=1

#Currently set up for CPU usage. Will need to change once I need GPU

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Finetune Analysis on DA peaks..."

# Run the python script
conda run -n Crested python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/03D_Finetune_Training.py

echo "Finetuning complete. Check output/training_models/ for results."
