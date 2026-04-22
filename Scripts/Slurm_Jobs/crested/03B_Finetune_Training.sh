#!/bin/bash
#SBATCH --job-name=crested_finetune
#SBATCH --output=logs/crested_finetune_%j.out
#SBATCH --error=logs/crested_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=a100-8-gm320-c96-m1152
#SBATCH --gpus=1

# Send email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

#Currently this is set up to run CPU for prechecking stuff prior to finetuning. Will need to adjust to use GPU eventually.

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted Finetune Training..."

# Run the Python script using the Crested environment
conda run -n Crested python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/03B_Finetune_Training.py

echo "Finetuning job complete."
