#!/bin/bash
#SBATCH --job-name=Bal_FT
#SBATCH --output=logs/Bal_FT%j.out
#SBATCH --error=logs/Bal_FT%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --partition=a100-8-gm320-c96-m1152
#SBATCH --gpus=1

# Send email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Balanced Finetuning..."

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate Crested

# Run the Python script
python /scratch/rprest2/Enhancer-Creation/Scripts/05_MPRA_Gen/02_Finetune_Training_Balanced.py

echo "Enhancer generation complete."
