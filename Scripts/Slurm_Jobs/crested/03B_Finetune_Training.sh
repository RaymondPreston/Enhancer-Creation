#!/bin/bash
#SBATCH --job-name=crested_finetune
#SBATCH --output=logs/crested_finetune_%j.out
#SBATCH --error=logs/crested_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --gpus=1

# Send email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

#Currently this is set up to run CPU for prechecking stuff prior to finetuning. Will need to adjust to use GPU eventually.

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted Gini Finetune Training..."
# Run the Python script using the Crested environment
conda run -n Crested python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/03B_Finetune_Training.py
echo "Finetuning job complete."

echo "Starting Finetune AnnData Generation..."
conda run -n Crested python Scripts/03_CREsted_Pipeline/03C_Generate_Finetune_AnnData.py
echo "Finetune AnnData generation complete."

echo "Starting Finetune Analysis on DA peaks..."
# Run the python script
conda run -n Crested python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/03D_Finetune_Training.py
echo "Finetuning complete. Check output/training_models/ for results."