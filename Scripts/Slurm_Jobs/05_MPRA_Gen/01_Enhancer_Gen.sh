#!/bin/bash
#SBATCH --job-name=MPRA_Gen
#SBATCH --output=logs/MPRA_Gen%j.out
#SBATCH --error=logs/MPRA_Gen%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --partition=h100-1-gm80-c16-m256
#SBATCH --gpus=1

# Send email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Enhancer Generation (In-Silico Evolution)..."

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate Crested

# Run the Python script
python /scratch/rprest2/Enhancer-Creation/Scripts/05_MPRA_Gen/01_Enhancer_Gen.py

echo "Enhancer generation complete."
