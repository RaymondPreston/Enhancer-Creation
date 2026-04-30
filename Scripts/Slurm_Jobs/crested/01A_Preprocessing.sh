#!/bin/bash
#SBATCH --job-name=crested_preprocess
#SBATCH --output=logs/crested_preprocess_%j.out
#SBATCH --error=logs/crested_preprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=c64-m512


# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted preprocessing..."

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate Crested

# Run the Python script
python /scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/01A_Preprocessing.py

echo "Preprocessing complete. Check output/CREsted_PreProcess/ for plots and input/training_inputs/ for the h5ad file."
