#!/bin/bash
#SBATCH --job-name=crested_preprocess
#SBATCH --output=logs/crested_preprocess_%j.out
#SBATCH --error=logs/crested_preprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=c128-m1024

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting CREsted preprocessing..."

# Run the Python script using the Crested environment
conda run -n Crested python Scripts/Crested_Preprocessing.py

echo "Preprocessing complete. Check output/CREsted_PreProcess/ for plots and input/training_inputs/ for the h5ad file."
