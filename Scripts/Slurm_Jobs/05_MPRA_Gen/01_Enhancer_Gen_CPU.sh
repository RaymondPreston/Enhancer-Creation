#!/bin/bash
#SBATCH --job-name=MPRA_Gen
#SBATCH --output=logs/MPRA_Gen%j.out
#SBATCH --error=logs/MPRA_Gen%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=c128-m1024

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
python /scratch/rprest2/Enhancer-Creation/Scripts/05_MPRA_Gen/01_Enhancer_Gen_NEW.py

echo "Enhancer generation complete."
