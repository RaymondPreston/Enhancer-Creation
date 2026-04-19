#!/bin/bash
#SBATCH --job-name=crested_cnn_train
#SBATCH --output=logs/crested_train_%j.out
#SBATCH --error=logs/crested_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                              # We need multiple CPUs to handle the on-the-fly data generation/shifting
#SBATCH --mem=64G                                      # High memory to load the entire .h5ad AnnData object into RAM
#SBATCH --time=24:00:00                                # Deep learning takes time; giving it 24 hours to be safe
#SBATCH --partition=a100-8-gm320-c96-m1152             # MUST BE A GPU PARTITION (Check your cluster's specific GPU partition name)
#SBATCH --gpus=1                                       # Request exactly 1 GPU


# Send email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# 1. Load the conda environment that contains crested, keras, and anndata
echo "Activating Conda Environment..."
source ~/.bashrc  # Or whatever script initializes your conda
conda activate Crested

# 3. Run the training script
echo "Starting CREsted Training..."
python /scratch/rprest2/Enhancer-Creation/Scripts/Crested_Training.py

echo "Training job finished."