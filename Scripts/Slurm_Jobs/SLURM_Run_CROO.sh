#!/bin/bash
#SBATCH --job-name=CROO_Organize         # Name of the job
#SBATCH --account=general              # Required parameter for Emory general partitions
#SBATCH --partition=c64-m512           # Default general-purpose CPU partition 
#SBATCH --nodes=1                      # Number of nodes requested
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=2              # CPUs for CROO
#SBATCH --time=04:00:00                # Time limit
#SBATCH --mem=8G                       # Memory
#SBATCH --output=%x_%j.out             # Output file: JobName_JobID.out
#SBATCH --error=%x_%j.err              # Error file: JobName_JobID.err

# Email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# Initialize Conda
conda init bash > /dev/null 2>&1
source ~/.bashrc

# Activate environment containing croo
conda activate /users/rprest2/.conda/envs/Enhancer-Creation

# Run the python wrapper script
python /scratch/rprest2/Enhancer-Creation/Scripts/Run_CROO.py
