#!/bin/bash
#SBATCH --job-name=SRA_Unpack_Missing  # Name of the job
#SBATCH --account=general              # Required parameter for Emory general partitions
#SBATCH --partition=c64-m512           # Default general-purpose CPU partition 
#SBATCH --nodes=1                      # Number of nodes requested
#SBATCH --ntasks=1                     # One task per array element
#SBATCH --cpus-per-task=4              # CPUs per file extraction
#SBATCH --time=12:00:00                # Time limit for a single file extraction
#SBATCH --mem=16G                      # Memory for fasterq-dump temporary buffers
#SBATCH --array=0-1                    # Launch 2 parallel tasks (for SRR31189867, SRR31192739)
#SBATCH --output=%x_%A_%a.out          # Output file: JobName_JobID_TaskID.out
#SBATCH --error=%x_%A_%a.err           # Error file: JobName_JobID_TaskID.err

# Email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# Initialize Conda
conda init bash > /dev/null 2>&1
source ~/.bashrc

# Activate environment
conda activate /users/rprest2/.conda/envs/Enhancer-Creation

# Execute the Python script for the specific array task ID
# The Python script will pick sra_files[SLURM_ARRAY_TASK_ID]
python /scratch/rprest2/Enhancer-Creation/Scripts/Unpack_SRA_Missing.py
