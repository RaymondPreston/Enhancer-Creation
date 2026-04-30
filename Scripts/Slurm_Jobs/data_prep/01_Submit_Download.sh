#!/bin/bash
#SBATCH --job-name=SRA_Download        # Name of the job [cite: 559]
#SBATCH --account=general              # Required parameter; 'general' is used for non-A100 partitions [cite: 560, 591]
#SBATCH --partition=c64-m512           # Default general-purpose CPU partition 
#SBATCH --nodes=1                      # Number of nodes requested [cite: 561]
#SBATCH --ntasks=1                     # Number of tasks [cite: 562]
#SBATCH --time=04:00:00                # Time requested (default max is 7-00:00:00) [cite: 580]
#SBATCH --mem=4G                       # Memory needed (Required parameter) [cite: 556, 576]
#SBATCH --output=%x_%j.out             # Standard output file (JobName_JobNumber.out) [cite: 573]
#SBATCH --error=%x_%j.err              # Standard error file (JobName_JobNumber.err) [cite: 574]

# Send email notifications (Optional: replace with your email if desired) [cite: 585]
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# Initialize Conda for Bash shell [cite: 588]
conda init bash > /dev/null 2>&1
source ~/.bashrc                       # Source the .bashrc file [cite: 587]

# Activate your specific conda environment (Uncomment and replace 'myenv' if you use one) 
conda activate Enhancer-Creation

# Execute the Python script
python /scratch/rprest2/Enhancer-Creation/Scripts/01_Data_Processing/01_Download_Data.py

# Execute the Python script for the specific array task ID
# The Python script will pick sra_files[SLURM_ARRAY_TASK_ID]
python /scratch/rprest2/Enhancer-Creation/Scripts/01_Data_Processing/02_Unpack_SRA.py