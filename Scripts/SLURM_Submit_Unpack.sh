#!/bin/bash
#SBATCH --job-name=SRA_Unpack          # Name of the job
#SBATCH --account=general              # Required parameter for Emory general partitions
#SBATCH --partition=c64-m512           # Default general-purpose CPU partition 
#SBATCH --nodes=1                      # Number of nodes requested
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=4              # Unpacking is CPU-bound; fasterq-dump uses multiple threads
#SBATCH --time=08:00:00                # Requested 8 hours for batch processing
#SBATCH --mem=16G                      # fasterq-dump requires significant memory for its temporary buffers
#SBATCH --output=%x_%j.out             # Standard output file (JobName_JobNumber.out)
#SBATCH --error=%x_%j.err              # Standard error file (JobName_JobNumber.err)

# Email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# Initialize Conda for Bash shell
conda init bash > /dev/null 2>&1
source ~/.bashrc

# Activate your specific conda environment
# Ensure this environment has 'sra-tools' installed
conda activate /users/rprest2/.conda/envs/Enhancer-Creation

# Execute the Python unpacking script
# Pointing to the absolute path on the scratch partition
python /scratch/rprest2/Enhancer-Creation/Scripts/Unpack_SRA.py
