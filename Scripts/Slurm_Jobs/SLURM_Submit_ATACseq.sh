#!/bin/bash
#SBATCH --job-name=ATACseq_Pipeline
#SBATCH --account=general
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Bowtie2 uses multiple threads
#SBATCH --time=12:00:00
#SBATCH --mem=32G                      # Sufficient for Bowtie2 and MACS3
#SBATCH --array=0-29                   # 30 ATAC-seq samples
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rprest2@emory.edu 

# Init Conda
conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate /users/rprest2/.conda/envs/Enhancer-Creation

# Run Pipeline
python /scratch/rprest2/Enhancer-Creation/Scripts/Process_ATACseq.py
