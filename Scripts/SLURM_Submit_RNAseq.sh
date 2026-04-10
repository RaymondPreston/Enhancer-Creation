#!/bin/bash
#SBATCH --job-name=RNAseq_Pipeline
#SBATCH --account=general
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # STAR works best with multiple cores
#SBATCH --time=12:00:00
#SBATCH --mem=64G                      # STAR requires high memory for genome index
#SBATCH --array=0-25                   # 26 RNA-seq samples
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
python /scratch/rprest2/Enhancer-Creation/Scripts/Process_RNAseq.py
