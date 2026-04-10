#!/bin/bash
#SBATCH --job-name=Build_mm10_Index
#SBATCH --account=general
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Init Conda
conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate /users/rprest2/.conda/envs/Enhancer-Creation

# Define directories
INDICES_DIR="/scratch/rprest2/indices"
BOWTIE_DIR="$INDICES_DIR/mm10_Bowtie2"

echo "Creating directories..."
mkdir -p $BOWTIE_DIR
cd $BOWTIE_DIR

echo "Downloading mm10 FASTA from UCSC..."
wget -qO mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz

echo "Unzipping mm10.fa.gz..."
gunzip mm10.fa.gz

echo "Building Bowtie2 index (this will take a while)..."
bowtie2-build --threads 8 mm10.fa mm10

echo "mm10 Bowtie2 index built successfully in $BOWTIE_DIR!"
