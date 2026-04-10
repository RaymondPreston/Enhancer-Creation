#!/bin/bash
#SBATCH --job-name=Build_GRCm39_Salmon
#SBATCH --account=general
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Init Conda
conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate /users/rprest2/.conda/envs/salmon_env

# Define directories
INDICES_DIR="/scratch/rprest2/indices"
SALMON_DIR="$INDICES_DIR/GRCm39_Salmon_Index"

echo "Creating directories..."
mkdir -p $INDICES_DIR
cd $INDICES_DIR

echo "Downloading GRCm39 (M33) transcripts from GENCODE..."
wget -qO gencode.vM33.transcripts.fa.gz https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/gencode.vM33.transcripts.fa.gz

echo "Building Salmon index..."
# We use the transcript fasta directly to build the index
salmon index -t gencode.vM33.transcripts.fa.gz -i $SALMON_DIR -p 8

echo "GRCm39 Salmon index built successfully in $SALMON_DIR!"
