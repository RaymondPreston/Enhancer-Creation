#!/bin/bash
#SBATCH --job-name=bam2bw_array
#SBATCH --output=logs/bam2bw_%A_%a.out
#SBATCH --error=logs/bam2bw_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=c64-m512
#SBATCH --array=0-24

# Create directories
mkdir -p logs
mkdir -p bigwigs

# Get the BAM file path from the manifest
BAM_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" Scripts/tn5_bam_list.txt)

# Extract sample name for the output file
# This removes the path and the long extension to keep the name clean
SAMPLE_NAME=$(basename "$BAM_FILE" .bam)
OUTPUT_BW="bigwigs/${SAMPLE_NAME}_cutsites.bw"

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing: $BAM_FILE"
echo "Output: $OUTPUT_BW"

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate encd-atac

# Run bamCoverage
bamCoverage \
  -b "$BAM_FILE" \
  -o "$OUTPUT_BW" \
  --normalizeUsing CPM \
  --binSize 1 \
  --Offset 1 \
  --samFlagExclude 256 \
  --numberOfProcessors $SLURM_CPUS_PER_TASK

echo "Done."
