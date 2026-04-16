#!/bin/bash
#SBATCH --job-name=tn5_shift_array
#SBATCH --output=logs/tn5_shift_%A_%a.out
#SBATCH --error=logs/tn5_shift_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=c64-m512
#SBATCH --array=0-24

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the file path for this specific task ID (0-indexed)
BAM_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" Scripts/bam_list.txt)
OUTPUT_BAM="${BAM_FILE%.bam}.tn5.bam"

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing BAM: $BAM_FILE"
echo "Output: $OUTPUT_BAM"

# Check if output already exists
if [ -f "$OUTPUT_BAM" ]; then
    echo "Output already exists, skipping."
    exit 0
fi

# 1. Index input if missing
if [ ! -f "${BAM_FILE}.bai" ] && [ ! -f "${BAM_FILE%.bam}.bai" ]; then
    echo "Indexing missing input BAM..."
    conda run -n encd-atac samtools index "$BAM_FILE"
fi

# 2. Perform Tn5 shift (+4/-5bp)
echo "Running alignmentSieve..."
# Note: alignmentSieve output is often unsorted/out of order after shifting
conda run -n encd-atac alignmentSieve \
    --numberOfProcessors $SLURM_CPUS_PER_TASK \
    --ATACshift \
    --bam "$BAM_FILE" \
    -o "${OUTPUT_BAM}.unsorted"

# 3. Sort the shifted BAM
echo "Sorting shifted BAM..."
conda run -n encd-atac samtools sort \
    -@ $SLURM_CPUS_PER_TASK \
    -o "$OUTPUT_BAM" \
    "${OUTPUT_BAM}.unsorted"

# 4. Index the shifted BAM
echo "Indexing shifted output..."
conda run -n encd-atac samtools index "$OUTPUT_BAM"

# Cleanup
rm "${OUTPUT_BAM}.unsorted"

echo "Done."
