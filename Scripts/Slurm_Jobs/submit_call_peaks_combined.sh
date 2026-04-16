#!/bin/bash
#SBATCH --job-name=call_peaks_combined
#SBATCH --output=logs/call_peaks_combined_%j.out
#SBATCH --error=logs/call_peaks_combined_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=c64-m512

set -euo pipefail

# Parameters
SAMPLE_DIR="/scratch/rprest2/Enhancer-Creation/croo_out/KPC-2_Lo3_combined"
# Use the UNSHIFTED BAM (tagalign task will do the shift)
BAM_FILE="${SAMPLE_DIR}/KPC-2_Lo3_merged.sra_1.fastq.trim.srt.nodup.no_chrM_MT.bam"
# The tagalign task will produce a .tn5.tagAlign.gz file automatically
TA_FILE="${SAMPLE_DIR}/KPC-2_Lo3_merged.sra_1.fastq.trim.srt.nodup.no_chrM_MT.tn5.tagAlign.gz"
CHRSZ="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv"
GENSZ="mm"
BLACKLIST="/scratch/rprest2/indices/ENCFF547MET.bed.gz"
PIPELINE_SRC="/scratch/rprest2/Enhancer-Creation/atac-seq-pipeline/src"

# Create peak directory if it doesn't exist (matching pipeline structure)
OUT_DIR="${SAMPLE_DIR}/peak/combined"
mkdir -p $OUT_DIR

# 1. Convert BAM to TAGALIGN (using pipeline script)
# This will perform the Tn5 shift (+4/-5bp) automatically unless --disable-tn5-shift is passed
echo "Converting BAM to TAGALIGN and performing Tn5 shift..."
conda run -n encd-atac python3 ${PIPELINE_SRC}/encode_task_bam2ta.py \
    $BAM_FILE \
    --paired-end \
    --out-dir $SAMPLE_DIR \
    --nth $SLURM_CPUS_PER_TASK

# 2. Call Peaks using MACS2 (using pipeline script)
# NOTE: Using encd-atac-macs2 environment as macs2 was missing in encd-atac
echo "Calling peaks with MACS2..."
conda run -n encd-atac-macs2 python3 ${PIPELINE_SRC}/encode_task_macs2_atac.py \
    $TA_FILE \
    --chrsz $CHRSZ \
    --gensz $GENSZ \
    --pval-thresh 0.01 \
    --cap-num-peak 300000 \
    --smooth-win 150 \
    --out-dir $OUT_DIR

# 3. Post-process (Blacklist filtering, etc.)
# This generates the .bfilt.narrowPeak.gz file
RAW_PEAK=$(ls ${OUT_DIR}/*.narrowPeak.gz | grep -v "bfilt")
echo "Post-processing peaks (Blacklist filter)..."
conda run -n encd-atac python3 ${PIPELINE_SRC}/encode_task_post_call_peak_atac.py \
    $RAW_PEAK \
    --ta $TA_FILE \
    --peak-type narrowPeak \
    --chrsz $CHRSZ \
    --blacklist $BLACKLIST \
    --out-dir $OUT_DIR

echo "Done. Results in $OUT_DIR"
