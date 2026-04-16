#!/bin/bash

# --- Configuration ---
BASE_DIR="/scratch/rprest2/Enhancer-Creation"
WDL_PATH="$BASE_DIR/atac-seq-pipeline/atac.wdl"
JSON_DIR="$BASE_DIR/input_jsons"

# The failed combined samples
TARGET_SAMPLES=(
    "KPC-2_Lo3_combined"
    "KPC-1_parental_combined"
)

echo "Starting ENCODE pipeline submission for failed combined ATAC-seq samples..."

# Ensure we are in the base directory
cd "$BASE_DIR" || exit 1

for sample in "${TARGET_SAMPLES[@]}"; do
    JSON_FILE="$JSON_DIR/${sample}_input.json"
    
    if [ -f "$JSON_FILE" ]; then
        echo "Submitting $sample to SLURM via Caper..."
        caper hpc submit "$WDL_PATH" -i "$JSON_FILE" --conda --leader-job-name "CAPER_$sample"
        
        echo "Submission successful for $sample."
        echo "------------------------------------------------"
        
        # A brief pause
        sleep 5
    else
        echo "Error: JSON configuration not found for $sample at $JSON_FILE"
    fi
done

echo "Finished submitting failed combined samples."
