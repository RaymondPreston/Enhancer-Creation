#!/bin/bash

# --- Configuration ---
BASE_DIR="/scratch/rprest2/Enhancer-Creation"
WDL_PATH="$BASE_DIR/atac-seq-pipeline/atac.wdl"
JSON_DIR="$BASE_DIR/input_jsons"

# The missing samples we unpacked
TARGET_SRRS=("SRR31189867" "SRR31192739")

echo "Starting ENCODE pipeline submission for missing ATAC-seq samples..."

# Ensure we are in the base directory
cd "$BASE_DIR" || exit 1

for srr in "${TARGET_SRRS[@]}"; do
    JSON_FILE="$JSON_DIR/${srr}_input.json"
    
    if [ -f "$JSON_FILE" ]; then
        echo "Submitting $srr to SLURM via Caper..."
        # Using the same Caper command formatting as your batch script
        caper hpc submit "$WDL_PATH" -i "$JSON_FILE" --conda --leader-job-name "CAPER_ENCODE_$srr"
        
        echo "Submission successful for $srr."
        echo "------------------------------------------------"
        
        # A brief 5-second pause to ensure SLURM registers the job before the next one
        sleep 5
    else
        echo "Error: JSON configuration not found for $srr at $JSON_FILE"
    fi
done

echo "Finished submitting missing samples."
