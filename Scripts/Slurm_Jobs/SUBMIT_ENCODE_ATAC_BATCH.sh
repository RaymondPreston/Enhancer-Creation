#!/bin/bash

# --- Configuration ---
BASE_DIR="/scratch/rprest2/Enhancer-Creation"
METADATA="$BASE_DIR/PRJNA960830_Metadata.csv"
INPUT_BASE="$BASE_DIR/input/SRP435350"
GENOME_TSV="/scratch/rprest2/indices/mm10.tsv"
WDL_PATH="$BASE_DIR/atac-seq-pipeline/atac.wdl"
JSON_DIR="$BASE_DIR/input_jsons"

# Ensure output directory for JSONs exists
mkdir -p $JSON_DIR

echo "Starting ENCODE pipeline submission for ATAC-seq samples..."

# Iterate through the metadata CSV
# Skipping the header and filtering for ATAC-seq rows
grep -i "ATAC-seq" $METADATA | while IFS=, read -r srr srx name assay rest; do
    
    echo "Generating configuration for $srr ($name)..."
    
    JSON_FILE="$JSON_DIR/${srr}_input.json"
    
    # Create the input.json file for this sample
    # Using the path structure seen in your screenshot: {INPUT_BASE}/{SRX}/{SRR}.sra_1.fastq
    cat <<EOF > $JSON_FILE
{
    "atac.pipeline_type": "atac",
    "atac.genome_tsv": "$GENOME_TSV",
    "atac.paired_end": true,
    "atac.fastqs_rep1_R1": ["$INPUT_BASE/$srx/${srr}.sra_1.fastq"],
    "atac.fastqs_rep1_R2": ["$INPUT_BASE/$srx/${srr}.sra_2.fastq"],
    "atac.auto_detect_adapter": true,
    "atac.multimapping": 0,
    "atac.title": "ENCODE_ATAC_$srr"
}
EOF

    echo "Submitting $srr to SLURM via Caper..."
    # --conda uses the pipeline environments you just installed
    caper hpc submit $WDL_PATH -i $JSON_FILE --conda --leader-job-name "ENCODE_$srr"
    
    echo "Submission successful for $srr."
    echo "------------------------------------------------"
done

echo "All ATAC-seq samples have been submitted."
