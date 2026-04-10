import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

# --- Configuration (Verified against Handler et al.) ---
BASE_DIR = "/scratch/rprest2/Enhancer-Creation"
INPUT_DIR = f"{BASE_DIR}/input/SRP435350"
OUTPUT_DIR = f"{BASE_DIR}/output/RNAseq"
METADATA_FILE = f"{BASE_DIR}/PRJNA960830_Metadata.csv"

# References
SALMON_INDEX = "/scratch/rprest2/indices/GRCm39_Salmon_Index" # GENCODE M33
ADAPTER_FASTA = "/users/rprest2/.conda/envs/Enhancer-Creation/share/trimmomatic/adapters/TruSeq3-PE.fa"

def run_cmd(cmd, shell=False):
    print(f"Executing: {' '.join(cmd) if not shell else cmd}")
    try:
        subprocess.run(cmd, check=True, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    df = pd.read_csv(METADATA_FILE)
    rna_df = df[df['Assay'].str.lower() == 'rna-seq'].sort_values('SRR')
    srr_list = rna_df['SRR'].tolist()

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        print("Error: Must run as SLURM array.")
        sys.exit(1)
    
    idx = int(task_id)
    if idx >= len(srr_list): return
    srr_id = srr_list[idx]
    
    sample_out = Path(OUTPUT_DIR) / srr_id
    sample_out.mkdir(parents=True, exist_ok=True)

    fq1 = Path(INPUT_DIR) / srr_id / f"{srr_id}_1.fastq"
    fq2 = Path(INPUT_DIR) / srr_id / f"{srr_id}_2.fastq"

    # 1. Adapter Trimming (Trimmomatic v0.39)
    # Using parameters standard for Illumina paired-end 150bp
    fq1_trimmed = sample_out / f"{srr_id}_1_trimmed.fastq"
    fq1_unpaired = sample_out / f"{srr_id}_1_unpaired.fastq"
    fq2_trimmed = sample_out / f"{srr_id}_2_trimmed.fastq"
    fq2_unpaired = sample_out / f"{srr_id}_2_unpaired.fastq"

    run_cmd([
        "trimmomatic", "PE", "-threads", "8",
        str(fq1), str(fq2),
        str(fq1_trimmed), str(fq1_unpaired),
        str(fq2_trimmed), str(fq2_unpaired),
        f"ILLUMINACLIP:{ADAPTER_FASTA}:2:30:10",
        "LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36"
    ])

    # 2. Quantification (Salmon v1.10.1)
    # Mapping-based mode with GC bias correction as per paper
    salmon_out = sample_out / "salmon_quant"
    run_cmd([
        "salmon", "quant",
        "-i", SALMON_INDEX,
        "-l", "A", # Auto library type
        "-1", str(fq1_trimmed), "-2", str(fq2_trimmed),
        "-p", "8",
        "--gcBias",
        "-o", str(salmon_out)
    ])

    # Cleanup temporary trimmed files to save space on scratch
    for f in [fq1_trimmed, fq1_unpaired, fq2_trimmed, fq2_unpaired]:
        if f.exists(): os.remove(f)

    print(f"Successfully processed RNA-seq: {srr_id}")

if __name__ == "__main__":
    main()
