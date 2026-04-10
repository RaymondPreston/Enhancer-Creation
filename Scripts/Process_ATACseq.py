import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

# --- Configuration (Verified against Handler et al.) ---
BASE_DIR = "/scratch/rprest2/Enhancer-Creation"
INPUT_DIR = f"{BASE_DIR}/input/SRP435350"
OUTPUT_DIR = f"{BASE_DIR}/output/ATACseq"
METADATA_FILE = f"{BASE_DIR}/PRJNA960830_Metadata.csv"

# References
BOWTIE2_INDEX = "/scratch/rprest2/indices/mm10_Bowtie2/mm10"

# ENCODE Adapter sequences for auto-detection
ADAPTERS = {
    "Illumina": "AGATCGGAAGAGC",
    "Nextera": "CTGTCTCTTATA",
    "smallRNA": "TGGAATTCTCGG"
}

def detect_adapter(fastq_path, max_reads=250000):
    """Auto-detect adapter by counting occurrences in the first N reads (ENCODE method)."""
    print(f"Auto-detecting adapter for {fastq_path}...")
    counts = {name: 0 for name in ADAPTERS}
    
    try:
        with open(fastq_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_reads * 4:
                    break
                if i % 4 == 1:  # The sequence line in FASTQ
                    seq = line.strip()
                    for name, adapter in ADAPTERS.items():
                        if adapter in seq:
                            counts[name] += 1
    except Exception as e:
        print(f"Warning: Could not read {fastq_path} for adapter detection: {e}")
        print("Defaulting to Nextera adapter.")
        return ADAPTERS["Nextera"]

    best_adapter_name = max(counts, key=counts.get)
    best_adapter_seq = ADAPTERS[best_adapter_name]
    
    if counts[best_adapter_name] == 0:
        print("Warning: No known adapters detected. Defaulting to Nextera.")
        return ADAPTERS["Nextera"]
        
    print(f"Detected {best_adapter_name} adapter ({counts[best_adapter_name]} matches).")
    return best_adapter_seq

def run_cmd(cmd, shell=False):
    print(f"Executing: {' '.join(cmd) if not shell else cmd}")
    try:
        subprocess.run(cmd, check=True, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    df = pd.read_csv(METADATA_FILE)
    atac_df = df[df['Assay'].str.lower() == 'atac-seq'].sort_values('SRR')
    srr_list = atac_df['SRR'].tolist()

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

    # 1. Adapter Trimming (cutadapt v1.9.1)
    # Auto-detect adapter using ENCODE methodology
    detected_adapter = detect_adapter(str(fq1))
    
    fq1_trimmed = sample_out / f"{srr_id}_1_trimmed.fastq.gz"
    fq2_trimmed = sample_out / f"{srr_id}_2_trimmed.fastq.gz"
    run_cmd([
        "cutadapt", "-a", detected_adapter, "-A", detected_adapter,
        "-o", str(fq1_trimmed), "-p", str(fq2_trimmed),
        "-m", "30", "--cores=8",
        str(fq1), str(fq2)
    ])

    # 2. Alignment (Bowtie2 v2.2.6)
    bam_raw = sample_out / f"{srr_id}_raw.bam"
    # Mapping to mm10 with parameters for ATAC-seq
    align_cmd = f"bowtie2 -p 8 -X 2000 --very-sensitive -x {BOWTIE2_INDEX} -1 {fq1_trimmed} -2 {fq2_trimmed} | samtools view -bS - > {bam_raw}"
    run_cmd(align_cmd, shell=True)

    # 3. Mark Duplicates (Picard v1.126)
    bam_marked = sample_out / f"{srr_id}_marked.bam"
    metrics_file = sample_out / f"{srr_id}_dup_metrics.txt"
    run_cmd([
        "picard", "MarkDuplicates",
        f"I={bam_raw}", f"O={bam_marked}",
        f"M={metrics_file}", "REMOVE_DUPLICATES=false", "VALIDATION_STRINGENCY=LENIENT"
    ])

    # 4. Filtering (Samtools v1.7)
    # Filter: quality >= 30, remove mitochondrial (chrM), remove unmapped/secondary/duplicates
    bam_filtered = sample_out / f"{srr_id}_filtered.bam"
    filter_cmd = f"samtools view -h -q 30 -F 1804 {bam_marked} | grep -v 'chrM' | samtools sort -o {bam_filtered}"
    run_cmd(filter_cmd, shell=True)
    run_cmd(["samtools", "index", str(bam_filtered)])

    # 5. Peak Calling (MACS2 v2.1.0)
    # Using BAMPE for paired-end ATAC-seq
    run_cmd([
        "macs2", "callpeak",
        "-t", str(bam_filtered),
        "-f", "BAMPE",
        "-g", "mm",
        "-n", srr_id,
        "--outdir", str(sample_out),
        "-q", "0.05",
        "--nomodel", "--shift", "-100", "--extsize", "200"
    ])

    # Cleanup large intermediate files
    for f in [bam_raw, bam_marked, fq1_trimmed, fq2_trimmed]:
        if f.exists(): os.remove(f)

    print(f"Successfully processed ATAC-seq: {srr_id}")

if __name__ == "__main__":
    main()
