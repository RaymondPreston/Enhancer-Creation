import os
import sys
import subprocess
from pathlib import Path

# Configuration
base_dir = "/scratch/rprest2/Enhancer-Creation/input/SRP435350"

def unpack_sra(sra_file_path, output_dir):
    """Unpacks a single .sra file using fasterq-dump."""
    print(f"Processing: {sra_file_path}")
    
    # Check if files already exist to avoid re-running (Fastq naming: SRRxxxxxx_1.fastq)
    srr_id = sra_file_path.stem
    if any(output_dir.glob(f"{srr_id}_*.fastq")):
        print(f"FASTQ files for {srr_id} already exist. Skipping.")
        return

    cmd = [
        "fasterq-dump",
        "--split-files",
        "--skip-technical",
        "--progress",
        "--outdir", str(output_dir),
        str(sra_file_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully unpacked to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error unpacking {sra_file_path}: {e}")

def main():
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        sys.exit(1)

    # Walk through the directory to find .sra files and sort them for consistent indexing
    sra_files = sorted(list(Path(base_dir).rglob("*.sra")))
    
    if not sra_files:
        print("No .sra files found.")
        sys.exit(0)

    # Check for SLURM_ARRAY_TASK_ID to run a single file in parallel
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    
    if task_id is not None:
        idx = int(task_id)
        if idx < len(sra_files):
            sra_path = sra_files[idx]
            output_dir = sra_path.parent
            unpack_sra(sra_path, output_dir)
        else:
            print(f"Task ID {idx} out of range for {len(sra_files)} files.")
    else:
        # Fallback to sequential if run manually
        print(f"No Task ID found. Processing {len(sra_files)} files sequentially...")
        for sra_path in sra_files:
            output_dir = sra_path.parent
            unpack_sra(sra_path, output_dir)

    print("Task completed.")

if __name__ == "__main__":
    main()
