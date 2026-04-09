import os
import subprocess
from pathlib import Path

# Configuration
base_dir = "/scratch/rprest2/Enhancer-Creation/input/SRP435350"
output_base = "/scratch/rprest2/Enhancer-Creation/input/SRP435350" # Extracting into the same study folder

def unpack_sra(sra_file_path, output_dir):
    """Unpacks a single .sra file using fasterq-dump."""
    print(f"Processing: {sra_file_path}")
    
    # --split-files is standard for paired-end data; --skip-technical ignores technical reads
    # --progress shows progress in the logs
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
        return

    # Walk through the directory to find .sra files
    # Structure assumed: base_dir/SRRxxxxxx/SRRxxxxxx.sra
    sra_files = list(Path(base_dir).rglob("*.sra"))
    
    if not sra_files:
        print("No .sra files found.")
        return

    print(f"Found {len(sra_files)} SRA files. Starting extraction...")

    for sra_path in sra_files:
        # We'll put the resulting FASTQ files in the same folder as the .sra file
        # or you can adjust this to a global 'fastq' folder.
        output_dir = sra_path.parent
        unpack_sra(sra_path, output_dir)

    print("All tasks completed!")

if __name__ == "__main__":
    main()
