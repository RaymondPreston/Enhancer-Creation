import os
import glob
import pandas as pd
import re
from collections import defaultdict

def main():
    rows = []
    # Find all sample directories in croo_out
    all_sample_dirs = sorted([d for d in glob.glob("croo_out/*") if os.path.isdir(d)])
    
    # Filter out parental
    sample_dirs = [d for d in all_sample_dirs if "parental" not in os.path.basename(d).lower()]
    
    # Group samples to identify technical replicates vs combined
    groups = defaultdict(list)
    for d in sample_dirs:
        name = os.path.basename(d)
        base_name = re.split(r'_(rep|combined|rep0)', name)[0]
        groups[base_name].append(d)

    final_dirs = []
    for base_name, dirs in groups.items():
        combined = [d for d in dirs if "combined" in os.path.basename(d)]
        if combined:
            final_dirs.extend(combined)
            print(f"Group {base_name}: Using combined sample only.")
        else:
            final_dirs.extend(dirs)
            print(f"Group {base_name}: Using all {len(dirs)} samples.")

    for d in final_dirs:
        sample_id = os.path.basename(d)
        
        # Find TN5 shifted BAM
        # Searching recursively within the sample dir for any file ending in .tn5.bam
        bam_matches = glob.glob(os.path.join(d, "**/*.tn5.bam"), recursive=True)
            
        if not bam_matches:
            print(f"Warning: No .tn5.bam file found for {sample_id}. Skipping.")
            continue
        bam_path = os.path.abspath(bam_matches[0])
        
        # Find Peaks
        # Searching recursively for the bfilt narrowPeak files
        peak_matches = glob.glob(os.path.join(d, "**/*.bfilt.narrowPeak.gz"), recursive=True)
            
        if not peak_matches:
            print(f"Warning: No narrowPeak file found for {sample_id}. Skipping.")
            continue
            
        peak_path = os.path.abspath(peak_matches[0])
        
        # Parse metadata
        parts = sample_id.split('_')
        tissue = parts[0]
        
        condition = "Unknown"
        if "Hi" in sample_id:
            condition = "Hi"
        elif "Lo" in sample_id:
            condition = "Lo"
            
        # Replicate
        replicate = "1"
        rep_match = re.search(r'(rep|rep0|Lo|Hi|combined)(\d*)', sample_id)
        if rep_match and rep_match.group(2):
            replicate = rep_match.group(2)
        else:
            replicate = "1"
        
        rows.append({
            "SampleID": sample_id,
            "Tissue": tissue,
            "Condition": condition,
            "Replicate": replicate,
            "bamReads": bam_path,
            "Peaks": peak_path,
            "PeakCaller": "narrowPeak"
        })
        
    df = pd.DataFrame(rows)
    df.to_csv("diffbind_sample_sheet.csv", index=False)
    print(f"\nGenerated filtered sample sheet with {len(df)} samples using TN5-shifted BAMs.")

if __name__ == "__main__":
    main()
