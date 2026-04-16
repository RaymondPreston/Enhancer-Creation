import os
import json
import glob
import subprocess
from pathlib import Path

def main():
    base_dir = "/scratch/rprest2/Enhancer-Creation"
    atac_dir = os.path.join(base_dir, "atac")
    croo_out_dir = os.path.join(base_dir, "croo_out")
    
    os.makedirs(croo_out_dir, exist_ok=True)
    
    metadata_files = glob.glob(os.path.join(atac_dir, "*", "metadata.json"))
    
    if not metadata_files:
        print("No metadata.json files found in the atac directory.")
        return

    processed_count = 0
    
    for meta_file in metadata_files:
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            
            status = data.get('status')
            if status != 'Succeeded':
                print(f"Skipping {meta_file} - Status is '{status}'.")
                continue
                
            sample_name = "Unknown"
            # Try to get the sample name from the caper labels
            if 'submittedFiles' in data and 'labels' in data['submittedFiles']:
                try:
                    labels = json.loads(data['submittedFiles']['labels'])
                    # caper-str-label usually contains things like SRR31189893_input or KPC-2_Hi1_combined_input
                    sample_name = labels.get('caper-str-label', 'Unknown')
                    # Clean up the name
                    sample_name = sample_name.replace('_input', '')
                except json.JSONDecodeError:
                    pass
            
            # Fallback if label parsing didn't work
            if sample_name == "Unknown":
                uuid = os.path.basename(os.path.dirname(meta_file))
                sample_name = uuid
                
            out_dir = os.path.join(croo_out_dir, sample_name)
            
            if os.path.exists(out_dir):
                print(f"Output directory {out_dir} already exists. Skipping {sample_name}.")
                continue
                
            print(f"Running CROO for sample: {sample_name}...")
            
            cmd = [
                "croo",
                meta_file,
                "--out-dir", out_dir,
                "--method", "copy"
            ]
            
            # Run croo
            subprocess.run(cmd, check=True)
            print(f"Successfully finished CROO for {sample_name}\n")
            processed_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"CROO command failed for {meta_file}: {e}\n")
        except Exception as e:
            print(f"Error processing {meta_file}: {e}\n")

    print(f"Finished organizing {processed_count} samples with CROO.")

if __name__ == "__main__":
    main()
