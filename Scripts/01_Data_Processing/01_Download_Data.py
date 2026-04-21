import pysradb

sra_web = pysradb.SRAweb()
project_id = "PRJNA960830"  # Your Project ID

print(f"Fetching detailed metadata for {project_id}...")
# Step 1: Fetch the DETAILED metadata dataframe for the project
df = sra_web.sra_metadata(project_id, detailed=True)

print(f"Found {len(df)} runs. Starting download...")
# Step 2: Pass skip_confirmation=True to bypass [Y/n] prompts in SLURM
sra_web.download(
    df=df, 
    out_dir="/scratch/rprest2/Enhancer-Creation/input", 
    skip_confirmation=True
)
print("Download complete!")