import anndata as ad
import pandas as pd
import pyfaidx

# Load genome FASTA
genome = pyfaidx.Fasta("/path/to/mm10.fa")

# Load data_set with DA specific peaks
adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad") 


# Get DA peak coordinates from adata_specific
# adata_specific.var_names are in format "chr:start-end"
records = []
for region, row in adata_specific.var.iterrows():
    chrom, coords = region.split(":")
    start, end = coords.split("-")
    start, end = int(start), int(end)
    
    # Extract sequence from genome
    seq = str(genome[chrom][start:end]).upper()
    
    # Skip if sequence contains N's (optional but recommended)
    if "N" in seq:
        continue
    
    records.append({
        "chr":      chrom,
        "start":    start,       # optional but useful
        "end":      end,         # optional but useful
        "sequence": seq,
        "TAG":      row["da_class"],  # "met_high" or "met_low"
    })

df = pd.DataFrame(records)
print(df["TAG"].value_counts())
print(f"Total peaks: {len(df)}")
print(df.head())

# Save as TSV
df[["chr", "sequence", "TAG"]].to_csv(
    "kpc_atac_dna_diffusion.tsv", sep="\t", index=False
)