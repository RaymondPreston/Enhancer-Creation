import anndata as ad
import pandas as pd
import crested
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import keras


#Initialize mm10 genome. 
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

bigwig_folder = "input/CREsted_BigWig"
bigwig_paths = sorted(glob.glob(os.path.join(bigwig_folder, "*.bw")))

#Import the bigwig files for the cell types
#Need to move the desired bigwig files to the input/ directory
# Initially don't specifyc target_region_width to let it use default consensus. I can modify this later with crested.pp.change_regions_width
adata = crested.import_bigwigs(
    bigwigs_folder = bigwig_paths,
    regions_file = "input/consensus_peaks_all.bed",
    target = "mean")

print(f"Successfully loaded {adata.n_vars} peaks across {adata.n_obs} samples.")

#Change region width to 2114bp to reflect optimal training results. Further investigation of this function is needed to identify gain from use
crested.pp.change_regions_width(adata, width=2114)

os.makedirs("output/CREsted_PreProcess", exist_ok=True)

# Test which top_k_percent is optimal
top_k_values = [0.01, 0.02, 0.03, 0.05, 0.10]
scaling_factors = {}

X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
if X.shape[0] < X.shape[1]:
    X = X.T # Ensure shape is (n_peaks, n_samples)

for k in top_k_values:
    n_top = int(k * X.shape[0])
    # Mean accessibility per peak across all samples
    peak_means = X.mean(axis=1)
    top_idx = np.argsort(peak_means)[-n_top:]
    # Scaling factor per sample = mean of top-k peaks in that sample
    sf = X[top_idx, :].mean(axis=0)
    scaling_factors[k] = sf

# Plot scaling factor CV across samples for each top_k
cvs = {k: np.std(v) / np.mean(v) for k, v in scaling_factors.items()}

plt.figure(figsize=(6, 4))
plt.plot(list(cvs.keys()), list(cvs.values()), marker='o')
plt.xlabel("top_k_percent")
plt.ylabel("CV of scaling factors across samples")
plt.title("Scaling factor stability vs. top_k")
plt.savefig("output/CREsted_PreProcess/top_k_diagnostic.svg", bbox_inches="tight")
plt.close()

#Normalize peaks. top_k_percent=0.03 is what the CREsted tutorial uses, however I should look into this further
crested.pp.normalize_peaks(adata, top_k_percent=0.03)

#Need to save these graphs to output/CREsted_PreProcess/ 

crested.pl.qc.normalization_weights(adata, title="Post-Process-Normalization weights per cell type", xtick_rotation=90)
plt.savefig("output/CREsted_PreProcess/Post-Process-normalization_weights.png", bbox_inches="tight")
plt.close()

crested.pl.corr.heatmap_self(adata, log_transform=True, vmin=0, vmax=1, reorder=True)
plt.savefig("output/CREsted_PreProcess/Post-Process-correlation_heatmap.png", bbox_inches="tight")
plt.close()

crested.pp.train_val_test_split(
    adata,
    strategy="chr",
    val_chroms=["chr8", "chr10"],
    test_chroms=["chr9", "chr18"],
)
print(adata.var["split"].value_counts())
print(f"Data loaded: {adata.n_vars} peaks across {adata.n_obs} samples.")

# Save the final preprocessing results
os.makedirs("/scratch/rprest2/Enhancer-Creation/input/training_inputs", exist_ok=True)  
adata.write_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_training_set.h5ad")
