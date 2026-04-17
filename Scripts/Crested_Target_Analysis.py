import os
import glob
import anndata as ad
import crested
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import pearsonr, spearmanr

# Initialize mm10 genome
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv"
)
crested.register_genome(genome)

# Find all the bigwigs we copied
bigwig_folder = "input/CREsted_BigWig"
bigwig_paths = sorted(glob.glob(os.path.join(bigwig_folder, "*.bw")))

# Load diffbind fold changes for check
diffbind_df = pd.read_csv("output/diffbind_results.csv")

# Create a peak ID in the format expected by crested (chr:start-end)
# BED files (used by crested) are 0-based start, 1-based end. 
# DiffBind CSV is 1-based start, 1-based end. So we subtract 1 from start.
diffbind_df['peak_id'] = diffbind_df['seqnames'].astype(str) + ':' + (diffbind_df['start'] - 1).astype(str) + '-' + diffbind_df['end'].astype(str)

results = []

for target in ["max", "mean", "count"]:
    print(f"\n======================================")
    print(f"Analyzing target parameter: {target}")
    print(f"======================================")
    
    # 1. Import bigwigs using crested
    # We use crested.pp.import_bigwigs as requested by user snippet
    adata = crested.import_bigwigs(
        bigwigs_folder=bigwig_paths,
        regions_file="input/consensus_peaks_all.bed",
        target=target)

            
    print(f"Successfully loaded {adata.n_vars} peaks across {adata.n_obs} samples.")

    # Normalize peaks using crested
    crested.pp.normalize_peaks(adata, top_k_percent=0.03)
    print("Normalized peaks using crested.pp.normalize_peaks(top_k_percent=0.03)")

    # Assign condition to samples based on filename
    conditions = []
    for name in adata.obs_names:
        if "Hi" in name:
            conditions.append("Hi")
        elif "Lo" in name:
            conditions.append("Lo")
        else:
            conditions.append("Unknown")
    adata.obs["Condition"] = conditions

    # ---------------------------------------------------------
    # Check 1: CV in differential peaks
    # ---------------------------------------------------------
    print(f"\n[Check 1] Calculating CV across all peaks...")
    # Calculate Coefficient of Variation (std / mean) for each peak across all samples
    # Assuming adata.X is a dense or sparse matrix
    X_mat = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    means = np.mean(X_mat, axis=0)
    stds = np.std(X_mat, axis=0)

    valid_mask = means > 0
    cvs = np.zeros_like(means)
    cvs[valid_mask] = stds[valid_mask] / means[valid_mask]

    median_cv = np.median(cvs[valid_mask])
    print(f"Median CV for {target}: {median_cv:.4f}")

    # ---------------------------------------------------------
    # Check 2: PCA separation
    # ---------------------------------------------------------
    print(f"\n[Check 2] Performing PCA...")
    sc.tl.pca(adata, svd_solver='arpack')

    var_explained = np.sum(adata.uns['pca']['variance_ratio'][:2])
    print(f"Variance explained by PC1 + PC2 for {target}: {var_explained:.4f}")

    sc.pl.pca(adata, color='Condition', show=False, title=f"PCA - Target: {target}")
    plt.savefig(f"output/Crested_PCA_{target}.png")
    plt.close()
    print(f"Saved PCA plot to output/Crested_PCA_{target}.png")

    # ---------------------------------------------------------
    # Check 3: Concordance with DiffBind fold-changes
    # ---------------------------------------------------------
    print(f"\n[Check 3] Calculating concordance with DiffBind fold-changes...")

    hi_idx = np.where(adata.obs["Condition"] == "Hi")[0]
    lo_idx = np.where(adata.obs["Condition"] == "Lo")[0]

    # Calculate mean accessibility per condition (using pseudo-count for log2FC)
    hi_means = np.mean(X_mat[hi_idx, :], axis=0)
    lo_means = np.mean(X_mat[lo_idx, :], axis=0)

    # Fold > 0 in DiffBind means increased in Lo vs Hi.
    # Therefore, crested log2FC = log2(Lo_mean) - log2(Hi_mean)
    crested_fc = np.log2(lo_means + 1e-5) - np.log2(hi_means + 1e-5)

    # Map adata var_names to DiffBind peaks
    crested_df = pd.DataFrame({
        'peak_id': adata.var_names,
        'crested_fc': crested_fc
    })

    # Clean peak_ids in case they have extra characters
    crested_df['peak_id'] = crested_df['peak_id'].astype(str)

    merged_df = pd.merge(crested_df, diffbind_df, on='peak_id', how='inner')

    if len(merged_df) > 0:
        pearson_r, p_val = pearsonr(merged_df['Fold'], merged_df['crested_fc'])
        spearman_r, sp_val = spearmanr(merged_df['Fold'], merged_df['crested_fc'])
        print(f"Matched {len(merged_df)} peaks between Crested and DiffBind.")
        print(f"Pearson Correlation (r): {pearson_r:.4f} (p={p_val:.2e})")
        print(f"Spearman Correlation (rho): {spearman_r:.4f} (p={sp_val:.2e})")
    else:
        pearson_r = np.nan
        spearman_r = np.nan
        print("Warning: Could not match peak IDs between Crested output and DiffBind results.")
        print("Crested peak format:", adata.var_names[0] if len(adata.var_names)>0 else "None")
        print("DiffBind peak format:", diffbind_df['peak_id'].iloc[0] if len(diffbind_df)>0 else "None")
        
    results.append({
        'Target': target,
        'Median_CV': median_cv,
        'PCA_Var_Explained_PC1_PC2': var_explained,
        'DiffBind_Concordance_Pearson_r': pearson_r,
        'DiffBind_Concordance_Spearman_rho': spearman_r
    })

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n======================================")
print("FINAL SUMMARY COMPARISON")
print("======================================")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
summary_df.to_csv("output/Crested_Target_Comparison_Summary.csv", index=False)
print("\nSaved summary to output/Crested_Target_Comparison_Summary.csv")
