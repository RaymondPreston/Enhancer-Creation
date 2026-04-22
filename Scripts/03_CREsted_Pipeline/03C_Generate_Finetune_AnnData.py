"""
03C_Generate_Finetune_AnnData.py
----------------------------
Subsets the base training AnnData (01_training_set.h5ad) to exactly the
8,414 DESeq2 differentially accessible (DA) peaks.

Naming Convention for Downstream Modeling:
- PROJECT_NAME: KPC_Metastasis_Analysis
- RUN_NAME: [DataSubset]_[ModelType]_[Version]
    - DataSubset: DApeaks (Differentially Accessible)
    - ModelType: DilatedCNN
    - Version: v1, v2...
Recommended for this run: "DApeaks_DilatedCNN_Finetune_v1"
"""

import os
import subprocess
import io
import anndata as ad
import pandas as pd

# ── Naming Convention ────────────────────────────────────────────────────────
PROJECT_NAME = "KPC_Metastasis_Analysis"
RUN_NAME     = "DApeaks_DilatedCNN_Finetune_v1"

# ── Paths ─────────────────────────────────────────────────────────────────────
ADATA_PATH   = "/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_training_set.h5ad"
DA_TSV_PATH  = "/scratch/rprest2/Enhancer-Creation/output/DA_peaks_for_finetune.tsv"
OUTPUT_PATH  = "/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad"

TMP_ADATA_BED   = "/tmp/adata_peaks.bed"
TMP_SUMMIT_BED  = "/tmp/da_summits.bed"

# ── 1. Load base AnnData and write peaks to BED ───────────────────────────────
print("Loading base AnnData...")
adata = ad.read_h5ad(ADATA_PATH)
print(f"  Loaded: {adata.n_vars} peaks x {adata.n_obs} samples")

# adata.var must have columns: chr, start, end (0-based half-open)
adata_bed = adata.var[["chr", "start", "end"]].copy()
adata_bed["name"] = adata.var.index  # peak ID as name column (e.g. chr1:3010677-3012791)
adata_bed.to_csv(TMP_ADATA_BED, sep="\t", header=False, index=False)
print(f"  Written adata peaks BED: {TMP_ADATA_BED}")

# ── 2. Load DESeq2 DA peaks and write summit BED ──────────────────────────────
print("\nLoading DESeq2 DA peaks...")
da = pd.read_csv(DA_TSV_PATH, sep="\t")
print(f"  Loaded: {len(da)} DA peaks  "
      f"({(da['da_class']=='met_high').sum()} met-high, "
      f"{(da['da_class']=='met_low').sum()} met-low)")

# DiffBind 501 bp peaks are centered on the summit.
# Summit position (0-based) = (start - 1) + 250
# where start is 1-based (as exported from R/GRanges).
da["summit_0based"] = (da["start"] - 1) + 250

summit_bed = pd.DataFrame({
    "chr":   da["seqnames"],
    "start": da["summit_0based"],
    "end":   da["summit_0based"] + 1,   # single-bp summit
    "name":  da["peak_id_1based"],
    "fold":  da["Fold"],
    "fdr":   da["FDR"],
    "da_class": da["da_class"],
})
summit_bed.to_csv(TMP_SUMMIT_BED, sep="\t", header=False, index=False)
print(f"  Written DA summit BED: {TMP_SUMMIT_BED}")

# ── 3. bedtools intersect: find adata peaks containing each DA summit ─────────
# -wa: return the adata peak entry
# -wb: return the DA summit entry
# Any overlap = the 2,114 bp adata peak window contains the 1 bp summit
print("\nRunning bedtools intersect (summit-in-adata-peak)...")
result = subprocess.run(
    ["bedtools", "intersect",
     "-a", TMP_ADATA_BED,
     "-b", TMP_SUMMIT_BED,
     "-wa", "-wb"],
    capture_output=True, text=True
)
if result.returncode != 0:
    raise RuntimeError(f"bedtools intersect failed:\n{result.stderr}")

cols = [
    "adata_chr", "adata_start", "adata_end", "adata_peak_id",
    "da_chr", "da_summit", "da_summit_end", "da_peak_id",
    "fold", "fdr", "da_class",
]
df = pd.read_csv(io.StringIO(result.stdout), sep="\t", header=None, names=cols)
print(f"  Total overlap rows:          {len(df)}")
print(f"  Unique adata peaks matched:  {df['adata_peak_id'].nunique()}")
print(f"  Unique DA peaks matched:     {df['da_peak_id'].nunique()} / {len(da)}")

# ── 4. Resolve ambiguity: one adata peak per DA peak ─────────────────────────
# When a DA summit falls in two overlapping 2,114 bp adata peaks, keep the
# adata peak whose center is closest to the summit.
df["adata_center"] = df["adata_start"] + 1057   # center of 2,114 bp window
df["dist_to_summit"] = (df["adata_center"] - df["da_summit"]).abs()

df_best = df.sort_values("dist_to_summit").groupby("da_peak_id", as_index=False).first()

print(f"\nAfter deduplication:")
print(f"  DA peaks with adata match:  {len(df_best)} / {len(da)}")
print(f"  Unique adata peaks used:    {df_best['adata_peak_id'].nunique()}")
print(f"  Distance to summit (bp) — mean: {df_best['dist_to_summit'].mean():.1f}, "
      f"median: {df_best['dist_to_summit'].median():.1f}, "
      f"max: {df_best['dist_to_summit'].max():.1f}")
print(f"\n  da_class breakdown:")
print(f"    met_high: {(df_best['da_class']=='met_high').sum()}")
print(f"    met_low:  {(df_best['da_class']=='met_low').sum()}")

# ── 5. Subset AnnData and annotate var ───────────────────────────────────────
print("\nSubsetting AnnData to DA peaks...")
selected_ids = df_best["adata_peak_id"].values
adata_ft = adata[:, adata.var.index.isin(selected_ids)].copy()

lookup = df_best.set_index("adata_peak_id")[["da_class", "fold", "fdr"]]
adata_ft.var["da_class"] = lookup["da_class"].reindex(adata_ft.var.index).values
adata_ft.var["fold"]     = lookup["fold"].reindex(adata_ft.var.index).values
adata_ft.var["fdr"]      = lookup["fdr"].reindex(adata_ft.var.index).values

print(f"  Final shape: {adata_ft.shape}  "
      f"({adata_ft.n_vars} peaks x {adata_ft.n_obs} samples)")

# ── 6. Sanity check ───────────────────────────────────────────────────────────
import numpy as np

hi_mask = adata_ft.obs_names.str.contains("_Hi")
lo_mask = adata_ft.obs_names.str.contains("_Lo")
mh_mask = adata_ft.var["da_class"] == "met_high"
ml_mask = adata_ft.var["da_class"] == "met_low"
X = adata_ft.X

print("\nSanity check — mean accessibility (should show expected directionality):")
print(f"  met-high peaks in Hi samples: {X[hi_mask, :][:, mh_mask].mean():.4f}")
print(f"  met-high peaks in Lo samples: {X[lo_mask, :][:, mh_mask].mean():.4f}  <- should be lower")
print(f"  met-low  peaks in Hi samples: {X[hi_mask, :][:, ml_mask].mean():.4f}  <- should be lower")
print(f"  met-low  peaks in Lo samples: {X[lo_mask, :][:, ml_mask].mean():.4f}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
adata_ft.write_h5ad(OUTPUT_PATH)
print(f"\nSaved: {OUTPUT_PATH}")
print("Done.")
