import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anndata as ad
import crested
import keras
from scipy.stats import pearsonr, spearmanr

#Load the genome
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

# SCRIPT NOT COMPLETE I NEED TO COMPLETE / FIX THIS CODE. Originally copied from Crested_training.py but removed because I don't want to focus on this tonight.

adata = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_training_set.h5ad")
adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_a_ft_training_set.h5ad") 

base_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/checkpoints/15.keras")
ft_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/TI_01_Finetune_Model_Training_v1/checkpoints/02.keras")

# ── Post-training evaluation ──────────────────────────────────────────────────

print("Evaluating model on held-out test chromosomes (chr9, chr18)...")
#Base model and FT model predictions on test set of adata_specific (cell type specific peaks)
#At some point, I should also compare base model and FT model on test set of adata (all peaks)
predictions_base = crested.tl.predict(adata_specific, base_model)
adata_specific.layers["Base model"] = predictions_base.T  # adata expects (classes, genes) instead of (genes, classes)
predictions_ft = crested.tl.predict(adata_specific, ft_model)
adata_specific.layers["Finetuned model"] = predictions_ft.T

crested.tl.evaluate(
    adata_specific,
    model='Finetuned model',
    metrics=crested.tl.default_configs('peak_regression_mean')
)


crested.tl.evaluate(
    adata_specific,
    model='Base model',
    metrics=crested.tl.default_configs('peak_regression_mean')
)

crested.pl.corr.violin(adata_specific)
plt.savefig("output/CREsted_Evaluation/Correlation_Violin_Plots.png")
plt.close()

crested.pl.corr.heatmap(
    adata_specific,
    split="test",
    log_transform=True,
    vmax=1,
    vmin=0,
)
plt.savefig("output/CREsted_Evaluation/Model_CellTypeHeatMap.png")
plt.close()