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

adata = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_a_ft_training_set.h5ad")
adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad") 

BM_01TS_prmean_2114_ep12 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_01TS_prmean_2114/checkpoints/12.keras")
BM_01TS_prmean_2114_ep12__FT_DA8414 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_01TS_prmean_2114_ep12__FT_DA8414_LR1e-4/checkpoints/03.keras")
BM_01TS_prmean_2114_ep12__FT_Gini1 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_01TS_prmean_2114_ep12__FT_Gini-1_LR1e-4/checkpoints/02.keras")

# ── Post-training evaluation ──────────────────────────────────────────────────

print("Evaluating model on held-out test chromosomes (chr9, chr18)...")
#Base model and FT model predictions on test set of adata_specific (cell type specific peaks)
#At some point, I should also compare base model and FT models on test set of adata (all peaks)

predictions_base = crested.tl.predict(adata_specific, BM_01TS_prmean_2114_ep12)
adata_specific.layers["Base model"] = predictions_base.T  # adata expects (classes, genes) instead of (genes, classes)
predictions_ft_DA = crested.tl.predict(adata_specific, BM_01TS_prmean_2114_ep12__FT_Gini1)
adata_specific.layers["Finetune on Gini=1"] = predictions_ft_DA.T
predictions_ft_Gi = crested.tl.predict(adata_specific, BM_01TS_prmean_2114_ep12__FT_DA8414)
adata_specific.layers["Finetune on DA peaks"] = predictions_ft_Gi.T

#Evaluate each model on the test set of adata_specific
crested.tl.evaluate(
    adata_specific,
    model='Base model',
    metrics=crested.tl.default_configs('peak_regression_mean')
)

crested.tl.evaluate(
    adata_specific,
    model='Finetune on Gini=1',
    metrics=crested.tl.default_configs('peak_regression_mean')
)

crested.tl.evaluate(
    adata_specific,
    model='Finetune on DA peaks',
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

crested.pl.region.bar(
    data = adata_specific,
    region= "chr3:135711468-135713582",
    pred_color= "lightblue",
    truth_color= "blue"
)
plt.savefig("output/CREsted_Evaluation/Region_Bar_Plot.png")