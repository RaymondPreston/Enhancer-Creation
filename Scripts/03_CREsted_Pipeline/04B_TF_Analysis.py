import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anndata as ad
import crested
import keras
from scipy.stats import pearsonr, spearmanr

# ----- Loading datasets, models, and genome -----
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad") 

BM_02TS_prmean_2114_nonorm_ep10 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm/checkpoints/10.keras")
BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")
BM_02TS_prmean_2114_nonorm_ep10__FT_Gini1 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_Gini-1_LR1e-4/checkpoints/04.keras")


# ----- Generate model predictions on Met specific peaks -----
#Information regarding adata_specific:
#AnnData object with n_obs × n_vars = 16 × 8414
#    obs: 'file_path'
#    var: 'chr', 'start', 'end', 'split', 'da_class', 'fold', 'fdr'

#Make directory for TF output
os.makedirs("/scratch/rprest2/Enhancer-Creation/output/modisco_results", exist_ok= True)

# Rename da_class to Class name as required by modisco
adata_specific.var.rename(columns={"da_class": "Class name"}, inplace=True)

# Store predictions for all our regions in the adata_specific object
predictions = crested.tl.predict(adata_specific, BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414)
adata_specific.layers["Finetuned on DA Peaks"] = predictions.T

# Calculate the average of the ground truth and predictions
adata_specific.layers['combined'] = (adata_specific.X + adata_specific.layers["Finetuned on DA Peaks"])/2

#Calculate all class contribution scores
crested.tl.contribution_scores(
    input=adata_specific,                  # all DA peaks, all samples
    target_idx= None,                      # or just None for all classes
    model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
    method="expected_integrated_grad",
    transpose=True,
    output_dir="/scratch/rprest2/Enhancer-Creation/output/modisco_results",
    all_class_names=list(adata_specific.obs_names),
    batch_size=128,
)

