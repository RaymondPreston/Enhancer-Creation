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

BM_01TS_prmean_2114_nonorm_ep10 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm/checkpoints/10.keras")
BM_01TS_prmean_2114_nonorm_ep10__FT_DA8414 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")
BM_01TS_prmean_2114__nonorm_ep10__FT_Gini1 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_Gini-1_LR1e-4/checkpoints/04.keras")


'''



'''
# ----- Create optimizer function 1 for ISE ----
