from nis import cat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import anndata as ad
import crested
from scipy.stats import pearsonr, spearmanr
import pickle
import keras
import umap
import seaborn as sns
import re
from optimizers import mutli_class_weighted_differences, intra_line_variance_MWD, cosine_similarity_optimizer, bal_cosine_similarty_optimizer


# ----- Loading datasets, models, genome, and setting global vars -----
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad")
BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")

output_dir = "/scratch/rprest2/Enhancer-Creation/output/R1_MPRA_Generation"
os.makedirs(output_dir,exists_ok=True)

acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    adata_specific,  # accepts any sequence input, same as before
    per_position=True,  # return a distirbution per position in the sequence
)

# ----- Generating enhancers -----
optimizer_dict = {
        "MWD": mutli_class_weighted_differences,
        "Adjusted_MWD": intra_line_variance_MWD,
        "Cos_Similarity": cosine_similarity_optimizer,
}

standard_state_dict = {
        "met_high": hi_idx,
        "met_low": lo_idx,
}

boolean_state_dict={
        "met_high": cos_hi_array,
        "met_low": cos_lo_array
}

shared_kwargs = dict(
        model= BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
        acgt_distribution= acgt_distribution,
        return_intermediate= True,
        n_mutations= 20,
        n_sequences= 200,
        target_len = 200,
)

Adjusted_Hi_MWD_kwargs = dict(
        kpc1_hi_idx= kpc1_hi_idx,
        kpc1_lo_idx= kpc1_lo_idx,
        kpc2_hi_idx= kpc2_hi_idx,
        kpc2_lo_idx= kpc2_lo_idx,
        weight_multiplier=1, #Default=1
        variance_weight=0.25, #Default=0.5
)

Adjusted_Lo_MWD_kwargs = dict(
        kpc1_hi_idx= kpc1_lo_idx,
        kpc1_lo_idx= kpc1_hi_idx,
        kpc2_hi_idx= kpc2_lo_idx,
        kpc2_lo_idx= kpc2_hi_idx,
        weight_multiplier=2, #Default=1
        variance_weight=0.25, #Default=0.5
)

# Map all dicts to run_configs to avoid messy if else statements I previously had
run_configs = {
    "MWD": {
        "met_high": {"target": hi_idx, "kwargs": shared_kwargs},
        "met_low":  {"target": lo_idx, "kwargs": shared_kwargs}
    },
    "Adjusted_MWD": {
        # Dynamically merge the specific kwargs using the | operator
        "met_high": {"target": hi_idx, "kwargs": shared_kwargs | Adjusted_Hi_MWD_kwargs},
        "met_low":  {"target": lo_idx, "kwargs": shared_kwargs | Adjusted_Lo_MWD_kwargs}
    },
    "Cos_Similarity": {
        "met_high": {"target": cos_hi_array, "kwargs": shared_kwargs},
        "met_low":  {"target": cos_lo_array, "kwargs": shared_kwargs}
    }
}

#Need to add a cache checkpointing here to run once
for opt_name, opt_func in optimizer_dict.items():
        #Set the optimization function
        optimization_function = crested.tl.design.EnhancerOptimizer(optimize_func=opt_func)
        for state_name in ["met_high","met_low"]:
                print(f"Running ISE for {state_name} using {opt_name}")
                if opt_name=="Cos_Similarity":
                        current_target = boolean_state_dict[state_name]
                else:
                        current_target = standard_state_dict[state_name]
                if opt_name=="Adjusted_MWD" and state_name=="met_high":
                        current_kwargs = shared_kwargs | Adjusted_Hi_MWD_kwargs
                elif opt_name=="Adjusted_MWD" and state_name=="met_low":
                        current_kwargs = shared_kwargs | Adjusted_Lo_MWD_kwargs
                else:
                        current_kwargs = shared_kwargs
                intermediate_results, designed_sequences = crested.tl.design.in_silico_evolution(
                        target = current_target,
                        enhancer_optimizer= optimization_function,
                        **current_kwargs,
                )

