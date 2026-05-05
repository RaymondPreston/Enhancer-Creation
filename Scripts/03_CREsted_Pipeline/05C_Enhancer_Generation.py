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


acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    adata_specific,  # accepts any sequence input, same as before
    per_position=True,  # return a distirbution per position in the sequence
)
acgt_distribution.shape

'''
I need to use crested.tl.design.in_silico_evolution() to generate synthethic enhancers
I first want to create an optimizer function that uses weighted differences and accepts a np.array of weights (1 or 0)

'''
# ----- Create optimizer function 1 for ISE: Multi Class MWD ----
def mutli_class_weighted_differences(
        mutated_predictions: np.ndarray,
        original_predictions: np.ndarray,
        target: np.ndarray, #Is a boolean array of shape (n_classes). 1 is target class, 0 is non-target class
        weight_multiplier: float = 1.0 #Scalar weight multiplier. When many off target classes present, use =2, when fewer off target classes, use =1
):
        """ 
        This function calculates the mean weighted difference for target classes vs. off target classes for each mutation index
        This function returns the highest scoring mutation index from a round of ISE
        Use weight_multiplier to negate mathmatical dilution due to large n of off target classes   
        """
        if len(original_predictions.shape) == 1:
                original_predictions = original_predictions[None]

        delta = mutated_predictions - original_predictions #Score each mutation based on predicted accessibility change for each mutation in each class_type. Shape is (n_mutations, n_classes)

        mean_target_delta = delta[:, target].mean(axis=1) #Takes the mean for each mutation index delta across the classes of interest. Axis=1 specifies across columns (whereas axis=0 specifies down the rows which would average together all mutation index)
        mean_bg_delta = delta[:, ~target].mean(axis=1) #Takes the mean for each mutation index delta across the other classes of interest

        score = mean_target_delta - (weight_multiplier * mean_bg_delta) #The penalty here is the number of off target classes. I think using a penalty of all classes (16) would result in too permissive of off target accessibility, so opting to use 
        return int(np.argmax(score))

# ----- Create optimizer function 2 for ISE: intra_line_variance_MWD ----
def intra_line_variance_MWD(
        mutated_predictions: np.ndarray,    # (n_mutations, n_classes)
        original_predictions: np.ndarray,   # (n_classes,)
        target: np.ndarray,                 # boolean (n_classes,) — True = target (Hi) class
        weight_multiplier: float = 1.0,     # scalar penalty on background mean.  When many off target classes present, use =2, when fewer off target classes, use =1
        variance_weight: float = 0.5,       # penalty on variance across target classes
        kpc1_hi_idx: np.ndarray = None,     # boolean (n_classes,) — KPC-1 Hi samples
        kpc1_lo_idx: np.ndarray = None,     # boolean (n_classes,) — KPC-1 Lo samples
        kpc2_hi_idx: np.ndarray = None,     # boolean (n_classes,) — KPC-2 Hi samples
        kpc2_lo_idx: np.ndarray = None,     # boolean (n_classes,) — KPC-2 Lo samples
):
        """
        Weighted difference optimizer with:
        - Intra-parental-line Hi vs Lo contrast (KPC-1 and KPC-2 scored separately, then averaged)
        - Variance penalty to prevent single samples from driving the mean
        """
        if original_predictions.ndim == 1:
                original_predictions = original_predictions[None]

        delta = mutated_predictions - original_predictions  # (n_mutations, n_classes)
        if kpc1_hi_idx is not None:
                # Intra-parental-line contrast
                kpc1_contrast = delta[:, kpc1_hi_idx].mean(axis=1) - (weight_multiplier * delta[:, kpc1_lo_idx].mean(axis=1))
                kpc2_contrast = delta[:, kpc2_hi_idx].mean(axis=1) - (weight_multiplier * delta[:, kpc2_lo_idx].mean(axis=1))
                mean_target_delta = kpc1_contrast + kpc2_contrast
        else:
                # Fallback: Global target mean minus global background mean
                print("Falling back to global means")
                target_gain = delta[:, target].mean(axis=1)
                bg_leak = delta[:, ~target].mean(axis=1)
                mean_target_delta = target_gain - (weight_multiplier * bg_leak)

        # Variance penalty across target (Hi) classes
        var_target_delta = delta[:, target].var(axis=1)

        score = mean_target_delta - (variance_weight * var_target_delta)
        return int(np.argmax(score))

# ----- Create optimizer function 3 for ISE: Cosine Similarity ----

def cosine_similarity_optimizer(
    mutated_predictions: np.ndarray,    # (n_mutations, n_classes)
    original_predictions: np.ndarray,   # (n_classes,) — unused but required by API
    target: np.ndarray,                 # (n_classes,) float — target accessibility profile
) -> int:
    """
    Select the mutation that maximizes cosine similarity between the
    predicted accessibility profile and the target profile.
    """
    # Normalize target vector once
    target_norm = target / (np.linalg.norm(target) + 1e-8)
    
    # Normalize each mutated prediction vector
    pred_norms = np.linalg.norm(mutated_predictions, axis=1, keepdims=True) + 1e-8
    pred_normalized = mutated_predictions / pred_norms
    
    # Cosine similarity: dot product of normalized vectors
    score = pred_normalized @ target_norm  # (n_mutations,)
    
    return int(np.argmax(score))
        
# ----- Create optimizer function 4 for ISE: Strength * Specificty Balanced Cosine Similarity  ----
def bal_cosine_similarty_optimizer(
        mutated_predictions: np.ndarray, #Shape: (n_mutations, n_classes)
        original_predictions: np.ndarray, #(n_classes,) This is unused but needs to be passed
        target: np.ndarray, #Define the target accessibility profile
):
        """
        This optimizer function should excel at acheiving the proper shape of accessiblity in the target classes.
        How does this work:
        Line 1: Euclidean normalization to create a target vector within euclidean space. Something to do with pythagoream theorem
        Line 2: Calculates each mutation index's vector length via pythagoream theorem. 
        Takes the length of each of the classes and uses this in pythagoream thereom to calculate a single unit for that mutation index which sums up it's magnitude
        Line 3: Divides each mutation index's class accessibility by the magnitude of that total mutation index. This erases the magnitude and leaves a ratio of the directionality for each class
        Line 4: Each mutation index is multiplied by it's directionality. A higher score means it matches the directionality of target classes. A lower score means it matches the directionality of the off target classes
        """
        # Normalize target vector once
        target_norm = target / (np.linalg.norm(target) + 1e-8)
        
        # Normalize each mutated prediction vector
        pred_norms = np.linalg.norm(mutated_predictions, axis=1, keepdims=True) + 1e-8
        pred_normalized = mutated_predictions / pred_norms
        
        # Cosine similarity: dot product of normalized vectors
        shape_score = pred_normalized @ target_norm  # (n_mutations,)

        # 2. Magnitude score — mean predicted accessibility in target samples
        target_mask = target > 0
        strength_scores = mutated_predictions[:, target_mask].mean(axis=1)

        # 3. Hybrid score
        score = shape_score * strength_scores

        return int(np.argmax(score))

# ----- Define Hi/Lo sample masks -----
hi_samples = [s for s in adata_specific.obs_names if "_Hi" in s]
lo_samples = [s for s in adata_specific.obs_names if "_Lo" in s]
hi_idx = np.array([s in hi_samples for s in adata_specific.obs_names])
lo_idx = np.array([s in lo_samples for s in adata_specific.obs_names])

print(f"Hi samples ({hi_idx.sum()}): {list(adata_specific.obs_names[hi_idx])}")
print(f"Lo samples ({lo_idx.sum()}): {list(adata_specific.obs_names[lo_idx])}")

# Define parental line masks
kpc1_hi_idx = np.array([s.startswith("KPC-1") and "_Hi" in s for s in adata_specific.obs_names])
kpc1_lo_idx = np.array([s.startswith("KPC-1") and "_Lo" in s for s in adata_specific.obs_names])
kpc2_hi_idx = np.array([s.startswith("KPC-2") and "_Hi" in s for s in adata_specific.obs_names])
kpc2_lo_idx = np.array([s.startswith("KPC-2") and "_Lo" in s for s in adata_specific.obs_names])

#Define float arrays for use in cosine similarity optimizer function
cos_hi_array = np.array([1.0 if "_Hi" in s else 0.0 for s in adata_specific.obs_names])
cos_lo_array = np.array([1.0 if "_Lo" in s else 0.0 for s in adata_specific.obs_names])

Optimizer_function = crested.tl.design.EnhancerOptimizer(optimize_func = cosine_similarity_optimizer)

met_hi_intermediate_results, met_hi_designed_sequences = crested.tl.design.in_silico_evolution(
        model= BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
        acgt_distribution= acgt_distribution,
        return_intermediate= True,
        target_len= 200,
        n_mutations= 20,
        n_sequences= 5,
        enhancer_optimizer= Optimizer_function,
        target= cos_hi_array,
)
met_lo_intermediate_results, met_lo_designed_sequences = crested.tl.design.in_silico_evolution(
        model= BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
        acgt_distribution= acgt_distribution,
        return_intermediate= True,
        target_len= 200,
        n_mutations= 20,
        n_sequences= 5,
        enhancer_optimizer=Optimizer_function,
        target= cos_lo_array,
)

# Check predictions for the designed sequences - ensure that they're high for our target class
os.makedirs("output/CREsted_ISE/Cos", exist_ok=True)
#Make plots for met-high generated enhancers:
print("Creating plots for generated enhancers")

fig, axs = plt.subplots(5, figsize = (15, 20), layout='constrained')
for i in range(len(met_hi_designed_sequences)):
    prediction = crested.tl.predict(met_hi_designed_sequences[i], model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414)
    crested.pl.region.bar(prediction, classes=list(adata_specific.obs_names), title=f"Designed enhancer {i+1}", ax=axs[i], show=False)
plt.savefig("output/CREsted_ISE/Cos/Comb_MH_Designed_enhancers.png", dpi=300)
plt.close()

crested.pl.design.step_predictions(
        met_hi_intermediate_results,
        target_classes=adata_specific.obs_names,
        obs_names=adata_specific.obs_names,
        separate=True,
        suptitle="Synthethic Met-High enhancers",
)
plt.savefig("output/CREsted_ISE/Cos/MH_Stepwise_Class_Predictions.png")
plt.close()

#Make plots for met-low generated enhancers:
fig, axs = plt.subplots(5, figsize = (15, 20), layout='constrained')
for i in range(len(met_lo_designed_sequences)):
    prediction = crested.tl.predict(met_lo_designed_sequences[i], model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414)
    crested.pl.region.bar(prediction, classes=list(adata_specific.obs_names), title=f"Designed enhancer {i+1}", ax=axs[i], show=False)
plt.savefig("output/CREsted_ISE/Cos/Comb_ML_Designed_enhancers.png", dpi=300)
plt.close()

crested.pl.design.step_predictions(
        met_lo_intermediate_results,
        target_classes=adata_specific.obs_names,
        obs_names=adata_specific.obs_names,
        separate=True,
        suptitle="Synthethic Met-Low enhancers",
)
plt.savefig("output/CREsted_ISE/Cos/ML_Stepwise_Class_Predictions.png")
plt.close()

# Save final designed sequences to CSV
met_hi_df = pd.DataFrame({'sequence': met_hi_designed_sequences, 'type': 'met_high'})
met_lo_df = pd.DataFrame({'sequence': met_lo_designed_sequences, 'type': 'met_low'})
all_sequences_df = pd.concat([met_hi_df, met_lo_df], ignore_index=True)
all_sequences_df.to_csv("output/CREsted_ISE/Cos/final_enhancer_sequences.csv", index=False)
print("Saved final enhancer sequences to output/CREsted_ISE/final_enhancer_sequences.csv")