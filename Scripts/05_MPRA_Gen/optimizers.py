import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anndata as ad
import crested
import keras
from scipy.stats import pearsonr, spearmanr

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