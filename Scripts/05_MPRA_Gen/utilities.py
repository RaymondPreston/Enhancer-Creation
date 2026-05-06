import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anndata as ad
import crested
import keras
from scipy.stats import pearsonr, spearmanr

def scan_cwm(contrib_track, cwm, both_strands=True):
    """
    Slide a CWM across a 1D contribution track and return the max match score.
    
    Parameters
    ----------
    contrib_track : np.ndarray (L,)
        Per-position contribution scores for one sequence.
    cwm : np.ndarray (pattern_len, 4)
        Contribution weight matrix for one pattern.
    both_strands : bool
        Also scan reverse complement of CWM.
    
    Returns
    -------
    float : max cross-correlation score across all positions
    """
    # Collapse CWM to 1D (sum across bases, weighted by IC)
    cwm_1d = cwm.sum(axis=-1)   # (pattern_len,)
    
    # Cross-correlate CWM with contribution track
    scores_fwd = correlate(contrib_track, cwm_1d, mode="valid")
    max_score = scores_fwd.max()
    
    if both_strands:
        cwm_rc = cwm[::-1, ::-1]   # reverse complement
        cwm_rc_1d = cwm_rc.sum(axis=-1)
        scores_rev = correlate(contrib_track, cwm_rc_1d, mode="valid")
        max_score = max(max_score, scores_rev.max())
    
    return float(max_score)