import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anndata as ad
import crested
import keras
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.signal import correlate
from collections import Counter
import Levenshtein
import itertools
import seaborn as sns
from itertools import product

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

def calculate_kmer(seq, k=6):
    """
    Slides a window of size k across the sequence to extract all k-mers,
    then calculates the maximum frequency and Shannon entropy.
    """
    # 1. Extract overlapping k-mers
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    kmer_counts = Counter(kmers)
    
    # 2. Calculate the Max Frequency (How many times did the AI spam the same motif?)
    max_freq = max(kmer_counts.values()) if kmer_counts else 0
    
    # 3. Calculate Shannon Entropy (Overall diversity of the sequence)
    # Higher entropy = more natural/diverse. Lower entropy = repetitive noise.
    counts = list(kmer_counts.values())
    seq_entropy = entropy(counts, base=2)
    
    return max_freq, seq_entropy



def calculate_library_diversity(sequences):
    """
    For each sequence, compute its mean Levenshtein distance to all other sequences.
    Returns a list of length n_sequences (one mean distance per sequence).
    A higher score = more unique/isolated in sequence space.
    Also prints the library-level mean pairwise distance as a summary stat.
    """
    n = len(sequences)
    print(f"Calculating mean pairwise edit distances for {n} sequences (~{n*(n-1)//2:,} comparisons)...")
    
    per_seq_mean = np.zeros(n)
    for i in range(n):
        dists = [Levenshtein.distance(sequences[i], sequences[j]) for j in range(n) if j != i]
        per_seq_mean[i] = np.mean(dists)
    
    library_mean = np.mean(per_seq_mean)
    print(f"Library-level mean pairwise edit distance: {library_mean:.2f} bp")
    
    return per_seq_mean.tolist()  # length = n_sequences, one value per row in df_all

def kmer_freq_vector(seq, k=6):
    """Return a normalized k-mer frequency vector for a DNA sequence."""
    kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    vec = np.zeros(len(kmers), dtype=np.float32)
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_idx:
            vec[kmer_idx[kmer]] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec
