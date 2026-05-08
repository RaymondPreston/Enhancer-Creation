from crested.tl.modisco import read_motif_to_tf_file
import pandas as pd
import numpy as np
import os
import anndata as ad
import crested
import seaborn as sns
from optimizers import mutli_class_weighted_differences, intra_line_variance_MWD, cosine_similarity_optimizer
from utilities import calculate_kmer, calculate_library_diversity, kmer_freq_vector
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# ----- Loading datasets, models, genome, and setting global vars -----
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad")
ft_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")

output_dir = "/scratch/rprest2/Enhancer-Creation/output/R1_MPRA_Generation"
os.makedirs(output_dir, exist_ok=True)

acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    adata_specific,  # accepts any sequence input, same as before
    per_position=True,  # return a distribution per position in the sequence
)

#Load in meme database and TF motif file
meme_db = "/scratch/rprest2/Enhancer-Creation/input/motif_db/motif_db.meme"
motif_to_tf_file = read_motif_to_tf_file("/scratch/rprest2/Enhancer-Creation/input/motif_db/motif_tf_collection.tsv")

# ----- Generating enhancers: Setting up indexes for ISE targets -----
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

# Define float arrays for use in cosine similarity optimizer function
cos_hi_array = np.array([1.0 if "_Hi" in s else 0.0 for s in adata_specific.obs_names])
cos_lo_array = np.array([1.0 if "_Lo" in s else 0.0 for s in adata_specific.obs_names])

# ----- Generating enhancers: Dictionary setup for configs -----
optimizer_dict = {
        "MWD": mutli_class_weighted_differences,
        "Adjusted_MWD": intra_line_variance_MWD,
        "Cos_Similarity": cosine_similarity_optimizer,
}

shared_kwargs = dict(
        model=ft_model,
        acgt_distribution=acgt_distribution,
        return_intermediate=False,
        n_mutations=20,
        n_sequences=200,
        target_len=200,
)

Adjusted_Hi_MWD_kwargs = dict(
        kpc1_hi_idx=kpc1_hi_idx,
        kpc1_lo_idx=kpc1_lo_idx,
        kpc2_hi_idx=kpc2_hi_idx,
        kpc2_lo_idx=kpc2_lo_idx,
        weight_multiplier=1,   # Default=1
        variance_weight=0.25,  # Default=0.5
)

Adjusted_Lo_MWD_kwargs = dict(
        kpc1_hi_idx=kpc1_lo_idx,
        kpc1_lo_idx=kpc1_hi_idx,
        kpc2_hi_idx=kpc2_lo_idx,
        kpc2_lo_idx=kpc2_hi_idx,
        weight_multiplier=2,   # Default=1
        variance_weight=0.25,  # Default=0.5
)

# This maps all dicts to run_configs to avoid messy if-else statements
run_configs = {
    "MWD": {
        "met_high": {"target": hi_idx,        "kwargs": shared_kwargs},
        "met_low":  {"target": lo_idx,        "kwargs": shared_kwargs}
    },
    "Adjusted_MWD": {
        "met_high": {"target": hi_idx,        "kwargs": shared_kwargs | Adjusted_Hi_MWD_kwargs},
        "met_low":  {"target": lo_idx,        "kwargs": shared_kwargs | Adjusted_Lo_MWD_kwargs}
    },
    "Cos_Similarity": {
        "met_high": {"target": cos_hi_array,  "kwargs": shared_kwargs},
        "met_low":  {"target": cos_lo_array,  "kwargs": shared_kwargs}
    }
}

# ----- Generating enhancers: ISE Core loop -----
# Check cache to see if ISE generated file exists
save_path = os.path.join(output_dir, "ise_sequences_all.tsv")
if not os.path.exists(save_path):
        # Core loop for generating all ISE needed enhancers
        all_results = []
        for opt_name, state_configs in run_configs.items():
                # Set the optimization function
                optimization_function = crested.tl.design.EnhancerOptimizer(optimize_func=optimizer_dict[opt_name])
                for state_name, config_data in state_configs.items():
                        print(f"Running ISE for {state_name} using {opt_name}")
                        designed_sequences = crested.tl.design.in_silico_evolution(
                                target=config_data["target"],
                                enhancer_optimizer=optimization_function,
                                **config_data["kwargs"],
                        )

                        for i, seq in enumerate(designed_sequences):
                                all_results.append({
                                        "sequence_id":       f"{state_name}_{opt_name}_{i+1}",
                                        "sequence":          seq,
                                        "cell_state_target": state_name,
                                        "optimizer_used":    opt_name,
                                        "sequence_type":     "synthetic"
                                })
                print(f"--> Collected {len(designed_sequences)} sequences ({state_name}, {opt_name})")
                print(f"Total of {len(all_results)} sequences")

        # ----- Single save after all runs complete -----
        df_all = pd.DataFrame(all_results)
        df_all["core_sequence"] = df_all["sequence"].str.slice(957, 957 + 200)
        print("Sample core sequence length:", len(df_all["core_sequence"].iloc[0]))
        df_all.to_csv(save_path, sep="\t", index=False)
        print(f"\nAll runs complete. Saved {len(df_all)} total sequences to {save_path}")
        print(df_all.groupby(["cell_state_target", "optimizer_used"]).size())  # sanity check
        print("All runs complete and safely banked!")
else:
        df_all = pd.read_csv(save_path, sep="\t")
        print("Sample core sequence length:", len(df_all["core_sequence"].iloc[0]))
        print(f"{save_path} already exists. Skipping ISE & loading all sequences into a dataframe")
        print(df_all.groupby(["cell_state_target", "optimizer_used"]).size())  # sanity check


# ----- Predictions & Contribution Scores -----
# TO DO: I need to redo the contributions section. I should be doing contributions for met-high and met-low classes seperately.
contr_dir = os.path.join(output_dir, "Contribution_Scores")
os.makedirs(contr_dir, exist_ok=True)

n_classes = ft_model.output_shape[-1]  # e.g. 16
pred_cache = os.path.join(contr_dir, "predictions.npy")

if os.path.exists(pred_cache):
    print("Loading predictions from cache...")
    predictions = np.load(pred_cache)
    print(f"  predictions shape: {predictions.shape}")
else:
    print("Running predictions for the first time...")
    predictions = crested.tl.predict(
        input=df_all["sequence"].tolist(),
        model=ft_model,
    )
    np.save(pred_cache, predictions)
    print(f"Predictions saved → {pred_cache}  shape={predictions.shape}")

# --- Contribution scores --- 
contrib_class_files = [
    os.path.join(contr_dir, f"class_id_{i}_contrib.npz") for i in range(n_classes)
]
oh_class_files = [
    os.path.join(contr_dir, f"class_id_{i}_oh.npz") for i in range(n_classes)
]

all_contrib_exist = all(os.path.exists(f) for f in contrib_class_files)
all_oh_exist      = all(os.path.exists(f) for f in oh_class_files)

if all_contrib_exist and all_oh_exist:
    print(f"Loading contribution scores from {n_classes} cached class files...")
    contrib_list  = [np.load(f)["arr_0"] for f in contrib_class_files]
    one_hot_list  = [np.load(f)["arr_0"] for f in oh_class_files]
    # Each element: (n_seqs, 4, seq_len) → stack along new axis → (n_seqs, n_classes, 4, seq_len)
    contrib_scores = np.stack(contrib_list, axis=1)
    one_hot_seqs   = one_hot_list[0]   # identical across classes; just keep one
    print(f"  contrib_scores shape: {contrib_scores.shape}")
    print(f"  one_hot_seqs shape:   {one_hot_seqs.shape}")
else:
    print("Running contribution scores for the first time (this will take a while)...")
    contrib_scores, one_hot_seqs = crested.tl.contribution_scores(
        input=df_all["sequence"].tolist(),
        target_idx=None,
        model=ft_model,
        method="integrated_grad",
        batch_size=256,
        transpose=False,
        output_dir=contr_dir,   # CREsted writes class_id_N_contrib.npz files here
    )
    # contrib_scores shape after return: (n_seqs, n_classes, 4, seq_len)
    print(f"  contrib_scores shape: {contrib_scores.shape}")
    print(f"  one_hot_seqs shape:   {one_hot_seqs.shape}")


# ---- Running TFMoDisco on the sequences -----
'''
#I should run tfmodisco on met_high 
os.makedirs(f"{output_dir}/tfmodisco", exist_ok=True)
if not os.path.exists(f"{output_dir}/Contribution_Scores/KPC-1_Hi1_report"):
    print("Running tfmodisco on contribution scores")
    crested.tl.modisco.tfmodisco(
        window = 100, #Core ISEs are 200bp, this gives 200bp window
        output_dir=f"{output_dir}/tfmodisco",
        contr_dir=f"{output_dir}/Contribution_Scores",
        report=True,
        meme_db=meme_db,
        max_seqlets=20000,
    )
else:
    "tfmodisco has already been run. Continuing..."

# NEED TO FINISH


'''
# ---- Calculating Strength & Specificity -----

if "Log2FC" not in df_all.columns:
    print("Calculating ISE sequence strength and Log2FC")
    #predictions.shape (1200, 16)
    mh_strength = np.mean(predictions[:,hi_idx], axis=1) #Shape is (1200,)
    print(f"Shape of mh_strength:{mh_strength.shape}") 
    ml_strength= np.mean(predictions[:,lo_idx], axis=1)  #Shape is (1200,)
    specificity = np.log2(mh_strength / ml_strength) #Shape is (1200,)
    print(f"First few rows of log2fc is: {specificity}")


    df_all["Met_High Strength"] = mh_strength
    df_all["Met_Low Strength"] = ml_strength
    df_all["Log2FC"] = specificity

    print(df_all[["sequence_id", "cell_state_target", "Log2FC"]].head())
    df_all.to_csv(save_path, sep="\t", index=False)
else:
    print("ISE sequence strength and Log2FC already calculated... Proceeding to next step")
    print(df_all[["sequence_id", "cell_state_target", "Log2FC"]].head())


# ---- Calculating K-mer & Shannon Entropy -----
if "max_6mer_freq" not in df_all.columns:
    print("Calculating 6-mer diversity scores...")
    df_all[['max_6mer_freq', '6mer_entropy']] = df_all['core_sequence'].apply(
        lambda seq: pd.Series(calculate_kmer(seq, k=6))
    )
    print("Scoring complete! Look at the summary stats:")
    print(df_all[['max_6mer_freq', '6mer_entropy']].describe())
    df_all.to_csv(save_path, sep="\t", index=False)
else:
    print("k-mer stats already computed")
    print(df_all[['max_6mer_freq', '6mer_entropy']].describe())

# ---- Calculating Sequence Diversity & Novelty -----

# Calculating Synthethic Sequence Diversity Scores
if "Sequence_Diversity" not in df_all.columns:
    print("Calculating Core sequence diversity and novelty scores")
    seq_diversity = calculate_library_diversity(df_all["core_sequence"].tolist())
    df_all["Sequence_Diversity"] = seq_diversity
    df_all.to_csv(save_path, sep="\t", index=False)
    print("Sequence identity score is now computed. Loading summary stats")
    print(df_all.groupby("cell_state_target")["Sequence_Diversity"].describe())
else:
    print("Sequence identity score already computed. Loading summary stats")
    print(df_all.groupby("cell_state_target")["Sequence_Diversity"].describe())

# Calculate Sequence Novelty score
# Novelty = cosine distance between ISE 6-mer profile and nearest training peak 6-mer profile.
# Score [0, 1]: 0 = identical to a training peak, 1 = completely novel.
if "Novelty_Score" not in df_all.columns:
    print("Calculating novelty scores (k-mer cosine distance to nearest training peak)...")
    
    # adata_specific.var_names are region strings like "chr1:1000000-1002114"
    training_seqs = crested.utils.fetch_sequences(list(adata_specific.var_names))

    training_seqs = [s[957:957+200] for s in training_seqs]     # Slice the 200 bp core
    print(f"Extracted {len(training_seqs)} training peak sequences, core length: {len(training_seqs[0])} bp")

    print(f"  Building k-mer matrix for {len(training_seqs)} training peaks...")
    train_kmer_mat = np.vstack([kmer_freq_vector(s) for s in training_seqs])  # (n_peaks, 4096)

    print(f"  Building k-mer matrix for {len(df_all)} ISE sequences...")
    ise_kmer_mat = np.vstack([kmer_freq_vector(s) for s in df_all["core_sequence"]])  # (1200, 4096)

    # For each ISE sequence, find cosine distance to nearest training peak
    # Process in batches to avoid memory issues
    batch_size = 200
    min_distances = []
    for start in range(0, len(ise_kmer_mat), batch_size):
        batch = ise_kmer_mat[start:start+batch_size]
        dists = cosine_distances(batch, train_kmer_mat)  # (batch, n_peaks)
        min_distances.extend(dists.min(axis=1).tolist())
        print(f"  Processed {min(start+batch_size, len(ise_kmer_mat))}/{len(ise_kmer_mat)}")

    df_all["Novelty_Score"] = min_distances
    df_all.to_csv(save_path, sep="\t", index=False)
    print("Novelty scores saved. Summary:")
    print(df_all.groupby("cell_state_target")["Novelty_Score"].describe())
else:
    print("Novelty scores already computed")
    print(df_all.groupby("cell_state_target")["Novelty_Score"].describe())

# ---- Bar plots of 20 random ISE enhancers across classes -----
met_high_samples = df_all[df_all['cell_state_target'] == 'met_high'].sample(n=10, random_state=42)
met_low_samples  = df_all[df_all['cell_state_target'] == 'met_low'].sample(n=10, random_state=42)

# Extract your target class names (subclones) from the AnnData object
target_classes = list(adata_specific.obs_names)

def plot_enhancer_subset(sample_df, target_name, filename):
    """Generates a 5x2 grid of CREsted prediction bar plots."""
    
    # Create the 5 rows by 2 columns figure
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20), layout='constrained')
    
    # Flatten the 5x2 matrix into a simple 1D array of 10 slots so we can easily loop over it
    axes = axes.flatten() 
    
    print(f"Generating local prediction plots for {target_name}...")
    
    for i, (index, row) in enumerate(sample_df.iterrows()):
        seq = row['sequence'] 
        seq_id = row['sequence_id']
        
        # 1. Run the prediction
        prediction = crested.tl.predict(seq, model=ft_model)
        
        # 2. Plot the bar chart into the specific subplot (ax=axes[i])
        crested.pl.region.bar(
            prediction, 
            classes=target_classes,
            title=f"{target_name} Target | {seq_id}", 
            ax=axes[i], 
            show=False
        )
        
    # Add a main title for the whole 10-panel figure
    fig.suptitle(f"Sampled {target_name} Enhancer Predictions", fontsize=24, fontweight='bold')
    
    # Save and show
    save_path = f"{output_dir}/{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"--> Saved plot to {save_path}\n")

if os.path.exists("/scratch/rprest2/Enhancer-Creation/output/R1_MPRA_Generation/QC_Sample_MetHigh_Predictions.png"):
    print(" Bar plots of 20 random ISE enhancers across classes already exist. No need to run")
else:
    plot_enhancer_subset(met_high_samples, "Met-High", "QC_Sample_MetHigh_Predictions")
    plot_enhancer_subset(met_low_samples, "Met-Low", "QC_Sample_MetLow_Predictions")

# ---- Met-high vs Met-low Histograms -----
STATE_COLORS = {"met_high": "#D62728", "met_low": "#1F77B4"}

metrics = [
    ("Met_High Strength", "Met-High Prediction Strength",  "Mean predicted accessibility (Hi classes)"),
    ("Met_Low Strength",  "Met-Low Prediction Strength",   "Mean predicted accessibility (Lo classes)"),
    ("Log2FC",            "Specificity (Log₂FC)",          "log₂(Met-High / Met-Low strength)"),
    ("max_6mer_freq",     "Max 6-mer Frequency",           "Most repeated 6-mer count (200 bp core)"),
    ("6mer_entropy",      "6-mer Shannon Entropy",         "Sequence diversity (bits)"),
    ("Sequence_Diversity","Inter-sequence Diversity",      "Mean edit distance to all other sequences (bp)"),
    ("Novelty_Score",     "Novelty Score",                 "100 − BLASTn best-hit identity (%)"),
]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for ax, (col, title, xlabel) in zip(axes, metrics):
    for state, color in STATE_COLORS.items():
        
        # --- NEW FILTERING LOGIC ---
        # Skip plotting Met-Low enhancers on the Met-High Strength plot
        if col == "Met_High Strength" and state != "met_high":
            continue
        # Skip plotting Met-High enhancers on the Met-Low Strength plot
        if col == "Met_Low Strength" and state != "met_low":
            continue
        # ---------------------------

        subset = df_all[df_all["cell_state_target"] == state][col].dropna()
        sns.histplot(
            subset,
            ax=ax,
            color=color,
            label=state,
            kde=True,
            bins=40,
            alpha=0.5,
            linewidth=0,
        )
        
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    
    # Only show legend if there is more than 1 item plotted (so it skips the first 2 plots)
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 1:
        ax.legend(fontsize=8, framealpha=0.8)
        
    sns.despine(ax=ax)

axes[-1].set_visible(False)  # hide unused 8th panel

fig.suptitle(
    "ISE sequence quality metrics — met-high vs. met-low\n(200 bp core, 1,200 sequences total)",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()

out_png = os.path.join(output_dir, "ise_metric_histograms.png")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {out_png}")

# ---- Plot metrics by optimizer Violin plots ----

OPT_ORDER = ["MWD", "Adjusted_MWD", "Cos_Similarity"]
OPT_PALETTE = {
    "MWD":           "#2CA02C",
    "Adjusted_MWD":  "#FF7F0E",
    "Cos_Similarity":"#9467BD",
}
fig1, axes1 = plt.subplots(2, 4, figsize=(24, 12))
axes1 = axes1.flatten()

for ax, (col, title, ylabel) in zip(axes1, metrics):
    sns.violinplot(
        data=df_all,
        x="optimizer_used",
        y=col,
        hue="cell_state_target",
        order=OPT_ORDER,
        hue_order=["met_high", "met_low"],
        palette=STATE_COLORS,
        split=True,          # mirrored halves per optimizer
        inner="quart",       # show quartile lines inside violin
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(OPT_ORDER, rotation=25, ha="right", fontsize=9)
    ax.legend(title="Cell state", fontsize=8, title_fontsize=8,
              loc="upper right", framealpha=0.8)
    sns.despine(ax=ax)

axes1[-1].set_visible(False)

fig1.suptitle(
    "ISE metrics by optimizer — met-high vs. met-low (violin)",
    fontsize=14, fontweight="bold", y=1.01,
)
fig1.tight_layout()
out1 = os.path.join(output_dir, "ise_optimizer_violin.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")

# ---- TF-MINDI Analysis -----
