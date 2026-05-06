import pandas as pd
import numpy as np
import os
import anndata as ad
import crested
from scipy.stats import pearsonr, spearmanr
import pickle
import keras
import umap
import seaborn as sns
import re
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from optimizers import mutli_class_weighted_differences, intra_line_variance_MWD, cosine_similarity_optimizer
from utilities import scan_cwm



# ----- Loading datasets, models, genome, and setting global vars -----
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

print("Loading pattern_matrix from cache")
with open("/scratch/rprest2/Enhancer-Creation/output/modisco_results/all_patterns.pkl", "rb") as f:
        all_patterns = pickle.load(f)
pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad")
ft_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")

output_dir = "/scratch/rprest2/Enhancer-Creation/output/R1_MPRA_Generation"
os.makedirs(output_dir,exist_ok=True)

acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    adata_specific,  # accepts any sequence input, same as before
    per_position=True,  # return a distirbution per position in the sequence
)

# ----- Loading pattern pattern_ids and pattern_dicts -----
matched_files = crested.tl.modisco.match_h5_files_to_classes(
    contribution_dir="/scratch/rprest2/Enhancer-Creation/output/modisco_results",
    classes=list(adata_specific.obs_names),
)
print("Matched files:", matched_files)

sim_matrix, pattern_ids, pattern_dict = crested.tl.modisco.calculate_tomtom_similarity_per_pattern(
    matched_files=matched_files, trim_ic_threshold=0.025, verbose=True
)

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

#Define float arrays for use in cosine similarity optimizer function
cos_hi_array = np.array([1.0 if "_Hi" in s else 0.0 for s in adata_specific.obs_names])
cos_lo_array = np.array([1.0 if "_Lo" in s else 0.0 for s in adata_specific.obs_names])

# ----- Generating enhancers: Dictionary setup for configs -----
optimizer_dict = {
        "MWD": mutli_class_weighted_differences,
        "Adjusted_MWD": intra_line_variance_MWD,
        "Cos_Similarity": cosine_similarity_optimizer,
}

shared_kwargs = dict(
        model= ft_model,
        acgt_distribution= acgt_distribution,
        return_intermediate= False,
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

# This maps all dicts to run_configs to avoid messy if-else statements I previously had
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

# ----- Generating enhancers: ISE Core loop -----
#Check cache to see if ISE generated file exists
save_path = os.path.join(output_dir, "ise_sequences_all.tsv")
cache_check = save_path
if not os.path.exists(cache_check):
        #Core loop for generating all ISE needed enhancers
        all_results = []
        for opt_name, state_configs in run_configs.items():
                #Set the optimization function
                optimization_function = crested.tl.design.EnhancerOptimizer(optimize_func=optimizer_dict[opt_name])
                for state_name, config_data in state_configs.items():
                        print(f"Running ISE for {state_name} using {opt_name}")
                        designed_sequences = crested.tl.design.in_silico_evolution(
                                target = config_data["target"],
                                enhancer_optimizer= optimization_function,
                                **config_data["kwargs"],
                        )

                        for i, seq in enumerate(designed_sequences):
                                all_results.append({
                                        "sequence_id": f"{state_name}_{opt_name}_{i+1}",
                                        "sequence": seq,
                                        "cell_state_target": state_name,
                                        "optimizer_used": opt_name,
                                        "sequence_type": "synthetic"
                        })
                print(f"--> Collected {len(designed_sequences)} sequences ({state_name}, {opt_name})")

        # ----- Single save after all runs complete -----
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(save_path, sep="\t", index=False)
        print(f"\nAll runs complete. Saved {len(df_all)} total sequences to {save_path}")
        print(df_all.groupby(["cell_state_target", "optimizer_used"]).size())  # sanity check

        print("All runs complete and safely banked!")
else:
        df_all = pd.read_csv(cache_check, sep="\t")
        df_all["core_sequence"] = df_all["sequence"].str.slice(957, 957 + 200)
        print("Sample core sequence length:", len(df_all["core_sequence"].iloc[0]))
        print(f"{cache_check} already exists. Skipping ISE & loading all sequences in a dataframe")


# ----- Predictions & Contribution Scores -----

# TO DO: I need to look into how to cache check these
predictions = crested.tl.predict(
        input=df_all["sequence"].tolist(),
        model=ft_model,
)
contr_dir = f"{output_dir}/Contribution_Scores"
os.makedirs(contr_dir, exist_ok=True)


#Need to create a cache check after first run. Can't right now because I don't know how data output looks
print("Running Contribution Scores for first time")
contrib_scores, one_hot_seqs = crested.tl.contribution_scores(
        input=df_all["sequence"].tolist(),   # list of 1200 sequences                   
        model=ft_model,
        method="integrated_grad",
        batch_size=256,                       
        transpose=False,                      
        output_dir=contr_dir,
)
# contrib_scores shape: (1200, n_classes, 4, 2114)
# one_hot_seqs shape:   (1200, 4, 2114)


# ----- Scan the ISE Enhancers with each Contribution Weight Matrices (CWMs) -----
contrib_1d = contrib_scores.sum(axis=-1)   # (1200, 2114)
contrib_1d_core = contrib_1d[:, 957 : 957 + 200]  # (1200, 200)

# Build the (1200 × n_patterns) matrix
print(f"Scanning {len(df_all)} sequences against {len(pattern_ids)} patterns...")

motif_matrix = np.zeros((len(df_all), len(pattern_ids)), dtype=np.float32)

for j, pid in enumerate(pattern_ids):
    cwm = pattern_dict[pid]["contrib_scores"]   # (pattern_len, 4) — your existing pattern dict
    for i in range(len(df_all)):
        motif_matrix[i, j] = scan_cwm(contrib_1d_core[i], cwm)
    if j % 10 == 0:
        print(f"  Pattern {j+1}/{len(pattern_ids)} done")

motif_df = pd.DataFrame(
    motif_matrix,
    index=df_all["sequence_id"].values,
    columns=pattern_ids,
)
# Shape: (1200, n_patterns) ← exactly what you want
print(motif_df.shape)


# ----- UMAP Embedding -----
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)
embedding = reducer.fit_transform(motif_df.values)  # (1200, 2)

df_umap = pd.DataFrame({
    "UMAP1":       embedding[:, 0],
    "UMAP2":       embedding[:, 1],
    "sequence_id": df_all["sequence_id"].values,
    "cell_state":  df_all["cell_state_target"].values,
    "optimizer":   df_all["optimizer_used"].values,
})


# ----- UMAP generation -----

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"]  = ["Liberation Sans", "Arimo", "DejaVu Sans"]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP A: Download SCENIC+ v10 mouse motif annotation table (once)
# ═══════════════════════════════════════════════════════════════════════════════
scenic_tbl_path = os.path.join(output_dir, "motifs-v10-nr.mgi-m0.001-o0.0.tbl")
if not os.path.exists(scenic_tbl_path):
    print("Downloading SCENIC+ v10 mouse motif annotation table...")
    subprocess.run([
        "wget", "-q",
        "https://resources.aertslab.org/cistarget/motif2tf/motifs-v10-nr.mgi-m0.001-o0.0.tbl",
        "-O", scenic_tbl_path,
    ], check=True)
    print("Downloaded.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP B: Load SCENIC+ annotation and map pattern_ids → TF family
# ═══════════════════════════════════════════════════════════════════════════════
motif_to_tf_df = crested.tl.modisco.read_motif_to_tf_file(scenic_tbl_path)

# Confirm column names — print and adjust below if different
print("SCENIC+ columns:", motif_to_tf_df.columns.tolist())
# Expected: ['#motif_id', 'gene_name', 'motif_similarity_qvalue',
#            'orthologous_identity', 'description', 'gene_family_name']

# Generate HTML paths for all 16 classes (same reports used in endogenous UMAP)
html_paths = crested.tl.modisco.generate_html_paths(
    all_patterns=all_patterns,
    classes=list(adata_specific.obs_names),
    contribution_dir="/scratch/rprest2/Enhancer-Creation/output/modisco_results",
)

# Match each pattern to its best SCENIC+ motif
# p_val_thr=1.0 — no filtering, we use family for labeling not statistics
pattern_match_dict = crested.tl.modisco.find_pattern_matches(
    all_patterns=all_patterns,
    html_paths=html_paths,
    p_val_thr=1.0,
)

# Build pattern → TF family dict via SCENIC+ metadata
pattern_tf_dict, all_tfs = crested.tl.modisco.create_pattern_tf_dict(
    pattern_match_dict=pattern_match_dict,
    motif_to_tf_df=motif_to_tf_df,
    all_patterns=all_patterns,
    cols=["gene_name", "gene_family_name"],
)

# Map each pattern_id → TF family string
def get_pattern_family(pid):
    try:
        idx = list(pattern_ids).index(pid)
        families = pattern_tf_dict.get(idx, {}).get("gene_family_name", [])
        return families[0] if families else "Other"
    except (ValueError, KeyError):
        return "Other"

pattern_families = {pid: get_pattern_family(pid) for pid in pattern_ids}

# Print distribution so you can verify and update palette keys
family_counts = pd.Series(pattern_families.values()).value_counts()
print("\nTF family distribution across patterns:")
print(family_counts.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# STEP C: Aggregate motif_df by TF family → (1200 × n_families) matrix
#
# Each sequence gets a score per TF family = sum of all pattern scores
# belonging to that family. This is the multi-motif-aware feature matrix.
# ═══════════════════════════════════════════════════════════════════════════════
all_families = sorted(set(pattern_families.values()))

family_score_matrix = pd.DataFrame(index=df_all["sequence_id"].values)
for family in all_families:
    family_patterns = [pid for pid in pattern_ids
                       if pattern_families.get(pid) == family]
    family_score_matrix[family] = (
        motif_df[family_patterns].sum(axis=1).values
        if family_patterns else 0.0
    )

print(f"\nFamily score matrix: {family_score_matrix.shape}")
print("Top families by total score:")
print(family_score_matrix.sum().sort_values(ascending=False).head(10).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# STEP D: Re-run UMAP on family score matrix
#
# Using family_score_matrix instead of raw motif_matrix:
#   - removes redundancy from correlated patterns of the same family
#   - makes UMAP geometry directly interpretable (distance = TF family difference)
#   - consistent with endogenous UMAP which also uses family-level annotation
# ═══════════════════════════════════════════════════════════════════════════════
reducer_family = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)
embedding_family = reducer_family.fit_transform(family_score_matrix.values)

df_umap["UMAP1_fam"] = embedding_family[:, 0]
df_umap["UMAP2_fam"] = embedding_family[:, 1]

# Also attach family scores to df_umap for coloring
for fam in family_score_matrix.columns:
    df_umap[f"score_{fam}"] = family_score_matrix[fam].values

# ═══════════════════════════════════════════════════════════════════════════════
# STEP E: TF family color palette
# Keys must match gene_family_name strings from SCENIC+ — verify after Step B
# ═══════════════════════════════════════════════════════════════════════════════
tf_palette = {
    "AP-1":                 "#E41A1C",   # red
    "KLF/SP":               "#377EB8",   # blue
    "Homeodomain":          "#FF7F00",   # orange  (SCENIC+ uses this for HOX/CDX)
    "ETS":                  "#984EA3",   # purple
    "RUNX":                 "#4DAF4A",   # green
    "CTCF":                 "#A65628",   # brown
    "TEAD":                 "#17BECF",   # cyan
    "bZIP":                 "#8DA0CB",   # steel blue
    "C/EBP":                "#FC8D62",   # salmon
    "IRF":                  "#BCBD22",   # olive
    "Nuclear receptor":     "#9467BD",   # purple
    "bHLH":                 "#E377C2",   # pink
    "Zinc finger":          "#7F7F7F",   # grey
    "Forkhead":             "#BCBD22",   # olive
    "NF-kB":                "#A65628",   # brown
    "Other":                "#CCCCCC",   # light grey
}

state_palette = {"met_high": "#D62728", "met_low": "#1F77B4"}
opt_palette   = {
    "MWD":            "#FF7F00",
    "Adjusted_MWD":   "#4DAF4A",
    "Cos_Similarity": "#984EA3",
}

# ═══════════════════════════════════════════════════════════════════════════════
# STEP F: Top 6 families by total score (for the family score panels)
# ═══════════════════════════════════════════════════════════════════════════════
top_families = (
    family_score_matrix.sum()
    .drop("Other", errors="ignore")
    .nlargest(6)
    .index.tolist()
)
print(f"\nTop 6 TF families for score panels: {top_families}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP G: 3-row figure
#   Row 1 (2 panels): cell state | optimizer
#   Row 2 (6 panels): TF family score panels, one per top family
# ═══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="ticks", font="Liberation Sans")

fig = plt.figure(figsize=(26, 16))
gs  = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.32)

ax_state = fig.add_subplot(gs[0, :3])
ax_opt   = fig.add_subplot(gs[0, 3:])
fam_axes = [fig.add_subplot(gs[1, i]) for i in range(6)]

x_col, y_col = "UMAP1_fam", "UMAP2_fam"
pt = dict(s=35, alpha=0.85, edgecolor="none", linewidth=0)

# ── Panel 1: Cell state ───────────────────────────────────────────────────────
for state, grp in df_umap.groupby("cell_state"):
    ax_state.scatter(grp[x_col], grp[y_col],
                     color=state_palette[state], label=state.replace("_", "-"), **pt)

# Direct cluster labels
texts_state = []
for state, grp in df_umap.groupby("cell_state"):
    cx, cy = grp[x_col].median(), grp[y_col].median()
    texts_state.append(ax_state.text(
        cx, cy, state.replace("_", "-"),
        fontsize=10, fontweight="bold", color=state_palette[state], ha="center",
    ))

ax_state.set_title("Cell state", fontsize=13, fontweight="bold")
ax_state.set_xlabel("UMAP 1", fontsize=11)
ax_state.set_ylabel("UMAP 2", fontsize=11)
ax_state.legend(fontsize=9, framealpha=0.9, loc="best")

# ── Panel 2: Optimizer ────────────────────────────────────────────────────────
for opt, grp in df_umap.groupby("optimizer"):
    ax_opt.scatter(grp[x_col], grp[y_col],
                   color=opt_palette.get(opt, "#CCCCCC"), label=opt, **pt)

texts_opt = []
for opt, grp in df_umap.groupby("optimizer"):
    cx, cy = grp[x_col].median(), grp[y_col].median()
    texts_opt.append(ax_opt.text(
        cx, cy, opt, fontsize=9, fontweight="bold",
        color=opt_palette.get(opt, "#333333"), ha="center",
    ))

ax_opt.set_title("Optimizer", fontsize=13, fontweight="bold")
ax_opt.set_xlabel("UMAP 1", fontsize=11)
ax_opt.set_ylabel("UMAP 2", fontsize=11)
ax_opt.legend(fontsize=9, framealpha=0.9, loc="best")

# ── Panels 3–8: TF family score panels ───────────────────────────────────────
for ax, family in zip(fam_axes, top_families):
    scores = df_umap[f"score_{family}"].values
    # Cap colormap at 95th percentile to prevent outliers washing out the scale
    vmax = np.percentile(scores[scores > 0], 95) if (scores > 0).any() else 1.0

    sc = ax.scatter(
        df_umap[x_col], df_umap[y_col],
        c=scores, cmap="Reds",
        vmin=0, vmax=vmax,
        **pt,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("CWM score (sum)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    fam_color = tf_palette.get(family, "#333333")
    ax.set_title(family, fontsize=11, fontweight="bold", color=fam_color)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)

    # Light cell state centroid markers for orientation
    for state, grp in df_umap.groupby("cell_state"):
        cx, cy = grp[x_col].median(), grp[y_col].median()
        ax.text(cx, cy, state.replace("_", "-"),
                fontsize=7, color="grey", ha="center", alpha=0.5)

fig.suptitle(
    "ISE-generated enhancers — TF motif composition UMAP\n"
    "(200 bp core, modisco CWM vocabulary, SCENIC+ v10 annotation)",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.tight_layout()

umap_png = os.path.join(output_dir, "ise_umap_tf_composition.png")
umap_svg = os.path.join(output_dir, "ise_umap_tf_composition.svg")
plt.savefig(umap_png, dpi=200, bbox_inches="tight")
plt.savefig(umap_svg, bbox_inches="tight")
plt.close()
print(f"Saved {umap_png}")
print(f"Saved {umap_svg}")