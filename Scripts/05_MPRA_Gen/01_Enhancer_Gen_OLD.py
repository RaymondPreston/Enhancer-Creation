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
# all_patterns is the dict returned by crested.tl.modisco.process_patterns():
#   {"0": {"pattern": {...}, "instances": {...}, "classes": {...}, "ic": float, ...}, "1": ...}
# All CREsted modisco functions (generate_nucleotide_sequences, generate_html_paths,
# find_pattern_matches) expect this exact structure — do NOT flatten or convert it.
print(f"all_patterns loaded: {len(all_patterns)} merged patterns (keys: {list(all_patterns.keys())[:5]}...)")

pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad")
ft_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")

output_dir = "/scratch/rprest2/Enhancer-Creation/output/R1_MPRA_Generation"
os.makedirs(output_dir, exist_ok=True)

acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    adata_specific,  # accepts any sequence input, same as before
    per_position=True,  # return a distribution per position in the sequence
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
        df_all["core_sequence"] = df_all["sequence"].str.slice(957, 957 + 200)
        print("Sample core sequence length:", len(df_all["core_sequence"].iloc[0]))
        print(f"{save_path} already exists. Skipping ISE & loading all sequences into a dataframe")


# ----- Predictions & Contribution Scores (cached) -----

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


# CREsted writes one file per class:
#   class_id_N_contrib.npz  →  array shape (n_seqs, 4, seq_len)
#   class_id_N_oh.npz       →  array shape (n_seqs, 4, seq_len)
#
# We check whether all 16 contrib files already exist (from a previous run).
# If yes, load and stack them → contrib_scores shape: (n_seqs, n_classes, 4, seq_len)
# If no, run contribution_scores() which will write them, then stack.

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


# ----- Verify and reduce contrib_scores to 1D per-position track -----
# Actual CREsted output shape: (n_seqs, n_classes, seq_len, n_bases)
#   axis 0 = sequences (1200)
#   axis 1 = classes   (16)
#   axis 2 = positions (2114)
#   axis 3 = bases     (4: A/C/G/T)
#
# To get a single per-position importance score per sequence:
#   1. Sum over classes (axis=1): (n_seqs, n_classes, seq_len, n_bases) → (n_seqs, seq_len, n_bases)
#   2. Sum over bases   (axis=2): (n_seqs, seq_len, n_bases)            → (n_seqs, seq_len)
#
# Verification: print shape at each step so you can confirm axes are correct.
assert contrib_scores.ndim == 4, (
    f"Expected 4D contrib_scores, got shape {contrib_scores.shape}"
)
n_seqs, n_cls, seq_len, n_bases = contrib_scores.shape
print(f"\ncontrib_scores axes: n_seqs={n_seqs}, n_classes={n_cls}, seq_len={seq_len}, n_bases={n_bases}")
assert n_bases == 4,    f"Expected 4 bases at axis 3, got {n_bases}"
assert seq_len == 2114, f"Expected seq_len=2114 at axis 2, got {seq_len}"

step1 = contrib_scores.sum(axis=1)   # (n_seqs, 2114, 4) — sum over classes
step2 = step1.sum(axis=2)            # (n_seqs, 2114)     — sum over bases
print(f"After sum(axis=1) [classes]: {step1.shape}")
print(f"After sum(axis=2) [bases]:   {step2.shape}")

contrib_1d      = step2                            # (n_seqs, 2114)
contrib_1d_core = contrib_1d[:, 957 : 957 + 200]  # (n_seqs, 200)

motif_cache = os.path.join(contr_dir, "motif_matrix.npy")

if os.path.exists(motif_cache):
    print("Loading motif matrix from cache...")
    motif_matrix = np.load(motif_cache)
    print(f"  motif_matrix shape: {motif_matrix.shape}")
else:
    print(f"Scanning {len(df_all)} sequences against {len(pattern_ids)} patterns...")

    # Pre-filter patterns with empty CWMs to avoid ValueError in correlate()
    valid_pattern_ids = []
    skipped = []
    for pid in pattern_ids:
        cwm = pattern_dict[pid].get("contrib_scores", np.array([]))
        if cwm.ndim == 2 and cwm.shape[0] > 0 and cwm.shape[1] == 4:
            valid_pattern_ids.append(pid)
        else:
            skipped.append(pid)
    if skipped:
        print(f"  Skipping {len(skipped)} patterns with empty/malformed CWMs: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    motif_matrix = np.zeros((len(df_all), len(valid_pattern_ids)), dtype=np.float32)

    for j, pid in enumerate(valid_pattern_ids):
        cwm = pattern_dict[pid]["contrib_scores"]   # (pattern_len, 4)
        for i in range(len(df_all)):
            motif_matrix[i, j] = scan_cwm(contrib_1d_core[i], cwm)
        if j % 10 == 0:
            print(f"  Pattern {j+1}/{len(valid_pattern_ids)} done")

    # Update pattern_ids to only the valid ones (keeps motif_df columns consistent)
    pattern_ids = valid_pattern_ids

    np.save(motif_cache, motif_matrix)
    print(f"Motif matrix saved → {motif_cache}  shape={motif_matrix.shape}")


motif_df = pd.DataFrame(
    motif_matrix,
    index=df_all["sequence_id"].values,
    columns=pattern_ids,
)
# Shape: (1200, n_patterns)
print(f"motif_df shape: {motif_df.shape}")

# ----- TF Family Mapping -----

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"]  = ["Liberation Sans", "Arimo", "DejaVu Sans"]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP A: Build motif → TF family lookup
#
# Two-step chain:
#   1. motif_tf_collection.tsv  : Motif_name → mouse gene symbol
#      (priority: Mouse_Direct_annot > Mouse_Orthology_annot >
#                 Cluster_Mouse_Direct_annot > Cluster_Mouse_Orthology_annot)
#   2. AnimalTFDB Mus_musculus_TF.txt : gene symbol → TF family (DBD-based, 72 families)
# ═══════════════════════════════════════════════════════════════════════════════
# ── Load AnimalTFDB mouse TF → family table ───────────────────────────────────
atfdb_path = "/scratch/rprest2/Enhancer-Creation/input/motif_db/Mus_musculus_TF.txt"
assert os.path.exists(atfdb_path), f"AnimalTFDB file not found: {atfdb_path}"

atfdb = pd.read_csv(atfdb_path, sep="\t")
print(f"AnimalTFDB loaded: {len(atfdb)} mouse TFs across {atfdb['Family'].nunique()} families")

# Build gene symbol → family dict (case-sensitive; AnimalTFDB uses title case e.g. "Fosl2")
gene_to_family = dict(zip(atfdb["Symbol"].str.strip(), atfdb["Family"].str.strip()))

# ── Load motif collection and extract best mouse TF per motif ─────────────────
scenic_tbl_path = "/scratch/rprest2/Enhancer-Creation/input/motif_db/motif_tf_collection.tsv"
assert os.path.exists(scenic_tbl_path), f"TF annotation not found: {scenic_tbl_path}"

motif_collection = pd.read_csv(scenic_tbl_path, sep="\t")
print(f"Motif collection loaded: {motif_collection.shape[0]} motifs")

def get_mouse_tf(row):
    """Return the first mouse TF gene symbol for a motif, in priority order."""
    for col in ["Mouse_Direct_annot", "Mouse_Orthology_annot",
                "Cluster_Mouse_Direct_annot", "Cluster_Mouse_Orthology_annot"]:
        val = row[col]
        if pd.notna(val) and str(val).strip():
            return str(val).split(",")[0].strip()   # take first if comma-separated list
    return None

motif_collection["mouse_tf"] = motif_collection.apply(get_mouse_tf, axis=1)

# ── Map gene symbol → AnimalTFDB family ──────────────────────────────────────
motif_collection["tf_family"] = motif_collection["mouse_tf"].map(gene_to_family).fillna("Other")

# ── Build final lookup: motif_name (lowercase) → tf_family ───────────────────
motif_to_family = dict(zip(
    motif_collection["Motif_name"].str.lower(),
    motif_collection["tf_family"]
))

# ── Coverage diagnostics ──────────────────────────────────────────────────────
n_total      = len(motif_collection)
n_has_tf     = motif_collection["mouse_tf"].notna().sum()
n_has_family = (motif_collection["tf_family"] != "Other").sum()
n_tf_not_in_atfdb = (motif_collection["mouse_tf"].notna() &
                     (motif_collection["tf_family"] == "Other")).sum()

print("\n── Motif annotation coverage ──────────────────────────────")
print(f"  Total motifs in collection : {n_total}")
print(f"  With a mouse TF name       : {n_has_tf}  ({100*n_has_tf/n_total:.1f}%)")
print(f"  With an AnimalTFDB family  : {n_has_family}  ({100*n_has_family/n_total:.1f}%)")
print(f"  TF name found but no family: {n_tf_not_in_atfdb}  "
      f"(gene in motif_coll but absent from AnimalTFDB — likely chromatin remodelers/cofactors)")

print(f"\n── AnimalTFDB family distribution (all {n_total} motifs) ──")
print(motif_collection["tf_family"].value_counts().to_string())

print("── Top 20 TF names missing from AnimalTFDB ────────────────")
missing_tfs = (motif_collection
               .loc[motif_collection["mouse_tf"].notna() &
                    (motif_collection["tf_family"] == "Other"), "mouse_tf"]
               .value_counts()
               .head(20))
print(missing_tfs.to_string())
print("  (These are mostly general factors / chromatin remodelers — 'Other' is correct)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP B: Map pattern_ids → TF family using modisco HTML match strings
#
# Each pattern_id's best motif match comes from the modisco HTML reports.
# We extract the matched motif name and look it up in motif_to_family.
# ═══════════════════════════════════════════════════════════════════════════════
html_paths = crested.tl.modisco.generate_html_paths(
    all_patterns=all_patterns,
    classes=list(adata_specific.obs_names),
    contribution_dir="/scratch/rprest2/Enhancer-Creation/output/modisco_results",
)

# Debug: html_paths is a list (one entry per merged pattern), each entry is a list of
# per-instance HTML paths. Print a sample to confirm the files are being found.
print(f"\ngenerate_html_paths returned {len(html_paths)} entries (one per merged pattern)")
if html_paths and html_paths[0]:
    print(f"  Sample html_paths[0]: {html_paths[0]}")

pattern_match_dict = crested.tl.modisco.find_pattern_matches(
    all_patterns=all_patterns,
    html_paths=html_paths,
    p_val_thr=1.0,   # no p-value filtering — we only need the motif name for family lookup
)

# Debug: pattern_match_dict keys are the all_patterns string keys ("0", "1", ...)
# Each value is {"matches": [...], "patterns": [...]}
print(f"\nfind_pattern_matches returned {len(pattern_match_dict)} entries")
non_empty = {k: v for k, v in pattern_match_dict.items() if v}
print(f"  Non-empty entries: {len(non_empty)}")
if non_empty:
    sample_k = list(non_empty.keys())[0]
    sample_v = non_empty[sample_k]
    print(f"  Sample key     : {sample_k!r}")
    print(f"  Sample value   : {sample_v}")

# ── Build pattern_id → all_patterns key reverse lookup ────────────────────────
# all_patterns keys are "0", "1", ... (string integers from process_patterns).
# Each all_patterns[k]["instances"] maps instance_id → pattern_dict, where
# instance_id is the human-readable pattern_id (e.g. "KPC-1_Hi1_pos_patterns_0").
# pattern_match_dict keys are the same all_patterns string keys.
# We need: pattern_id (from pattern_ids list) → all_patterns key → matches.
pid_to_apkey = {}
for ap_key, ap_val in all_patterns.items():
    for instance_id in ap_val.get("instances", {}):
        pid_to_apkey[instance_id] = ap_key

print(f"\nReverse lookup built: {len(pid_to_apkey)} instance IDs → all_patterns keys")
print(f"  Sample: {list(pid_to_apkey.items())[:3]}")

def get_pattern_family(pid):
    """
    Look up the best-match motif name for a pattern_id via pattern_match_dict,
    then map to TF family via motif_to_family.
    pattern_match_dict[ap_key] = {"matches": [motif_name, ...], "patterns": [...]}
    Returns (family_str, matched_motif_name, matched_tf_name).
    """
    try:
        ap_key = pid_to_apkey.get(pid)
        if ap_key is None:
            return "Other", None, None
        match_entry = pattern_match_dict.get(ap_key, {})
        matches = match_entry.get("matches", []) if isinstance(match_entry, dict) else []
        if not matches:
            return "Other", None, None
        motif_name = matches[0]   # already a plain string (e.g. "jaspar__MA0099.3")
        family = motif_to_family.get(motif_name.lower(), "Other")
        tf_row = motif_collection.loc[
            motif_collection["Motif_name"].str.lower() == motif_name.lower(), "mouse_tf"
        ].values
        tf_name = tf_row[0] if len(tf_row) > 0 else None
        return family, motif_name, tf_name
    except (KeyError, IndexError, TypeError):
        return "Other", None, None

# Build per-pattern results including motif name and TF name for diagnostics
pattern_results = {pid: get_pattern_family(pid) for pid in pattern_ids}
pattern_families  = {pid: r[0] for pid, r in pattern_results.items()}
pattern_motifs    = {pid: r[1] for pid, r in pattern_results.items()}
pattern_tf_names  = {pid: r[2] for pid, r in pattern_results.items()}

# ── Pattern-level diagnostics ─────────────────────────────────────────────────
n_patterns      = len(pattern_ids)
n_matched       = sum(1 for m in pattern_motifs.values() if m)
n_annotated     = sum(1 for f in pattern_families.values() if f != "Other")
n_matched_no_fam = sum(1 for pid in pattern_ids
                       if pattern_motifs[pid] and pattern_families[pid] == "Other")

print(f"\n── Pattern annotation coverage ({n_patterns} total patterns) ──")
print(f"  Patterns with a motif match    : {n_matched}  ({100*n_matched/n_patterns:.1f}%)")
print(f"  Patterns with an ATFDB family  : {n_annotated}  ({100*n_annotated/n_patterns:.1f}%)")
print(f"  Matched but no family (Other)  : {n_matched_no_fam}  "
      f"(motif matched but TF not in AnimalTFDB)")
print(f"  No motif match at all          : {n_patterns - n_matched}  "
      f"(pattern not found in HTML reports)")

print(f"\n── TF family distribution across {n_patterns} patterns ──────")
family_counts = pd.Series(pattern_families.values()).value_counts()
print(family_counts.to_string())

print("── Per-pattern annotation table (first 30) ──────────────────")
pattern_annot_df = pd.DataFrame({
    "pattern_id" : list(pattern_ids),
    "matched_motif": [pattern_motifs[p] for p in pattern_ids],
    "mouse_tf"   : [pattern_tf_names[p] for p in pattern_ids],
    "tf_family"  : [pattern_families[p] for p in pattern_ids],
})
print(pattern_annot_df.head(30).to_string(index=False))

print("── Patterns annotated as 'Other' — top matched TF names ─────")
other_tfs = (pattern_annot_df
             .loc[(pattern_annot_df["tf_family"] == "Other") &
                  pattern_annot_df["mouse_tf"].notna(), "mouse_tf"]
             .value_counts()
             .head(15))
if len(other_tfs):
    print(other_tfs.to_string())
else:
    print("  None — all matched patterns have an AnimalTFDB family.")

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

df_umap = pd.DataFrame({
    "UMAP1_fam":       embedding_family[:, 0],
    "UMAP2_fam":       embedding_family[:, 1],
    "sequence_id": df_all["sequence_id"].values,
    "cell_state":  df_all["cell_state_target"].values,
    "optimizer":   df_all["optimizer_used"].values,
})

# Also attach family scores to df_umap for coloring
for fam in family_score_matrix.columns:
    df_umap[f"score_{fam}"] = family_score_matrix[fam].values

# ═══════════════════════════════════════════════════════════════════════════════
# STEP E: TF family color palette
# Keys must match gene_family_name strings from SCENIC+ — verify after Step B
# ═══════════════════════════════════════════════════════════════════════════════
tf_palette = {
    "AP-1":             "#E41A1C",   # Bright Red
    "KLF/SP":           "#377EB8",   # Deep Blue
    "Homeodomain":      "#FF7F00",   # Orange (Standard for HOX/CDX)
    "ETS":              "#984EA3",   # Purple
    "RUNX":             "#4DAF4A",   # Bright Green
    "CTCF":             "#A65628",   # Brown
    "TEAD":             "#17BECF",   # Cyan
    "bZIP":             "#F781BF",   # Pink
    "C/EBP":            "#F46D43",   # Coral/Salmon
    "IRF":              "#BCBD22",   # Olive Green
    "Nuclear receptor": "#08519C",   # Dark Navy Blue (High contrast)
    "bHLH":             "#006D2C",   # Dark Forest Green
    "Zinc finger":      "#000000",   # Black (Stands out well for generic/broad classes)
    "Forkhead":         "#E6AB02",   # Mustard Gold (Replaces the duplicate olive)
    "NF-kB":            "#810F7C",   # Dark Magenta/Plum (Replaces the duplicate brown)
    "Other":            "#CCCCCC",   # Light Grey (Fades to the background)
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
# STEP G: 2-row figure
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

for state, grp in df_umap.groupby("cell_state"):
    cx, cy = grp[x_col].median(), grp[y_col].median()
    ax_state.text(cx, cy, state.replace("_", "-"),
                  fontsize=10, fontweight="bold", color=state_palette[state], ha="center")

ax_state.set_title("Cell state", fontsize=13, fontweight="bold")
ax_state.set_xlabel("UMAP 1", fontsize=11)
ax_state.set_ylabel("UMAP 2", fontsize=11)
ax_state.legend(fontsize=9, framealpha=0.9, loc="best")

# ── Panel 2: Optimizer ────────────────────────────────────────────────────────
for opt, grp in df_umap.groupby("optimizer"):
    ax_opt.scatter(grp[x_col], grp[y_col],
                   color=opt_palette.get(opt, "#CCCCCC"), label=opt, **pt)

for opt, grp in df_umap.groupby("optimizer"):
    cx, cy = grp[x_col].median(), grp[y_col].median()
    ax_opt.text(cx, cy, opt, fontsize=9, fontweight="bold",
                color=opt_palette.get(opt, "#333333"), ha="center")

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
plt.savefig(umap_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {umap_png}")
print("Enhancer generation complete.")
