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

# ----- Loading datasets, models, and genome -----
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad") 

BM_02TS_prmean_2114_nonorm_ep10 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm/checkpoints/10.keras")
BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")
BM_02TS_prmean_2114_nonorm_ep10__FT_Gini1 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_Gini-1_LR1e-4/checkpoints/04.keras")

output_dir = "/scratch/rprest2/Enhancer-Creation/output/modisco_results"

# ----- Generate model predictions on Met specific peaks -----
#Information regarding adata_specific:
#AnnData object with n_obs × n_vars = 16 × 8414
#    obs: 'file_path'
#    var: 'chr', 'start', 'end', 'split', 'da_class', 'fold', 'fdr'

#Make directory for TF output
os.makedirs("/scratch/rprest2/Enhancer-Creation/output/modisco_results", exist_ok= True)


# ── Get all Hi and Lo indices ──────────────────────────────────────────────────
hi_idx = [i for i, n in enumerate(adata_specific.obs_names) if "_Hi" in n]
lo_idx = [i for i, n in enumerate(adata_specific.obs_names) if "_Lo" in n]

# ── Split peaks by class ───────────────────────────────────────────────────────
mh_mask = adata_specific.var["da_class"] == "met_high"
ml_mask = adata_specific.var["da_class"] == "met_low"

adata_mh = adata_specific[:, mh_mask]   # shape: (16, ~2659)
adata_ml = adata_specific[:, ml_mask]   # shape: (16, ~5756)
print(f"met_high peaks: {mh_mask.sum()}")
print(f"met_low  peaks: {ml_mask.sum()}")


## ── Contribution scores: met_high peaks → all Hi classes ──────────────────────
#Cache check
path_check = "/scratch/rprest2/Enhancer-Creation/output/modisco_results/met_high/KPC-1_Hi1_contrib.npz"
if not os.path.exists(path_check):
    print("Contribution Scores not ran. Running now")
    # Store predictions for all our regions in the adata_specific object
    predictions = crested.tl.predict(adata_specific, BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414)
    adata_specific.layers["Finetuned on DA Peaks"] = predictions.T

    # Calculate the average of the ground truth and predictions
    adata_specific.layers['combined'] = (adata_specific.X + adata_specific.layers["Finetuned on DA Peaks"])/2

    crested.tl.contribution_scores(
        input=adata_mh,
        target_idx=hi_idx,                          # all Hi sample indices
        model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
        method="integrated_grad",
        transpose=True,
        output_dir="/scratch/rprest2/Enhancer-Creation/output/modisco_results/met_high_IG",
        all_class_names=list(adata_specific.obs_names),
        batch_size=128,
    )
    print("met_high contribution scores done.")

    # ── Contribution scores: met_low peaks → all Lo classes ───────────────────────
    crested.tl.contribution_scores(
        input=adata_ml,
        target_idx=lo_idx,                          # all Lo sample indices
        model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
        method="integrated_grad",
        transpose=True,
        output_dir="/scratch/rprest2/Enhancer-Creation/output/modisco_results/met_low_IG",
        all_class_names=list(adata_specific.obs_names),
        batch_size=128,
    )
    print("met_low contribution scores done.")
else:
    print("Files already exist.")
    

output_dir = "/scratch/rprest2/Enhancer-Creation/output/modisco_results"
contrib_dir_mh = "/scratch/rprest2/Enhancer-Creation/output/modisco_results/met_high_IG"
contrib_dir_ml = "/scratch/rprest2/Enhancer-Creation/output/modisco_results/met_low_IG"

meme_db, motif_to_tf_file = crested.get_motif_db()
#I can change where this downloads by apparently setting system variable $CRESTED_DATA_DIR. 

#Cache check to see if this has been ran prior
path_check = "/scratch/rprest2/Enhancer-Creation/output/modisco_results/KPC-1_Hi1_report"
if not os.path.exists(output_dir):
    print("crested.tl.modisco.tfmodisco")
    for contrib_dir, class_label in [
        (contrib_dir_mh, "met_high"),
        (contrib_dir_ml, "met_low"),
    ]:
        print(f"Running modisco for {class_label}...")
        crested.tl.modisco.tfmodisco(
            window=1000,
            output_dir=output_dir,
            contrib_dir=contrib_dir,
            report=True,          # set False if TOMTOM unavailable
            meme_db=meme_db,
            max_seqlets=20000,
        )
        print(f"Done: {class_label}")
else:
    print("crested.tl.modisco.tfmodisco has already been run. Proceeding with analysis")

#Converge all modisco results into a single call. Unsure on the output of this. Should update with shape
matched_files = crested.tl.modisco.match_h5_files_to_classes(
    contribution_dir=output_dir,
    classes=list(adata_specific.obs_names),
)
print("Matched files:", matched_files)

#Independent pattern comparison
#This will show how how similar each pattern motif is in each class
sim_matrix, pattern_ids, pattern_dict = crested.tl.modisco.calculate_tomtom_similarity_per_pattern(
    matched_files=matched_files, trim_ic_threshold=0.025, verbose=True
)
hi_class = [i for i in adata_specific.obs_names if "_Hi" in i]
lo_class = [i for i in adata_specific.obs_names if "_Lo" in i]


# ------- Graphs -------

groups, groups_2 = [], []
for pattern_id in pattern_ids:
    # Pattern IDs look like: "KPC-1_Hi1_pos_patterns_0"
    # Strip last 3 tokens to get class name: "KPC-1_Hi1"
    ct = "_".join(pattern_id.split("_")[:-3])

    if ct in hi_class:
        groups.append("met_high")
    elif ct in lo_class:
        groups.append("met_low")
    else:
        raise ValueError(f"Unknown class: {ct} from pattern_id: {pattern_id}")

    groups_2.append(ct)

# Broad group colors (met_high vs met_low)
group_colors = {
    "met_high": "#E05C5C",   # red
    "met_low":  "#5C7BE0",   # blue
}

# Fine-grained per-sample colors
unique_cats = pd.unique(groups_2)
group_colors_2 = {cat: mcolors.to_hex(plt.get_cmap("tab20", len(unique_cats))(i)) for i, cat in enumerate(unique_cats)}


# ── Plot 1: All classes only with high seqlet filtering ────────────────
fig = crested.pl.modisco.clustermap_tomtom_similarities(
    sim_matrix=sim_matrix,
    ids=pattern_ids,
    pattern_dict=pattern_dict,
    group_info=[(groups, group_colors), (groups_2, group_colors_2)],
    min_seqlets=300,      # was 100 — raise until the plot is readable (~50-80 patterns)
    threshold=3,
)
# Suppress tick labels entirely — the color bars carry the information
fig.ax_heatmap.set_xticklabels([])
fig.ax_heatmap.set_yticklabels([])
fig.ax_heatmap.set_xlabel("")
fig.ax_heatmap.set_ylabel("")
plt.savefig(f"{output_dir}/Motif_Similarities_overview.png", dpi=200, bbox_inches="tight")
plt.close()


# ── Plot 2: Met-high classes only — internal consistency check ────────────────
fig = crested.pl.modisco.clustermap_tomtom_similarities(
    sim_matrix=sim_matrix,
    ids=pattern_ids,
    pattern_dict=pattern_dict,
    group_info=[(groups, group_colors), (groups_2, group_colors_2)],
    class_names=hi_class,   # only Hi patterns
    min_seqlets=200,
    threshold=3,
)
fig.ax_heatmap.set_xticklabels([])
fig.ax_heatmap.set_yticklabels([])
plt.savefig(f"{output_dir}/Motif_Similarities_methi_only.png", dpi=200, bbox_inches="tight")
plt.close()

# ── Plot 3: Met-Low classes only — internal consistency check ────────────────
fig = crested.pl.modisco.clustermap_tomtom_similarities(
    sim_matrix=sim_matrix,
    ids=pattern_ids,
    pattern_dict=pattern_dict,
    group_info=[(groups, group_colors), (groups_2, group_colors_2)],
    class_names=lo_class,   # only Hi patterns
    min_seqlets=200,
    threshold=3,
)
fig.ax_heatmap.set_xticklabels([])
fig.ax_heatmap.set_yticklabels([])
plt.savefig(f"{output_dir}/Motif_Similarities_metlo_only.png", dpi=200, bbox_inches="tight")
plt.close()


# ── Plot 4: Pattern Clustering ────────────────

path_check = "/scratch/rprest2/Enhancer-Creation/output/modisco_results/all_patterns.pkl"
if not os.path.exists(path_check):
    print("Patterns have not grouped before. Performing now")
    all_patterns = crested.tl.modisco.process_patterns(
        matched_files,
        sim_threshold=6.5,  # The similarity threshold used for matching patterns. We take the -log10(pval), pval obtained through TOMTOM matching from memesuite-lite
        trim_ic_threshold=0.05,  # Information content (IC) threshold on which to trim patterns
        discard_ic_threshold=0.2,  # IC threshold used for discarding single instance patterns
        verbose=True,  # Useful for doing sanity checks on matching patterns
    )

    pattern_matrix = crested.tl.modisco.create_pattern_matrix(
        classes=list(adata_specific.obs_names),
        all_patterns=all_patterns,
        normalize=False,
        pattern_parameter="seqlet_count_log"
    )
    # pattern_matrix shape: (16, n_patterns)
    # Each row = one Hi or Lo class; each column = one motif cluster
    pattern_matrix.shape

    # Save pattern clusters 
    with open("/scratch/rprest2/Enhancer-Creation/output/modisco_results/all_patterns.pkl", 'wb') as f:
        pickle.dump(all_patterns, f)
else:
    print("Loading pattern_matrix from cache")
    with open("/scratch/rprest2/Enhancer-Creation/output/modisco_results/all_patterns.pkl", "rb") as f:
        all_patterns = pickle.load(f)
    
    pattern_matrix = crested.tl.modisco.create_pattern_matrix(
        classes=list(adata_specific.obs_names),
        all_patterns=all_patterns,
        normalize=False,
        pattern_parameter="seqlet_count_log"
    )
    # pattern_matrix shape: (16, n_patterns)
    # Each row = one Hi or Lo class; each column = one motif cluster
    pattern_matrix.shape

pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)

crested.pl.modisco.clustermap(
    pattern_matrix=pattern_matrix,
    classes=list(adata_specific.obs_names),   # all 16 classes (rows)
    grid=True,
    cmap="coolwarm",
    center=0,                                  # white = absent, red = positive, blue = negative
    pat_seqs=pat_seqs,                         # sequence logos on x-axis
    dendrogram_ratio=(0.05, 0.3),              # taller row dendrogram to separate Hi/Lo
    importance_threshold=3,                    # drop patterns with log(seqlet_count) < 3 in all classes
    width=25,
    height=4,                                  # short height — you only have 16 rows
    method="average",
)
plt.savefig(f"{output_dir}/pattern_clustermap.png", dpi=200, bbox_inches="tight")
plt.close()

# Hi classes only
crested.pl.modisco.clustermap(
    pattern_matrix=pattern_matrix,
    classes=list(adata_specific.obs_names),
    subset=list(hi_class),                     # only met-high rows
    grid=True,
    cmap="coolwarm",
    center=0,
    pat_seqs=pat_seqs,
    importance_threshold=3,
    width=25,
    height=2.5,
    dendrogram_ratio=(0.05, 0.3),
)
plt.savefig(f"{output_dir}/pattern_clustermap_methi.png", dpi=200, bbox_inches="tight")
plt.close()

# Lo classes only
crested.pl.modisco.clustermap(
    pattern_matrix=pattern_matrix,
    classes=list(adata_specific.obs_names),
    subset=list(lo_class),                     # only met-low rows
    grid=True,
    cmap="coolwarm",
    center=0,
    pat_seqs=pat_seqs,
    importance_threshold=3,
    width=25,
    height=2.5,
    dendrogram_ratio=(0.05, 0.3),
)
plt.savefig(f"{output_dir}/pattern_clustermap_metlo.png", dpi=200, bbox_inches="tight")
plt.close()


# ── Plot 5: UMAP TF Motif Clustering ────────────────
# sim_matrix is -log10(p-val): higher = more similar
# UMAP needs distances: lower = more similar
# Cap at 15 to avoid inf, then invert
sim_capped = np.clip(sim_matrix, 0, 15)
dist_matrix = 15 - sim_capped                    # now: 0 = identical, 15 = unrelated
np.fill_diagonal(dist_matrix, 0)                 # self-distance = 0

reducer = umap.UMAP(
    metric="precomputed",     # use our distance matrix directly
    n_neighbors=15,           # increase if clusters are too fragmented
    min_dist=0.3,             # increase for more spread
    n_components=2,
    random_state=42,
)
embedding = reducer.fit_transform(dist_matrix)

# ── Metacluster → TF family (for opaque CisBP/JASPAR IDs) ────────────────────
# Built from named entries in the same HTML reports.
METACLUSTER_FAMILY = {
    1:   "HOX/CDX",          # HXC9, CDX2, HOXA10/11/13, HOXC10/11/13, HOXD11/13
    3:   "KLF/SP",           # SP1/SP3
    8:   "RUNX",             # RUNX, RUNX-AML, RUNX3, PEBB
    11:  "Nuclear receptor", # ftz-f1 (NR5A), NR4A2, RORA
    14:  "bHLH",             # tgo/ARNT bHLH-PAS
    22:  "AP-1/bZIP",        # MAFK
    34:  "bHLH",             # GFY/THA11
    35:  "bZIP/ATF-CREB",    # ATF/CREB
    52:  "Forkhead/FOX",     # Foxk1
    57:  "bHLH",             # bHLH (CisBP M07824)
    65:  "ZNF/KRAB",         # ZNF513
    88:  "ZNF/KRAB",         # ZNF513
    101: "CTCF",             # CTCF_ENCODE
    111: "bZIP/ATF-CREB",    # ATF6B, ATF1
    116: "HOX/CDX",          # CDX4, HOXC11
    121: "Nuclear receptor", # RORA
    130: "bHLH",
    137: "AP-1/bZIP",        # MAFK, FOSL1, FOSL2
    148: "Nuclear receptor",
    159: "CTCF",             # co-clusters with metacluster_101
    163: "KLF/SP",           # KLF5, KLF4
    165: "KLF/SP",
    166: "ETS",              # ELK4, FLI1, ETV3
    170: "KLF/SP",           # SP4
    171: "TEAD",             # TEAD4, TEAD1
    174: "IRF",              # IRF3
    182: "NF1/CTF",          # NF1-halfsite
    195: "KLF/SP",           # SP4
}


def _parse_tf_family(match_string):
    m = match_string.lower()

    # Metacluster lookup for opaque CisBP/JASPAR IDs
    mc_match = re.match(r"metacluster_(\d+)\.", m)
    if mc_match:
        mc_num = int(mc_match.group(1))
        after_dunder = m.split("__", 1)[1] if "__" in m else m
        is_opaque = (
            re.match(r"m\d{5}", after_dunder) or   # cisbp M-codes
            re.match(r"ma\d{4}", after_dunder) or   # JASPAR MA-codes
            after_dunder.startswith("md0")           # tfdimers MD-codes
        )
        if is_opaque and mc_num in METACLUSTER_FAMILY:
            return METACLUSTER_FAMILY[mc_num]

    if any(x in m for x in ["fosl1","fosl2","fosb","fos_","fosl","jun_","junb","jund","jdp2","batf","mafk","mafb","mafa","maff","mafg","ma0835","ma1988","ma1141","ma0099","ma0100","fos::","jun::"]):
        return "AP-1/bZIP"
    if any(x in m for x in ["atf1_","atf2_","atf3_","atf4_","atf6","atf7","creb1","creb3","__creb","__atf","xbp1","nfe2l","nrf2"]):
        return "bZIP/ATF-CREB"
    if any(x in m for x in ["cebpa","cebpb","cebpd","cebpg","cebpz","__cebp"]):
        return "C/EBP"
    if any(x in m for x in ["klf1","klf2","klf3","klf4","klf5","klf6","klf7","klf8","klf9","klf10","klf11","klf12","klf13","klf14","klf15","klf16","sp1_","sp2_","sp3_","sp4_","sp1.","sp3.","sp4.","__sp1","__sp2","__sp3","__sp4"]):
        return "KLF/SP"
    if any(x in m for x in ["ctcf","ctcfl","boris"]):
        return "CTCF"
    if any(x in m for x in ["elk1","elk3","elk4","elf1","elf2","elf3","elf4","elf5","etv1","etv2","etv3","etv4","etv5","etv6","etv7","ets1","ets2","erg_","fev_","fli1","fli2","gabpa","gabpb","spi1","spib","spic","spdef","pu.1","erf_","elf_",]):
        if "runx" not in m: return "ETS"
    if any(x in m for x in ["runx1","runx2","runx3","__runx","_runx","runx-aml","pebb"]):
        return "RUNX"
    if any(x in m for x in ["hoxa","hoxb","hoxc","hoxd","hxa","hxb","hxc","hxd","cdx1","cdx2","cdx4","__cdx","evx1","evx2","__evx","barx1","barx2","__barx","pbx1","pbx2","pbx3","pbx4","__pbx","meis1","meis2","meis3"]):
        return "HOX/CDX"
    if any(x in m for x in ["nfkb","rela_","relb_","nfkb1","nfkb2","__rela","__relb"]):
        return "NF-κB"
    if any(x in m for x in ["rora","rorb","rorc","rxra","rxrb","rxrg","rara","rarb","rarg","nr4a1","nr4a2","nr4a3","ftz-f1","ftzf1","esrra","esrrb","esrrg","ppara","ppard","pparg"]):
        return "Nuclear receptor"
    if any(x in m for x in ["tead1","tead2","tead3","tead4","__tead"]):
        return "TEAD"
    if any(x in m for x in ["hnf1a","hnf1b","__hnf1","hnf4a","hnf4g","__hnf4"]):
        return "HNF1/4"
    if any(x in m for x in ["foxa1","foxa2","foxa3","foxo1","foxo3","foxo4","foxk1","foxk2","foxm1","foxp1","foxp2","foxp3"]):
        return "Forkhead/FOX"
    if any(x in m for x in ["irf1","irf2","irf3","irf4","irf5","irf7","irf8","irf9","__irf"]):
        return "IRF"
    if any(x in m for x in ["nfya","nfyb","nfyc","__nfy"]):
        return "NF-Y"
    if any(x in m for x in ["tbx1","tbx2","tbx3","tbx4","tbx5","tbx6","tbx18","tbx20","tbx21","tbox","__tbx","tha11","tbet"]):
        return "T-box"
    if any(x in m for x in ["gata1","gata2","gata3","gata4","gata5","gata6","__gata"]):
        return "GATA"
    if any(x in m for x in ["sox2","sox4","sox9","sox10","sox17","__sox"]):
        return "SOX"
    if any(x in m for x in ["myc_","mycn","max_","mlx_","mnt_","usf1","usf2","tfeb","tfec","mitf","tgo_","arnt","ahr_","neurod","atoh","ascl"]):
        return "bHLH"
    if any(x in m for x in ["znf","kznf","zbtb"]):
        return "ZNF/KRAB"
    if any(x in m for x in ["nf1-","nf1_","__nf1","nfia","nfib","nfic","nfix"]):
        return "NF1/CTF"
    if any(x in m for x in ["nfat","nfatc"]):
        return "NFAT"
    return "Other"


def get_top_tf_family(pattern_id, contribution_dir, p_val_threshold=0.05):
    parts      = pattern_id.split("_")
    class_name = "_".join(parts[:-3])
    pos_neg    = parts[-3]
    pat_num    = parts[-1]

    html_path = os.path.join(contribution_dir, f"{class_name}_report", "motifs.html")
    if not os.path.exists(html_path):
        return "No report"
    try:
        df_html = pd.read_html(html_path)[0]
    except Exception:
        return "No report"

    row = df_html[df_html["pattern"] == f"{pos_neg}_patterns.pattern_{pat_num}"]
    if row.empty:
        return "Other"

    # Pass 1: scan match0→match2 with p-value filter
    for i in range(3):
        match_col = f"match{i}"
        if match_col not in row.columns:
            continue
        family = _parse_tf_family(str(row[match_col].values[0]))
        if family != "Other":
            return family

    return "Other"

# ── Build TF family labels ─────────────────────────────────────────────────────
print("Building TF family labels from HTML reports...")
tf_labels = [get_top_tf_family(pid, output_dir) for pid in pattern_ids]
print(f"TF families found: {pd.Series(tf_labels).value_counts().to_dict()}")

# ── Seqlet counts ──────────────────────────────────────────────────────────────
seqlet_counts = np.array([
    int(pattern_dict[pid]["n_seqlets"][0])
    if hasattr(pattern_dict[pid]["n_seqlets"], "__len__")
    else int(pattern_dict[pid]["n_seqlets"])
    for pid in pattern_ids
])

# ── Build unified DataFrame ────────────────────────────────────────────────────
df = pd.DataFrame({
    "UMAP1":        embedding[:, 0],
    "UMAP2":        embedding[:, 1],
    "Cell State":   groups,                          # ← fix: was state_labels built from pid string
    "Pattern Type": ["Positive" if "pos_patterns" in pid else "Negative" for pid in pattern_ids],
    "TF Family":    tf_labels,
    "Seqlet Count": seqlet_counts,
})

print(df["Cell State"].value_counts())   # sanity check — should show both met_high and met_low

# ── TF family color palette ────────────────────────────────────────────────────
tf_palette = {
    # ── Core families (well-resolved) ─────────────────────────────────────────
    "AP-1/bZIP":        "#E41A1C",   # red 
    "KLF/SP":           "#377EB8",   # blue
    "CTCF":             "#4DAF4A",   # green
    "ETS":              "#984EA3",   # purple
    "RUNX":             "#FF7F00",   # orange  
    "HOX/CDX":          "#E78AC3",   # pink 
    "IRF":              "#A6D854",   # yellow-green
    "TEAD":             "#66C2A5",   # teal (Hippo pathway)
    "bZIP/ATF-CREB":    "#8DA0CB",   # steel blue 
    "C/EBP":            "#FC8D62",   # salmon 
    "NF-κB":            "#A65628",   # brown
    "Nuclear receptor": "#F781BF",   # light pink
    "T-box":            "#DDDDAA",   # tan — kept but rare
    "NF-Y":             "#B3B3FF",   # lavender — kept but rare
    "bHLH":             "#17BECF",   # cyan — new (tgo/ARNT patterns)
    "ZNF/KRAB":         "#7F7F7F",   # grey — non-specific, de-emphasized
    "Forkhead/FOX":     "#BCBD22",   # olive — new (Foxk1)
    "NF1/CTF":          "#9467BD",   # medium purple 
    "HNF1/4":             "#AEC7E8", # light blue 
    "GATA":             "#FFBB78",   # light orange 
    "SOX":              "#98DF8A",   # light green 
    "NFAT":             "#C5B0D5",   # light purple 
    # ── unresolved ───────────────────────────────────────────────────────
    "Other":            "#CCCCCC",   # light grey
    "No report":        "#EEEEEE",   # near-white
}

state_palette = {"met_high": "#D62728", "met_low": "#1F77B4"}

# ── Side-by-side plot ──────────────────────────────────────────────────────────
sns.set_theme(style="ticks", font="Liberation Sans")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

shared_kwargs = dict(
    data=df,
    x="UMAP1",
    y="UMAP2",
    style="Pattern Type",
    markers={"Positive": "o", "Negative": "^"},
    #size="Seqlet Count",
    #sizes=(40, 200),
    s=70,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.4,
)

# Left panel — cell state
sns.scatterplot(
    ax=axes[0],
    hue="Cell State",
    palette=state_palette,
    **shared_kwargs,
)
axes[0].set_title("Colored by cell state", fontsize=14, fontweight="bold")
axes[0].set_xlabel("UMAP 1", fontsize=12)
axes[0].set_ylabel("UMAP 2", fontsize=12)
axes[0].legend(
    bbox_to_anchor=(0, -0.18), loc="upper left",
    ncol=3, fontsize=9, framealpha=0.9,
    title=None,
)

# Right panel — TF family
# Only include families actually present in the data
present_families = [f for f in tf_palette if f in df["TF Family"].values]
sns.scatterplot(
    ax=axes[1],
    hue="TF Family",
    palette={k: tf_palette[k] for k in present_families},
    hue_order=present_families,
    **shared_kwargs,
)
axes[1].set_title("Colored by TF family", fontsize=14, fontweight="bold")
axes[1].set_xlabel("UMAP 1", fontsize=12)
axes[1].set_ylabel("UMAP 2", fontsize=12)
axes[1].legend(
    bbox_to_anchor=(1.02, 1), loc="upper left",
    fontsize=9, framealpha=0.9,
    title="TF Family", title_fontsize=10,
)


fig.suptitle("TF motif patterns — sequence similarity UMAP", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{output_dir}/motif_umap_sidebyside.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved motif_umap_sidebyside.png/.png")

