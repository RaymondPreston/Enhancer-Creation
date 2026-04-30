import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anndata as ad
import crested
import keras
from scipy.stats import pearsonr, spearmanr

#Load the genome
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

adata = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_training_set.h5ad")
adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad") 

BM_01TS_prmean_2114_nonorm_ep10 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm/checkpoints/10.keras")
BM_01TS_prmean_2114_nonorm_ep10__FT_DA8414 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")
BM_01TS_prmean_2114__nonorm_ep10__FT_Gini1 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_Gini-1_LR1e-4/checkpoints/04.keras")

# ── Post-training evaluation ──────────────────────────────────────────────────

print("Evaluating model on held-out test chromosomes (chr9, chr18)...")
#Base model and FT model predictions on test set of adata_specific (cell type specific peaks)
#At some point, I should also compare base model and FT models on test set of adata (all peaks)

predictions_base = crested.tl.predict(adata_specific, BM_01TS_prmean_2114_nonorm_ep10)
adata_specific.layers["Base model"] = predictions_base.T  # adata expects (classes, genes) instead of (genes, classes)
predictions_ft_DA = crested.tl.predict(adata_specific, BM_01TS_prmean_2114__nonorm_ep10__FT_Gini1)
adata_specific.layers["Finetune on Gini=1"] = predictions_ft_DA.T
predictions_ft_Gi = crested.tl.predict(adata_specific, BM_01TS_prmean_2114_nonorm_ep10__FT_DA8414)
adata_specific.layers["Finetune on DA peaks"] = predictions_ft_Gi.T

#Evaluate each model on the test set of adata_specific
crested.tl.evaluate(
    adata_specific,
    model='Base model',
    metrics=crested.tl.default_configs('peak_regression_mean')
)

crested.tl.evaluate(
    adata_specific,
    model='Finetune on Gini=1',
    metrics=crested.tl.default_configs('peak_regression_mean')
)

crested.tl.evaluate(
    adata_specific,
    model='Finetune on DA peaks',
    metrics=crested.tl.default_configs('peak_regression_mean')
)

crested.pl.corr.violin(adata_specific)
plt.savefig("output/CREsted_Evaluation/02_Correlation_Violin_Plots.png")
plt.close()

crested.pl.corr.heatmap(
    adata_specific,
    split="test",
    log_transform=True,
    vmax=1,
    vmin=0,
)
plt.savefig("output/CREsted_Evaluation/02_Model_CellTypeHeatMap.png")
plt.close()


#Test chromsomes are 9 and 18. Pull two genes from there
#This is for the MetLow gene Dsg1c. 
crested.pl.region.bar(
    data = adata_specific,
    region= "chr18:20180596-20182710",
    pred_color= "lightblue",
    truth_color= "blue"
)
plt.savefig("output/CREsted_Evaluation/02_ML_Dsg1c_RegionBar_Plot.png")
plt.close()

#For Met-low enriched gene Rgs13
crested.pl.region.bar(
    data = adata_specific,
    region= "chr1:144127097-144129211",
    pred_color= "lightblue",
    truth_color= "blue"
)
plt.savefig("output/CREsted_Evaluation/02_ML_Rgs13_RegionBar_Plot.png")
plt.close()

#For Met-high enriched gene Afap1l1
crested.pl.region.bar(
    data = adata_specific,
    region= "chr18:61836815-61838929",
    pred_color= "lightblue",
    truth_color= "blue"
)
plt.savefig("output/CREsted_Evaluation/02_MH_Afap1l1_RegionBar_Plot.png")
plt.close()

#For Met-high enriched gene Glb1l2 this is in a promoter region, so the model should do really well here. 
crested.pl.region.bar(
    data = adata_specific,
    region= "chr9:26805458-26807572",
    pred_color= "lightblue",
    truth_color= "blue"
)
plt.savefig("output/CREsted_Evaluation/02_MH_Glb1l2_RegionBar_Plot.png")
plt.close()

#Returns:
#KeysView(Layers with keys: Base model, Finetune on Gini=1, Finetune on DA peaks)
#AnnData object with n_obs × n_vars = 16 × 8414
#    obs: 'file_path'
#    var: 'chr', 'start', 'end', 'split', 'da_class', 'fold', 'fdr'
#    layers: 'Base model', 'Finetune on Gini=1', 'Finetune on DA peaks'

#Additionally, I should answer the question: How do the models perform on predicting Met-High peaks in met-high samples and met-low peaks in met-low samples
#To do this I need to:
#vars.fold is already done (ground truth)
#I just need the predicted fold change for X model and to plot grouth truth FC vs predicted FC for each model, and color them by met-high or met-low
#Fold is measured as mean(met-low) - mean(met-high)
#So I'll use -fold to flip the value so this looks better visually.
#Predicted vs. Actual Log2FC Scatter — all three models 

hi_idx = np.array(["_Hi" in obs for obs in adata_specific.obs_names])
lo_idx = ~hi_idx

da_color_map = {"met_high": "#D62728", "met_low": "#1F77B4"}

true_log2fc = adata_specific.var["fold"].values      # shape: (8414,)
da_class    = adata_specific.var["da_class"].values  # shape: (8414,)
test_mask   = adata_specific.var["split"] == "test"

def compute_predicted_log2fc(layer):
    pred = adata_specific.layers[layer]              # (n_obs, n_vars) = (16, 8414)
    pred_hi_mean = pred[hi_idx, :].mean(axis=0)
    pred_lo_mean = pred[lo_idx, :].mean(axis=0)
    return np.log2(pred_lo_mean) - np.log2(pred_hi_mean)  # Lo - Hi, matches DESeq2

def plot_log2fc_scatter_3panel(true_log2fc, pred_base, pred_gini, pred_da,
                               da_class, title_suffix, fname_suffix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    models = [
        ("Base model",        pred_base),
        ("Finetune on Gini=1", pred_gini),
        ("Finetune on DA peaks", pred_da),
    ]

    for ax, (model_label, pred_log2fc) in zip(axes, models):
        for da_cls, color in da_color_map.items():
            mask = da_class == da_cls
            ax.scatter(
                true_log2fc[mask], pred_log2fc[mask],
                c=color, s=4, alpha=0.4, linewidths=0,
                label=da_cls.replace("_", "-"),
                rasterized=True,
            )

        lim = max(np.abs(true_log2fc).max(), np.abs(pred_log2fc).max()) * 1.05
        lim = max(lim, 0.5)

        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.4, linestyle=":")

        r, _ = pearsonr(true_log2fc, pred_log2fc)
        ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(true_log2fc):,} peaks",
                transform=ax.transAxes, fontsize=9, va="top")

        ax.set_xlabel("DESeq2 Log2FC (Lo − Hi)", fontsize=10)
        ax.set_ylabel("Predicted Log2FC (Lo − Hi)", fontsize=10)
        ax.set_title(model_label, fontsize=10)
        ax.legend(fontsize=7, markerscale=2, loc="lower right")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

    fig.suptitle(f"Predicted vs. DESeq2 Log2FC — {title_suffix}", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join("output/CREsted_Evaluation", f"log2fc_scatter_{fname_suffix}.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[04] Saved: {out_path}")


pred_base  = compute_predicted_log2fc("Base model")
pred_gini  = compute_predicted_log2fc("Finetune on Gini=1")
pred_da    = compute_predicted_log2fc("Finetune on DA peaks")

# Test set only
plot_log2fc_scatter_3panel(
    true_log2fc[test_mask],
    pred_base[test_mask],
    pred_gini[test_mask],
    pred_da[test_mask],
    da_class[test_mask],
    title_suffix=f"test set only — chr9, chr18 ({test_mask.sum():,} peaks)",
    fname_suffix="3panel_test"
)

# All DA peaks
plot_log2fc_scatter_3panel(
    true_log2fc,
    pred_base,
    pred_gini,
    pred_da,
    da_class,
    title_suffix=f"all DA peaks ({test_mask.sum():,} peaks)",  # fix: use adata_specific.n_vars
    fname_suffix="3panel_all"
)

#Predict LSD synthethic enhancer activites as a control:

#Line 1 is metlow enhancer Line 2 is met-high enhancer
#I initally just copied the enhancers in and the code failed due to the length of sequences. I need to zero-pad the sequences to 2114bp to match model input.
LSD_enhancers = ["CCAGCGACATTGTGCACCACTGCACATTCATGGCGGGGGTCAGTCGAATGCACCATTTAAATCATTAGATCACCTCCGTCCGCTTCTCCTAATTAGAGTCCTTACAATACATTTTTATCTGGTAATTAGCTGAGATTGGCTGCTTCCTCTGCGCCCCCCGCGAGCTCTGTTGTTTACCCAGCAGGTGGATGTGACGTCAGGGACGCACAACAGCAAAAAATAACAACATCTCCCTCCGCGACGCGATGGCCTCTCTTGCGCGCCTTATTTATTTATTTGTTCCGGGATGCGGGGGGAAGGGGTCACCCTAATTTTTAATTATTGGTTTTCAGTTTTTAATTCTACAATCTTTTATGGTATAAATATATGTACAGAAAAGTAAAAAAAAAAAACCACAATAAATCAATACCATTACATTACTATTACTATTGTTGTATATTTCACACAAACACCCGGACCGCACAAAGCCCTGAGCCCCCGCCATCGCGCGCGGCTCGCTCCCCCGTGGGCGCCAAGTCCCGCCCCACGCGTCCCGCCCCACACCATGACGTCACCGCGGCGGCAGGCCCCGCCCCATCCAGGACCCGGCGGCATGACGTCACCGTCCTAGCTTAGCATGTTATCTTCAGTAGGCGCCAGAACATTCAAATGTTTATTTTATTGCACAAAAGGATCAAGGACCCAGAGGCAGAGAAAATTGACATATAGGAAGAATGGCAACAATAAATCATTCCACGGTGCGCTGAGAAGTGGAAAAAATATGCTTTTTTTTTTTTTTTAAAAACTGAAAACTGTATTCTATCATACAAAAAACAAGACTTTATTTTGGGAAATCAATTTCTAATTACAGCAAGTTTTACTTTCAGAAATAAATCCACAGAACAGCACATTCTAGTCCACCCTTACGGGCCGCTGGGGGTGTCAAGGGCTGCTGCCTCGGACAAAGCGGCGGCACCACCCCAAAGCGCGGACCAATGGAATGAATGGGCTATAAATAGCCGCCAATGGGAGGCCGGCGACGCGCCCCTTAAGAGCTCAGGGAGCAGCGAGCAGCCGT",
                "TCCCGAACACTTCCTGCGTACAGGTTGTTCAGGCCCATGAGAGTGAGGTGACAGGCCTCAGCCTCCACGCTACTGGAGACTATCTCCTGAGAGCCTCTGATGATCAGTACTGGGCCTTCTCTGACGTGACAGATGAGACCTCCGGCTGCTCGCCTTCAAATCAGGTGGCAGTGTGTTGCTCAATGAGATGGAGGTGACCCTACCCCATGTGGCAGGTGAGTGGCTCAGAGAAGGATCCTGAACTGCTGCCAGTCAACAGAGCCAGCTGAGATTCGGACAGCCTAAGAGTCTGGGCAAGGAGAAAAAGTATTTGTATATGTATACATATATGTCTGTCGGGGGGGGGGTATGTGTGTCAGAGGTCAGGTGCCTTCAGAGGCCAGAGGCAGGGGTATTAGATGCGCCTGGGGTGAGTTACAGGTAATTATGAGCTACCCAATGAGGTTAATAAAGGTGGGGAGGGGAGGAGAGTGACAGGTGAAAGAGGAAGAGATGCCTCATCAATGATCTTATTGATTCAGATGTGCAGAGGAAGAAAGCCAAGGAGCCTAGAAGGGGCTGAGAGAGAGAGATGGGGAAATCCCCAAGTTACAGGGGTCAAACACCGCAAACCTCTGATTCACACATTCACACCTGCTAGCGGTGGCGCGAGCGACACCTGCTAGGCAGGACGAAGAGCGAAGGGTGAACGGGCATCCAGGAAGGGTGAAGATTCTCGGAGCTGCTCCCAGGAGCGGAGCAGGAAACAAAATCTC"]

def pad_or_trim_to_length(seq, target_len=2114):
    seq_len = len(seq)
    if seq_len == target_len:
        return seq
    elif seq_len > target_len:
        # Trim: take center
        start = (seq_len - target_len) // 2
        return seq[start:start + target_len]
    else:
        # Pad: add N's symmetrically on both sides
        total_pad = target_len - seq_len
        left_pad  = total_pad // 2
        right_pad = total_pad - left_pad
        return "N" * left_pad + seq + "N" * right_pad

LSD_enhancers_padded = [pad_or_trim_to_length(s, 2114) for s in LSD_enhancers]

# Verify
for i, s in enumerate(LSD_enhancers_padded):
    print(f"Sequence {i+1}: {len(s)} bp")


LSD_predictions = crested.tl.predict(LSD_enhancers_padded, model=BM_01TS_prmean_2114_nonorm_ep10__FT_DA8414)
# LSD_predictions shape: (2, 16) — one row per sequence

seq_labels = ["MetLow_LSD", "MetHigh_LSD"]

for i, (pred, label) in enumerate(zip(LSD_predictions, seq_labels)):
    crested.pl.region.bar(pred, classes=list(adata_specific.obs_names))
    plt.suptitle(label, fontsize=12)
    plt.savefig(f"output/CREsted_Evaluation/LSD_Enhancer_{label}.png", bbox_inches="tight")
    plt.close()
    print(f"[04] Saved: LSD_Enhancer_{label}.png")

#The last thing to validate here is to ensure the model is learning TF motif patterns in met-high and met-low using tfmodisco.
