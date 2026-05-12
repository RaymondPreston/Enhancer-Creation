import os
import anndata as ad
import numpy as np
import pandas as pd
import crested
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotting import plot_log2fc_scatter

# ----- Paths ------
BASE_ADATA_PATH = "/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_training_set.h5ad"
FT_ADATA_PATH   = "/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad"
BASE_MODEL_PATH = "/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm/checkpoints/10.keras"
SAVE_DIR        = "/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA_Balanced_OS15_LW15"
RUN_NAME        = "BM_02TS_prmean_2114_nonorm_ep10__FT_DA_Balanced_OS15_LW15"
BAL_MODEL       = "/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA_Balanced_OS15_LW15/checkpoints/01.keras"
OUTPUT_DIR      = "/scratch/rprest2/Enhancer-Creation/output/R1_MPRA_Generation/Balanced_Model_Graphs"

# ----- Imbalance correction parameters ------
TARGET_RATIO    = 1.5   # target met-low : met-high ratio after oversampling
JITTER_MAX_BP   = 10    # ±bp range for pre-baked coordinate jitter on duplicates
MET_HIGH_WEIGHT = 1.5   # weight assigned to met-high peaks in the weight vector
RANDOM_SEED     = 42

# ---- Loading models ---- 
print("\n" + "=" * 70)
print("Loading models...")
base_model = crested.utils.load_model(BASE_MODEL_PATH)
ft_gini_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_Gini-1_LR1e-4/checkpoints/04.keras")
ft_da_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")


# ---- Load data ----
print("=" * 70)
print("Loading datasets...")
adata      = ad.read_h5ad(BASE_ADATA_PATH)
adata_ft   = ad.read_h5ad(FT_ADATA_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"  Base AnnData:      {adata.n_vars:,} peaks × {adata.n_obs} samples")
print(f"  Fine-tune AnnData: {adata_ft.n_vars:,} peaks × {adata_ft.n_obs} samples")

# Confirm da_class annotation exists
assert "da_class" in adata_ft.var.columns, \
    "adata_ft.var must have a 'da_class' column (produced by 03C_Generate_Finetune_AnnData.py)"
assert set(adata_ft.var["da_class"].unique()) <= {"met_high", "met_low"}, \
    f"Unexpected da_class values: {adata_ft.var['da_class'].unique()}"

# Confirm coordinate columns exist (needed for jitter)
for col in ("chr", "start", "end"):
    assert col in adata_ft.var.columns, \
        f"adata_ft.var must have a '{col}' column for coordinate-based jitter"

genome = crested.Genome(
    fasta="/scratch/rprest2/indices/mm10_encode.fa",
    chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv",
)
crested.register_genome(genome)

# ---- Jitter & oversample met-high peaks ----
print("\n" + "=" * 70)
print("Stage 1: Jittered oversampling of met-high peaks")
print("-" * 70)

hi_mask   = adata_ft.var["da_class"] == "met_high"
lo_mask   = adata_ft.var["da_class"] == "met_low"
n_hi      = int(hi_mask.sum())
n_lo      = int(lo_mask.sum())
obs_ratio = n_lo / n_hi

print("  Before oversampling:")
print(f"    met_high: {n_hi:,}")
print(f"    met_low:  {n_lo:,}")
print(f"    ratio (lo:hi): {obs_ratio:.2f}  (target: {TARGET_RATIO:.2f})")

# Number of duplicates needed to reach TARGET_RATIO
n_duplicates = max(0, round(n_lo / TARGET_RATIO) - n_hi)
print(f"\n  Duplicates to add: {n_duplicates:,}")
if not os.path.exists(SAVE_DIR):
    if n_duplicates == 0:
        print("  No oversampling needed — ratio already at or below target.")
        adata_balanced = adata_ft.copy()
        adata_balanced.var["is_duplicate"] = False
    else:
        rng = np.random.default_rng(RANDOM_SEED)

        # Integer positions (within adata_ft.var) of met-high peaks
        train_mask = adata_ft.var["split"] == "train"
        hi_var_indices = np.where(hi_mask & train_mask)[0]

        # Sample with replacement from met-high pool
        sampled_indices = rng.choice(hi_var_indices, size=n_duplicates, replace=True)

        # Pre-compute jitter offsets: uniform integers in [-JITTER_MAX_BP, +JITTER_MAX_BP]
        jitter_offsets = rng.integers(
            low=-JITTER_MAX_BP - 5, high=JITTER_MAX_BP + 1, size=n_duplicates
        )

        print("  Jitter offset distribution (bp):")
        print(f"    min={jitter_offsets.min()}, max={jitter_offsets.max()}, "
            f"mean={jitter_offsets.mean():.1f}")

        # Build duplicate AnnData slice (peaks are in var, so index along axis=1)
        adata_hi_dupes = adata_ft[:, sampled_indices].copy()

        # Apply jitter to genomic coordinates.
        # The 2,114 bp window shifts with start/end — AnnDataModule will fetch a
        # slightly different sequence for each duplicate at training time.
        adata_hi_dupes.var["start"] = (
            adata_hi_dupes.var["start"].values + jitter_offsets
        )
        adata_hi_dupes.var["end"] = (
            adata_hi_dupes.var["end"].values + jitter_offsets
        )

        # Rename var_names to avoid index collisions after concat
        adata_hi_dupes.var_names = pd.Index([
            f"{row.chr}:{row.start}-{row.end}"
            for row in adata_hi_dupes.var[["chr", "start", "end"]].itertuples()
        ])

        # Mark as duplicates for traceability
        adata_hi_dupes.var["is_duplicate"] = True

        # Sanity check: print first 5 duplicates
        print("\n  First 5 duplicate var_names and shifted coordinates:")
        for i, row in enumerate(adata_hi_dupes.var.head(5).itertuples()):
            print(f"    {row.Index}  chr={row.chr}  "
                f"start={row.start}  end={row.end}  "
                f"da_class={row.da_class}  jitter={jitter_offsets[i]:+d} bp")

        # Mark originals
        adata_ft_marked = adata_ft.copy()
        adata_ft_marked.var["is_duplicate"] = False

        # Concatenate along var axis (axis=1 = peaks dimension)
        adata_balanced = ad.concat(
            [adata_ft_marked, adata_hi_dupes],
            axis=1,
            merge="same",   # obs (sample) metadata must match exactly
        )

    # Final class breakdown
    n_hi_final  = int((adata_balanced.var["da_class"] == "met_high").sum())
    n_lo_final  = int((adata_balanced.var["da_class"] == "met_low").sum())
    final_ratio = n_lo_final / n_hi_final

    print("\n  After oversampling:")
    print(f"    met_high: {n_hi_final:,}  "
        f"(original: {n_hi:,}, jittered duplicates: {n_hi_final - n_hi:,})")
    print(f"    met_low:  {n_lo_final:,}")
    print(f"    ratio (lo:hi): {final_ratio:.2f}  (target: {TARGET_RATIO:.2f})")
    print(f"    total peaks: {adata_balanced.n_vars:,}")
    print("\n  Full da_class breakdown:")
    print(adata_balanced.var["da_class"].value_counts().to_string())
else:
    print("Model already generated. Skipping generation of adata_balanced")



# ----- Build training config and train on jitter oversampled adata  ----
print("\n" + "=" * 70)
print("Building TaskConfig...")

if not os.path.exists(SAVE_DIR):
    print("\n" + "=" * 70)
    print("Building AnnDataModule...")
    datamodule = crested.tl.data.AnnDataModule(
        adata_balanced,
        batch_size=64,
        max_stochastic_shift=3,        # runtime jitter on top of pre-baked offsets
        always_reverse_complement=True,
    )
    print("  batch_size=64, max_stochastic_shift=3, always_reverse_complement=True")
    base_cfg      = crested.tl.default_configs("peak_regression_mean", num_classes=adata.n_obs)

    base_loss     = base_cfg.loss
    '''
    weighted_loss = MetHighWeightedLoss(
        base_loss=base_loss,
        global_weight_scalar=dataset_mean_weight,
        name="met_high_weighted_loss",
    )
    '''

    config = crested.tl.TaskConfig(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=base_loss,
        metrics=base_cfg.metrics,
    )
    print(f"  Base loss:           {type(base_loss).__name__}")
    print("  Optimizer:           Adam(lr=1e-4)")
    print(f"  Metrics:             {[type(m).__name__ for m in base_cfg.metrics]}")
    print(config)
    # Printing validation trainings
    print("\n" + "=" * 70)
    print("PRE-TRAINING VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  adata_balanced shape:      {adata_balanced.shape}")
    print(f"  met_high peaks:            {n_hi_final:,}  "
        f"(orig {n_hi:,} + {n_hi_final - n_hi:,} jittered dupes)")
    print(f"  met_low  peaks:            {n_lo_final:,}")
    print(f"  lo:hi ratio:               {final_ratio:.2f}  (target {TARGET_RATIO:.2f})")
    print(f"  jitter range:              {JITTER_MAX_BP} bp (pre-baked) + ±3 bp (runtime)")
    print(f"  primary correction:        oversampling — {n_hi_final - n_hi:,} extra "
        f"met-high peaks per epoch (~{final_ratio:.1f}× more hi gradient updates)")
    print(f"  run_name:                  {RUN_NAME}")
    print(f"  save_dir:                  {SAVE_DIR}")
    print("=" * 70)

    # --- Model Training ----
    os.makedirs(SAVE_DIR, exist_ok=True)

    trainer = crested.tl.Crested(
        data=datamodule,
        model=base_model,
        config=config,
        project_name="KPC_Metastasis_Enhancer",
        run_name=RUN_NAME,
        logger="wandb",
    )

    print("\nStarting training...")
    trainer.fit(
        epochs=60,
        learning_rate_reduce_patience=5,
        early_stopping_patience=6,
        save_dir=SAVE_DIR,
    )

    print(f"\nTraining complete. Checkpoints saved to: {SAVE_DIR}")
    ft_bal_model = crested.utils.load_model(BAL_MODEL)  # ← add this
else:
    print("Model already generated. Loading model now")
    ft_bal_model = crested.utils.load_model(BAL_MODEL)

# ---- Run predictions on models ----
models_dict={
    "Base model": base_model,
    "Finetuned on Gini peaks":ft_gini_model,
    "Finetuned on DA peaks":ft_da_model,
    "Finetuned on balanced DA peaks":ft_bal_model,
}

for model_name, model_var in models_dict.items():
    predictions = crested.tl.predict(adata_ft,model_var)
    adata_ft.layers[model_name] = predictions.T

    crested.tl.evaluate(
        adata_ft,
        model = model_var,
        metrics=crested.tl.default_configs('peak_regression_mean')
    )

# ---- Generate plots of models -----
crested.pl.corr.violin(adata_ft)
plt.savefig(f"{OUTPUT_DIR}/Corr_Plot.png")
plt.close()

#Plotting log2fc parameters: plot_log2fc_scatter(true_log2fc, predictions_dict, da_class, title_suffix, fname_suffix)

true_log2fc = adata_ft.var["fold"].values      # shape: (8414,)
da_class    = adata_ft.var["da_class"].values  # shape: (8414,)
test_mask   = adata_ft.var["split"] == "test"
hi_idx = np.array(["_Hi" in obs for obs in adata_ft.obs_names])
lo_idx = ~hi_idx

def compute_predicted_log2fc(adata, layer):
    pred = adata.layers[layer]              
    pred_hi_mean = pred[hi_idx, :].mean(axis=0)
    pred_lo_mean = pred[lo_idx, :].mean(axis=0)
    return np.log2(pred_lo_mean) - np.log2(pred_hi_mean)

all_predictions = {}
for model_name in models_dict.keys():
    all_predictions[model_name] = compute_predicted_log2fc(adata_ft, model_name)

test_predictions = {
    name: preds[test_mask] for name, preds in all_predictions.items()
}

# Plot 1: Test set only
fig_test_peaks = plot_log2fc_scatter(
    true_log2fc[test_mask],
    test_predictions,
    da_class[test_mask],
    title_suffix=f"test set only — chr9, chr18 ({test_mask.sum():,} peaks)",
    fname_suffix="4panel_test"
)
fig_test_peaks.savefig(f"{OUTPUT_DIR}/test_peaks_log2fc.png")
plt.close()

# Plot 2: All DA peaks
fig_DA = plot_log2fc_scatter(
    true_log2fc,
    all_predictions,
    da_class,
    title_suffix=f"all DA peaks ({len(true_log2fc):,} peaks)", 
    fname_suffix="4panel_all"
)
fig_DA.savefig(f"{OUTPUT_DIR}/all_peaks_log2fc.png")
plt.close()


# ----- Building Dataset for concordantly expressed genes-----
