#!/usr/bin/env python
"""
06_DNA_Diffusion_Evaluation.py
Evaluate DNA-Diffusion generated sequences using the fine-tuned CREsted model.
Scores all 2000 sequences (1000 met_high + 1000 met_low), computes strength and
log-specificity per sequence, and generates a Pareto front plot comparable to
the Chebyshev ISE results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import seaborn as sns
import crested
import keras
import os
import anndata as ad

# ── Paths ──────────────────────────────────────────────────────────────────────
MH_TXT   = "/scratch/rprest2/Enhancer-Creation/DNA-Diffusion/data/outputs/met_high.txt"   # DNA-Diffusion output
ML_TXT   = "/scratch/rprest2/Enhancer-Creation/DNA-Diffusion/data/outputs/met_low.txt"
MODEL_PATH = "/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras"
OUT_DIR  = "/scratch/rprest2/Enhancer-Creation/output/dnadiff_evaluation"
os.makedirs(OUT_DIR, exist_ok=True)
adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad") 

#Load genome
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

# ── Sample metadata — must match adata_specific.obs_names order ───────────────
# These are the 16 sample names in the order the model was trained on.
hi_samples = [s for s in adata_specific.obs_names if "_Hi" in s]
lo_samples = [s for s in adata_specific.obs_names if "_Lo" in s]
HI_IDX = np.array([s in hi_samples for s in adata_specific.obs_names])
LO_IDX = np.array([s in lo_samples for s in adata_specific.obs_names])

# ── 1. Load sequences ──────────────────────────────────────────────────────────
def load_sequences(path):
    with open(path) as f:
        seqs = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(seqs)} sequences from {path}")
    assert all(len(s) == 200 for s in seqs), "Not all sequences are 200 bp!"
    assert all(set(s) <= set("ACGT") for s in seqs), "Non-ACGT characters found!"
    return seqs


# ── 2. Pad sequences to 2114 bp with symmetric N-flanking ─────────────────────
def pad_to_2114(seq, target_len=2114):
    seq_len = len(seq)
    if seq_len == target_len:
        return seq
    elif seq_len > target_len:
        # Trim: take center (shouldn't happen for 200 bp input)
        start = (seq_len - target_len) // 2
        return seq[start:start + target_len]
    else:
        # Pad: add N's symmetrically
        total_pad = target_len - seq_len   # 1914
        left_pad  = total_pad // 2         # 957
        right_pad = total_pad - left_pad   # 957
        return "N" * left_pad + seq + "N" * right_pad


print("Loading sequences...")
mh_seqs = load_sequences(MH_TXT)
ml_seqs = load_sequences(ML_TXT)

print("Padding sequences to 2114 bp...")
mh_seqs_padded = [pad_to_2114(s) for s in mh_seqs]
ml_seqs_padded = [pad_to_2114(s) for s in ml_seqs]

# Verify padding
assert all(len(s) == 2114 for s in mh_seqs_padded), "met_high padding failed!"
assert all(len(s) == 2114 for s in ml_seqs_padded), "met_low padding failed!"
assert all(s[957:1157] == orig for s, orig in zip(mh_seqs_padded, mh_seqs)), \
    "met_high core sequence corrupted by padding!"
assert all(s[957:1157] == orig for s, orig in zip(ml_seqs_padded, ml_seqs)), \
    "met_low core sequence corrupted by padding!"

print(f"  met_high: {len(mh_seqs_padded)} sequences padded  "
      f"(example: N×957 + {mh_seqs_padded[0][957:967]}... + N×957)")
print(f"  met_low:  {len(ml_seqs_padded)} sequences padded")

all_seqs = mh_seqs_padded + ml_seqs_padded
labels   = ["met_high"] * len(mh_seqs_padded) + ["met_low"] * len(ml_seqs_padded)
print(f"  Total: {len(all_seqs)} sequences at {len(all_seqs[0])} bp")

# ── 2. Load model and predict ──────────────────────────────────────────────────
print("\nLoading CREsted model...")
model = crested.utils.load_model(MODEL_PATH)

print("Running predictions on all 2000 sequences...")
# crested.tl.predict() accepts a plain list of strings directly [13]
preds = crested.tl.predict(
    input=all_seqs,
    model=model,
    batch_size=256,
)
# preds shape: (2000, n_classes) — one score per sample per sequence
print(f"  Predictions shape: {preds.shape}")

# ── 3. Compute per-sequence metrics ───────────────────────────────────────────
print("\nComputing strength and log-specificity...")

# Strength = mean predicted accessibility across target-class samples
mh_strength = preds[:, HI_IDX].mean(axis=1)
ml_strength = preds[:, LO_IDX].mean(axis=1)

# Log-specificity = log(target mean / non-target mean)
# For met_high sequences: target=Hi samples, non-target=Lo samples
# For met_low sequences:  target=Lo samples, non-target=Hi samples
eps = 1e-8  # avoid log(0)
mh_logspec = np.log((preds[:, HI_IDX].mean(axis=1) + eps) /
                    (preds[:, LO_IDX].mean(axis=1) + eps))
ml_logspec = np.log((preds[:, LO_IDX].mean(axis=1) + eps) /
                    (preds[:, HI_IDX].mean(axis=1) + eps))

# Assign the right metric to each sequence based on its class
strength   = np.where(np.array(labels) == "met_high", mh_strength, ml_strength)
log_spec   = np.where(np.array(labels) == "met_high", mh_logspec,  ml_logspec)

# ── 4. Build results dataframe ─────────────────────────────────────────────────
df = pd.DataFrame({
    "sequence":      all_seqs,
    "TAG":           labels,
    "strength":      strength,
    "log_specificity": log_spec,
})

# Add all per-sample predictions as columns for downstream use
for i, name in enumerate(adata_specific.obs_names):
    df[f"pred_{name}"] = preds[:, i]

df.to_csv(os.path.join(OUT_DIR, "dnadiff_predictions.csv"), index=False)
print(f"  Saved predictions to {OUT_DIR}/dnadiff_predictions.csv")

# ── 5. Pareto front ────────────────────────────────────────────────────────────
def is_pareto(strength, log_spec):
    """Return boolean mask of Pareto-optimal sequences (maximise both axes)."""
    n = len(strength)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if pareto[i]:
            # dominated if another point is >= on both axes and > on at least one
            dominated = (
                (strength >= strength[i]) &
                (log_spec  >= log_spec[i]) &
                ((strength > strength[i]) | (log_spec > log_spec[i]))
            )
            dominated[i] = False
            pareto[dominated] = False
    return pareto

df_mh = df[df["TAG"] == "met_high"].copy().reset_index(drop=True)
df_ml = df[df["TAG"] == "met_low"].copy().reset_index(drop=True)

df_mh["pareto"] = is_pareto(df_mh["strength"].values, df_mh["log_specificity"].values)
df_ml["pareto"] = is_pareto(df_ml["strength"].values, df_ml["log_specificity"].values)

print("Pareto-optimal sequences:")
print(f"  met_high: {df_mh['pareto'].sum()}")
print(f"  met_low:  {df_ml['pareto'].sum()}")

# ── 6. Visualisation ───────────────────────────────────────────────────────────
sns.set_theme(style="ticks", font="Liberation Sans")
COLORS = {"met_high": "#E05A4E", "met_low": "#4E7EC8"}

# --- 6a. Pareto scatter (one panel per class, side by side) -------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (df_cls, cls) in zip(axes, [(df_mh, "met_high"), (df_ml, "met_low")]):
    non_pareto = df_cls[~df_cls["pareto"]]
    pareto     = df_cls[df_cls["pareto"]]

    ax.scatter(non_pareto["strength"], non_pareto["log_specificity"],
               c=COLORS[cls], alpha=0.35, s=18, linewidths=0, label="Non-Pareto")
    ax.scatter(pareto["strength"], pareto["log_specificity"],
               c=COLORS[cls], alpha=0.95, s=55, edgecolors="black", linewidths=0.8,
               label=f"Pareto (n={len(pareto)})", zorder=5)

    ax.set_xlabel("Strength (mean target-class prediction)", fontsize=12)
    ax.set_ylabel("Log-specificity (log target / non-target)", fontsize=12)
    ax.set_title(f"DNA-Diffusion — {cls.replace('_', '-')}", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pareto_scatter.svg"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "pareto_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pareto_scatter.svg/.png")

# --- 6b. Heatmap: all 2000 sequences × 16 samples ----------------------------
# Sort sequences: met_high first (by strength desc), then met_low (by strength desc)
df_sorted = pd.concat([
    df_mh.sort_values("strength", ascending=False),
    df_ml.sort_values("strength", ascending=False),
]).reset_index(drop=True)

pred_cols = [f"pred_{n}" for n in adata_specific.obs_names]
heatmap_data = df_sorted[pred_cols].values  # (2000, 16)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=np.percentile(heatmap_data, 99))

# Dividing line between met_high and met_low blocks
ax.axhline(len(df_mh) - 0.5, color="black", lw=1.5)

# --- 6b. Heatmap axis labels --------------------------------------------------
ax.set_xticks(range(len(adata_specific.obs_names)))
ax.set_xticklabels(adata_specific.obs_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks([len(df_mh) // 2, len(df_mh) + len(df_ml) // 2])
ax.set_yticklabels(["met_high\n(1000 seqs)", "met_low\n(1000 seqs)"], fontsize=10)
ax.set_xlabel("KPC subclone sample", fontsize=11)
ax.set_title("CREsted predictions — DNA-Diffusion generated sequences", fontsize=12)

plt.colorbar(im, ax=ax, label="Predicted accessibility", shrink=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_all2000.svg"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "heatmap_all2000.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved heatmap_all2000.svg/.png")

# --- 6c. Violin: strength and log-specificity distributions by class ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, metric, ylabel in zip(
    axes,
    ["strength", "log_specificity"],
    ["Strength (mean target-class prediction)", "Log-specificity"],
):
    sns.violinplot(
        data=df, x="TAG", y=metric, palette=COLORS,
        order=["met_high", "met_low"], inner="box", ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(metric.replace("_", " ").title(), fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["met-high", "met-low"], fontsize=11)
    if metric == "log_specificity":
        ax.axhline(0, color="grey", lw=0.8, ls="--")
    sns.despine(ax=ax)

plt.suptitle("DNA-Diffusion sequence quality — CREsted scoring", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "violin_distributions.svg"), bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "violin_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved violin_distributions.svg/.png")

# ── 7. Summary stats ───────────────────────────────────────────────────────────
print("\n=== Summary ===")
for cls, df_cls in [("met_high", df_mh), ("met_low", df_ml)]:
    print(f"\n{cls}:")
    print(f"  Strength:        mean={df_cls['strength'].mean():.4f}  "
          f"median={df_cls['strength'].median():.4f}  "
          f"max={df_cls['strength'].max():.4f}")
    print(f"  Log-specificity: mean={df_cls['log_specificity'].mean():.4f}  "
          f"median={df_cls['log_specificity'].median():.4f}  "
          f"max={df_cls['log_specificity'].max():.4f}")
    print(f"  Pareto-optimal:  {df_cls['pareto'].sum()} sequences")
    print(f"  Log-spec > 0:    {(df_cls['log_specificity'] > 0).sum()} sequences "
          f"({100*(df_cls['log_specificity'] > 0).mean():.1f}%)")

print("\nDone. All outputs saved to:", OUT_DIR)