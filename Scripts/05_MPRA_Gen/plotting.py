import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_log2fc_scatter(true_log2fc, predictions_dict, da_class, title_suffix, fname_suffix):
    n_models = len(predictions_dict)
    
    # Automatically scale figure size
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    # Pre-format the class names for the legend (e.g., "met_high" -> "met-high")
    clean_da_class = np.array([cls.replace("_", "-") for cls in da_class])
    
    # Map the new clean names to your specific hex colors
    sns_color_map = {"met-high": "#D62728", "met-low": "#1F77B4"}

    for ax, (model_label, pred_log2fc) in zip(axes, predictions_dict.items()):
        
        # 1. Package the data into a Pandas DataFrame for Seaborn
        df = pd.DataFrame({
            "DESeq2 Log2FC": true_log2fc,
            "Predicted Log2FC": pred_log2fc,
            "Class": clean_da_class
        })

        # 2. Seaborn Scatterplot (Replaces the nested color/masking loops)
        sns.scatterplot(
            data=df,
            x="DESeq2 Log2FC",
            y="Predicted Log2FC",
            hue="Class",
            palette=sns_color_map,
            s=15,             # slightly larger size standard for seaborn
            alpha=0.4,
            linewidth=0,      # removes borders around dots
            ax=ax,
            rasterized=True,
            legend=(ax == axes[-1]) # ONLY put the legend on the far-right plot
        )

        # 3. Calculate dynamic axis limits
        lim = max(np.abs(true_log2fc).max(), np.abs(pred_log2fc).max()) * 1.05
        lim = max(lim, 0.5)

        # 4. Add Matplotlib structural lines (Seaborn axes are just Matplotlib axes)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.4, linestyle=":")

        # 5. Add Correlation Text
        r, _ = pearsonr(true_log2fc, pred_log2fc)
        ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(true_log2fc):,} peaks",
                transform=ax.transAxes, fontsize=9, va="top")

        # 6. Formatting
        ax.set_title(model_label, fontsize=10)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        
        # Keep axis labels clean
        ax.set_xlabel("DESeq2 Log2FC (Lo − Hi)", fontsize=10)
        if ax != axes[0]: 
            ax.set_ylabel("") # Remove Y-label for inner plots

    # Clean up the legend on the final plot
    if n_models > 0:
        axes[-1].legend(fontsize=8, markerscale=1.5, loc="lower right", title=None)

    # Save
    fig.suptitle(f"Predicted vs. DESeq2 Log2FC — {title_suffix}", fontsize=11)
    plt.tight_layout()
    
    return fig