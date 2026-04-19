from scipy.stats import pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt


# SCRIPT NOT COMPLETE I NEED TO COMPLETE / FIX THIS CODE. Originally copied from Crested_training.py but removed because I don't want to focus on this tonight.


# ── Post-training evaluation ──────────────────────────────────────────────────

print("Evaluating model on held-out test chromosomes (chr9, chr18)...")
test_metrics = trainer.test(return_metrics=True)
print("Test metrics:", test_metrics)

# Per-sample correlation on test set
print("Computing per-sample predictions on test set...")
crested.tl.predict(adata, model=trainer.model)  # stores predictions in adata.layers["predictions"]

# Per-sample Pearson correlation


test_mask = adata.var["split"] == "test"
y_true = adata[:, test_mask].X          # shape: (n_samples, n_test_peaks)
y_pred = adata[:, test_mask].layers["predictions"]
print("\nPer-sample test correlations:")
print(f"{'Sample':<30} {'Pearson r':>10} {'Spearman rho':>14}")
print("-" * 56)
for i, sample in enumerate(adata.obs_names):
    r, _  = pearsonr(y_true[i, :], y_pred[i, :])
    rho, _ = spearmanr(y_true[i, :], y_pred[i, :])
    print(f"{sample:<30} {r:>10.4f} {rho:>14.4f}")

# Save per-sample correlations to CSV
corr_rows = []
for i, sample in enumerate(adata.obs_names):
    r, _   = pearsonr(y_true[i, :], y_pred[i, :])
    rho, _ = spearmanr(y_true[i, :], y_pred[i, :])
    corr_rows.append({"sample": sample, "pearson_r": r, "spearman_rho": rho})

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv("output/test_correlations_base_model.csv", index=False)
print("\nSaved per-sample correlations to output/test_correlations_base_model.csv")

# Loss curve plot
print("Plotting training loss curves...")
crested.pl.training.loss(adata)

plt.savefig("output/training_loss_curve.svg")
plt.close()
print("Saved loss curve to output/training_loss_curve.svg")

print("All post-training evaluation complete.")