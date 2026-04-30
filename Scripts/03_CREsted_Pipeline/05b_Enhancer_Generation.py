import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anndata as ad
import crested
import keras
from scipy.stats import pearsonr, spearmanr

# ----- Loading datasets, models, and genome -----
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad")

BM_02TS_prmean_2114_nonorm_ep10 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm/checkpoints/10.keras")
BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414_LR1e-4/checkpoints/02.keras")
BM_02TS_prmean_2114_nonorm_ep10__FT_Gini1 = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm_ep10__FT_Gini-1_LR1e-4/checkpoints/04.keras")

acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    adata_specific,
    per_position=True,
)
acgt_distribution.shape

# =============================================================================
# ----- Optimizer function 4: Weighted Chebyshev Scalarization -----
# =============================================================================
def chebyshev_optimizer(
        mutated_predictions: np.ndarray,    # (n_mutations, n_classes)
        original_predictions: np.ndarray,   # (n_classes,) — unused but required by API
        target: np.ndarray,                 # boolean or float (n_classes,) — 1.0 = target class
        lambda_strength: float = 0.5,       # weight on the strength objective [0, 1]
        lambda_specificity: float = 0.5,    # weight on the specificity objective [0, 1]
                                            # lambda_strength + lambda_specificity should sum to 1
) -> int:
        """
        Weighted Chebyshev scalarization optimizer for ISE.

        Balances two competing objectives at each mutation step:
          - Strength:     mean predicted accessibility across target-class samples
          - Specificity:  log ratio of target-class mean to background-class mean

        The Chebyshev score for each mutation is:
            min(lambda_strength * strength, lambda_specificity * log_specificity)

        Taking the minimum (rather than a weighted sum) prevents either objective
        from being ignored when one is already very high — the optimizer is forced
        to improve the weaker objective. This property guarantees that varying
        lambda_strength across ISE runs traces the Pareto front between strength
        and specificity.

        Args:
            lambda_strength:    Weight on strength. Higher values bias the optimizer
                                toward absolute accessibility magnitude.
            lambda_specificity: Weight on specificity. Higher values bias the optimizer
                                toward Hi/Lo fold-enrichment.
        """
        target_mask = target.astype(bool)
        bg_mask = ~target_mask

        # Objective 1: strength — mean accessibility across target samples
        strength = mutated_predictions[:, target_mask].mean(axis=1)  # (n_mutations,)

        # Objective 2: log specificity — log(mean_target / mean_background)
        bg_mean = mutated_predictions[:, bg_mask].mean(axis=1) + 1e-8  # (n_mutations,)
        log_specificity = np.log(strength + 1e-8) - np.log(bg_mean)   # (n_mutations,)

        # Chebyshev scalarization: score = min(lambda_s * strength, lambda_sp * specificity)
        cheby_score = np.minimum(
                lambda_strength * strength,
                lambda_specificity * log_specificity,
        )  # (n_mutations,)

        return int(np.argmax(cheby_score))


# =============================================================================
# ----- Define Hi/Lo sample masks -----
# =============================================================================
hi_samples = [s for s in adata_specific.obs_names if "_Hi" in s]
lo_samples = [s for s in adata_specific.obs_names if "_Lo" in s]
hi_idx = np.array([s in hi_samples for s in adata_specific.obs_names])
lo_idx = np.array([s in lo_samples for s in adata_specific.obs_names])

print(f"Hi samples ({hi_idx.sum()}): {list(adata_specific.obs_names[hi_idx])}")
print(f"Lo samples ({lo_idx.sum()}): {list(adata_specific.obs_names[lo_idx])}")

# Parental line masks
kpc1_hi_idx = np.array([s.startswith("KPC-1") and "_Hi" in s for s in adata_specific.obs_names])
kpc1_lo_idx = np.array([s.startswith("KPC-1") and "_Lo" in s for s in adata_specific.obs_names])
kpc2_hi_idx = np.array([s.startswith("KPC-2") and "_Hi" in s for s in adata_specific.obs_names])
kpc2_lo_idx = np.array([s.startswith("KPC-2") and "_Lo" in s for s in adata_specific.obs_names])

# Float target arrays (used by cosine and Chebyshev optimizers)
cos_hi_array = np.array([1.0 if "_Hi" in s else 0.0 for s in adata_specific.obs_names])
cos_lo_array = np.array([1.0 if "_Lo" in s else 0.0 for s in adata_specific.obs_names])


# =============================================================================
# ----- Chebyshev ISE run: sweep lambda to trace the Pareto front -----
# =============================================================================
N_SEQUENCES = 5
lambdas = np.linspace(0.1, 0.9, N_SEQUENCES)

os.makedirs("output/CREsted_ISE/Cheby", exist_ok=True)
cheby_optimizer = crested.tl.design.EnhancerOptimizer(optimize_func=chebyshev_optimizer)

# --- Met-High Chebyshev ISE ---
print("Running Chebyshev ISE for met-high sequences...")
cheby_mh_sequences = []
cheby_mh_intermediates = []

for lam in lambdas:
        intermediate, sequences = crested.tl.design.in_silico_evolution(
                model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
                enhancer_optimizer=cheby_optimizer,
                target=hi_idx.astype(float),
                return_intermediate=True,
                acgt_distribution=acgt_distribution,
                n_sequences=1,
                target_len=200,
                n_mutations=20,
                lambda_strength=float(lam),
                lambda_specificity=float(1.0 - lam),
        )
        cheby_mh_sequences.append(sequences[0])
        cheby_mh_intermediates.append(intermediate)
        print(f"  lambda_strength={lam:.2f} done")

# --- Met-Low Chebyshev ISE ---
print("Running Chebyshev ISE for met-low sequences...")
cheby_ml_sequences = []
cheby_ml_intermediates = []

for lam in lambdas:
        intermediate, sequences = crested.tl.design.in_silico_evolution(
                model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414,
                enhancer_optimizer=cheby_optimizer,
                target=lo_idx.astype(float),
                return_intermediate=True,
                acgt_distribution=acgt_distribution,
                n_sequences=1,
                target_len=200,
                n_mutations=20,
                lambda_strength=float(lam),
                lambda_specificity=float(1.0 - lam),
        )
        cheby_ml_sequences.append(sequences[0])
        cheby_ml_intermediates.append(intermediate)
        print(f"  lambda_strength={lam:.2f} done")


# =============================================================================
# ----- Score all Chebyshev sequences and identify the empirical Pareto front -----
# =============================================================================
def score_sequence(seq, model, target_mask):
        """Return (strength, log_specificity) for a single designed sequence.
        
        crested.tl.predict returns shape (1, n_classes) — squeeze to (n_classes,)
        before applying the boolean mask.
        """
        pred = crested.tl.predict(seq, model=model).squeeze()  # (n_classes,)
        strength = float(pred[target_mask].mean())
        bg_mean = float(pred[~target_mask].mean()) + 1e-8
        log_spec = float(np.log(strength + 1e-8) - np.log(bg_mean))
        return strength, log_spec


def find_pareto_front(strengths, specificities):
        """
        Returns a boolean mask of Pareto-optimal sequences.
        A sequence is Pareto-optimal if no other sequence is >= on both objectives
        and strictly > on at least one.
        """
        scores = np.column_stack([strengths, specificities])
        n = scores.shape[0]
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
                for j in range(n):
                        if i == j:
                                continue
                        if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                                is_dominated[i] = True
                                break
        return ~is_dominated


# Score met-high sequences
mh_strengths, mh_specificities = [], []
for seq in cheby_mh_sequences:
        s, sp = score_sequence(seq, BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414, hi_idx)
        mh_strengths.append(s)
        mh_specificities.append(sp)

# Score met-low sequences
ml_strengths, ml_specificities = [], []
for seq in cheby_ml_sequences:
        s, sp = score_sequence(seq, BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414, lo_idx)
        ml_strengths.append(s)
        ml_specificities.append(sp)

mh_pareto = find_pareto_front(mh_strengths, mh_specificities)
ml_pareto = find_pareto_front(ml_strengths, ml_specificities)

print(f"\nMet-High: {mh_pareto.sum()} / {N_SEQUENCES} sequences on Pareto front")
print(f"Met-Low:  {ml_pareto.sum()} / {N_SEQUENCES} sequences on Pareto front")


# =============================================================================
# ----- Plots: bar plots, stepwise trajectories, and Pareto scatter -----
# =============================================================================
print("Creating plots for Chebyshev-generated enhancers...")

# Bar plots — met-high
for i, seq in enumerate(cheby_mh_sequences):
        fig, ax = plt.subplots(figsize=(20, 5), layout='constrained')
        # squeeze (1, n_classes) → (n_classes,) for crested.pl.region.bar
        prediction = crested.tl.predict(seq, model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414).squeeze()
        crested.pl.region.bar(
                prediction,
                classes=list(adata_specific.obs_names),
                title=f"MH Chebyshev enhancer {i+1} (lambda_s={lambdas[i]:.2f}, Pareto={'Yes' if mh_pareto[i] else 'No'})",
                ax=ax,
                show=False,
        )
        plt.savefig(f"output/CREsted_ISE/Cheby/MH_Designed_enhancer_{i+1}.png")
        plt.close()

# Bar plots — met-low
for i, seq in enumerate(cheby_ml_sequences):
        fig, ax = plt.subplots(figsize=(20, 5), layout='constrained')
        prediction = crested.tl.predict(seq, model=BM_02TS_prmean_2114_nonorm_ep10__FT_DA8414).squeeze()
        crested.pl.region.bar(
                prediction,
                classes=list(adata_specific.obs_names),
                title=f"ML Chebyshev enhancer {i+1} (lambda_s={lambdas[i]:.2f}, Pareto={'Yes' if ml_pareto[i] else 'No'})",
                ax=ax,
                show=False,
        )
        plt.savefig(f"output/CREsted_ISE/Cheby/ML_Designed_enhancer_{i+1}.png")
        plt.close()

# Pareto scatter: strength vs. log specificity
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')

for ax, strengths, specificities, pareto, title in [
        (axes[0], mh_strengths, mh_specificities, mh_pareto, "Met-High Pareto Front"),
        (axes[1], ml_strengths, ml_specificities, ml_pareto, "Met-Low Pareto Front"),
]:
        sc = ax.scatter(
                strengths, specificities,
                c=lambdas, cmap="coolwarm", s=100, zorder=3,
                edgecolors=["black" if p else "none" for p in pareto],
                linewidths=1.5,
        )
        for i, (s, sp) in enumerate(zip(strengths, specificities)):
                ax.annotate(
                        f"lambda={lambdas[i]:.1f}",
                        (s, sp),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                )
        plt.colorbar(sc, ax=ax, label="lambda_strength")
        ax.set_xlabel("Strength (mean target accessibility)")
        ax.set_ylabel("Log specificity (log Hi/Lo ratio)")
        ax.set_title(title)

plt.suptitle("Chebyshev ISE: Strength vs. Specificity Trade-off", fontsize=13)
plt.savefig("output/CREsted_ISE/Cheby/Pareto_scatter.png", dpi=150)
plt.close()
print("Saved Pareto scatter to output/CREsted_ISE/Cheby/Pareto_scatter.png")


# =============================================================================
# ----- Save Chebyshev sequences to CSV with scores -----
# =============================================================================
cheby_mh_df = pd.DataFrame({
        'sequence':        cheby_mh_sequences,
        'type':            'met_high',
        'lambda_strength': lambdas,
        'strength':        mh_strengths,
        'log_specificity': mh_specificities,
        'pareto_optimal':  mh_pareto,
})
cheby_ml_df = pd.DataFrame({
        'sequence':        cheby_ml_sequences,
        'type':            'met_low',
        'lambda_strength': lambdas,
        'strength':        ml_strengths,
        'log_specificity': ml_specificities,
        'pareto_optimal':  ml_pareto,
})
pd.concat([cheby_mh_df, cheby_ml_df], ignore_index=True).to_csv(
        "output/CREsted_ISE/Cheby/final_enhancer_sequences.csv", index=False
)
print("Saved Chebyshev ISE sequences to output/CREsted_ISE/Cheby/final_enhancer_sequences.csv")