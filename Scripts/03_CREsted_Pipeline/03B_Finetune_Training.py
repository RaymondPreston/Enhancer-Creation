import os
import anndata as ad
import crested
import keras
from crested.tl import Crested, default_configs
from crested.tl.data import AnnDataModule
import numpy as np
import matplotlib.pyplot as plt


print("Loading preprocessed dataset...")
# Load the preprocessed anndata
adata = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_training_set.h5ad")

print(f"Data loaded: {adata.n_vars} peaks across {adata.n_obs} samples.")

genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

base_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/checkpoints/15.keras")

crested.pl.qc.filter_cutoff(adata, cutoffs=[1.5, 1, 0.5], width=8, height=6)
plt.savefig("output/CREsted_PreProcess/Gini_Cutoff.png")
plt.close()

#Based on gini_cutoff plot, we will use a gini cut off of 1 to filter out majority of consensus peaks. All peaks 1 STD away from mean gini are kept
#May need to play around with this. It's unclear to me if we are training on the peaks that correspond to metastatic potential here.
#To validate these, I should use adata_specific and pull out which peaks are being kept to see if they are the same metastatic peaks that I want them to be.
adata_specific = crested.pp.filter_regions_on_specificity(adata, gini_std_threshold=1.0, inplace=False)
adata_specific

adata_specific.write_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_a_ft_training_set.h5ad")

datamodule = crested.tl.data.AnnDataModule(
    adata_specific,
    batch_size=64,  # Recommended to go for a smaller batch size than in the base model
    max_stochastic_shift=3,
    always_reverse_complement=True,
)


old_config = crested.tl.default_configs("peak_regression_mean", num_classes=adata.n_obs)
new_optimizer = keras.optimizers.Adam(learning_rate=1e-4)  # Lower learning rate for fine-tuning (per the CREsted paper)
config = crested.tl.TaskConfig(new_optimizer, old_config.loss, old_config.metrics)
print(config)

# setup the trainer
trainer = crested.tl.Crested(
    data=datamodule,
    model=base_model,
    config=config,
    project_name="KPC_Metastasis_Enhancer",  # change to your liking
    run_name="TI_01_Finetune_Model_Training_v1",  # change to your liking
    logger="wandb",  # or 'wandb', 'tensorboard'
)

trainer.fit(
    epochs=60,
    learning_rate_reduce_patience=5,
    early_stopping_patience=6,
    save_dir="/scratch/rprest2/Enhancer-Creation/input/training_models/TI_01_Finetune_Model_Training_v1"
)