import os
import anndata as ad
import crested
import keras
from crested.tl import Crested, default_configs
from crested.tl.data import AnnDataModule
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print("Loading preprocessed dataset...")
# Load the preprocessed anndata
adata = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_training_set.h5ad")

print(f"Data loaded: {adata.n_vars} peaks across {adata.n_obs} samples.")

genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

base_model = crested.utils.load_model("/scratch/rprest2/Enhancer-Creation/input/training_models/checkpoints/15.keras")


#Here I will use the fine tuning training set based on methigh and metlow DA peaks.
#I have the DA peaks in /scratch/rprest2/Enhancer-Creation/output/annotated_differentially_accessible_peaks_no_H10.RDS
#Something to note is that the DA peaks are a different length than the peaks used in base model training
#This is handled in @/scratch/rprest2/Enhancer-Creation/Scripts/03_CREsted_Pipeline/03C_Generate_Finetune_AnnData.py
#Note, that script is AI generated however I am convinced that the script does what I want it to do with the appropriate QC.

adata_specific = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_finetune_DA_peaks.h5ad")

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
    project_name="KPC_Metastasis_Analysis",
    run_name="DApeaks_DilatedCNN_Finetune_v1",
    logger="wandb",  # or 'wandb', 'tensorboard'
)

trainer.fit(
    epochs=60,
    learning_rate_reduce_patience=5,
    early_stopping_patience=6,
    save_dir="/scratch/rprest2/Enhancer-Creation/input/training_models/TI_01_Finetune_Model_Training_v1"
)