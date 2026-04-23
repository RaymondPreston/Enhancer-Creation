#Currently set up for models trained on Crested non-normalized data.

import os
import anndata as ad
import crested
import keras
import os
from crested.tl import Crested, default_configs
from crested.tl.data import AnnDataModule
import numpy as np


print("Loading preprocessed dataset...")
# Load the preprocessed anndata
adata = ad.read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/02_training_set.h5ad")

print(f"Data loaded: {adata.n_vars} peaks across {adata.n_obs} samples.")

genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

# Create the Data Module
print("Initializing Data Module...")
datamodule = AnnDataModule(adata, 
                        batch_size = 128,
                        max_stochastic_shift=3,
                        always_reverse_complement=True
)

# Define the Model Architecture
# seq_len is 2114 as per the preprocessing script change_regions_width
print("Building model architecture...")
model = crested.tl.zoo.dilated_cnn(seq_len=2114, num_classes=adata.n_obs)

# Retrieve Default Configuration for Peak Regression
#This step defines the optimizer, loss function, and other metrics to use in training
#This likely requires extensive optimization. I should come back to this and modify in future for better performance.
print("Fetching training configurations...")
configs = crested.tl.default_configs("peak_regression_mean", num_classes=adata.n_obs)
print(configs)



# Initialize the Crested Trainer
print("Initializing CREsted Trainer...")
trainer = Crested(
    data=datamodule,
    model=model,
    config=configs,
    project_name="KPC_Metastasis_Enhancer",
    run_name="BM_02TS_prmean_2114_nonorm",
    seed=8,
    logger="wandb"  #Note that there will still be a degree of randomness in training due to GPU nondeterminism operations
)

# Start Training
print("Starting model training...")
# You can adjust the number of epochs or early stopping parameters as needed
trainer.fit(epochs=60,
            learning_rate_reduce_patience=5,
            save_dir="/scratch/rprest2/Enhancer-Creation/input/training_models/BM_02TS_prmean_2114_nonorm"
)
print("Training complete. Models and logs are saved in the project directory.")