import anndata as ad
import crested
import numpy as np
import matplotlib.pyplot as plt
import keras


#Initialize mm10 genome. 
genome = crested.Genome(
        fasta="/scratch/rprest2/indices/mm10_encode.fa",
        chrom_sizes="/scratch/rprest2/indices/mm10_no_alt.chrom.sizes.tsv")
crested.register_genome(genome)

bigwig_folder = "input/CREsted_BigWig"
bigwig_paths = sorted(glob.glob(os.path.join(bigwig_folder, "*.bw")))

#Import the bigwig files for the cell types
#Need to move the desired bigwig files to the input/ directory
# Initially don't specifyc target_region_width to let it use default consensus. I can modify this later with crested.pp.change_regions_width
adata = crested.import_bigwig(
    bigwigs_folder = bigwig_paths,
    regions_file = "input/consensus_peaks_all.bed",
    target = "mean")

print(f"Successfully loaded {adata.n_vars} peaks across {adata.n_obs} samples.")

#Change region width to 2114bp to reflect optimal training results. Further investigation of this function is needed to identify gain from use
crested.pp.change_regions_width(adata, width=2114)

#Normalize peaks. top_k_percent=0.03 is what the CREsted tutorial uses, however I should look into this further
crested.pp.normalize_peaks(adata, top_k_percent=0.03)

#Need to save these graphs to output/ 
crested.pl.qc.normalization_weights(adata, title="Normalization weights per cell type", xtick_rotation=90)
crested.pl.corr.heatmap_self(adata, log_transform=True, vmin=0, vmax=1, reorder=True)

# Save the final preprocessing results
adata.write_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_training_set.h5ad")