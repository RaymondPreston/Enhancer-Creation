# Scripts/Generate_Consensus_Peaks.R
# This script uses DiffBind to merge individual ATAC-seq peaks into a master consensus BED file.

# Load required library
suppressPackageStartupMessages(library(DiffBind))

# --- Configuration ---
# You will need to generate this CSV based on your croo output paths
SAMPLE_SHEET_PATH <- "/scratch/rprest2/Enhancer-Creation/ATAC_samplesheet.csv"
OUTPUT_BED <- "/scratch/rprest2/Enhancer-Creation/output/consensus_peaks.bed"
QC_PLOT <- "/scratch/rprest2/Enhancer-Creation/output/DiffBind_Initial_Correlation.pdf"

print(paste("Loading sample sheet from:", SAMPLE_SHEET_PATH))
samples <- read.csv(SAMPLE_SHEET_PATH)

# 1. Initialize DiffBind object
print("Initializing DiffBind object (Reading peaks)...")
dba_obj <- dba(sampleSheet=samples)

# Optional: Generate a correlation heatmap of the peak overlaps for QC
print(paste("Saving initial peak correlation plot to:", QC_PLOT))
pdf(QC_PLOT)
plot(dba_obj)
dev.off()

# 2. Generate Consensus Peaks & Count
# dba.count identifies overlapping peaks to create the consensus set. 
# By default (minOverlap=2), a peak must be present in at least 2 samples to be kept.
print("Generating consensus peakset and counting reads (This may take some time)...")
dba_obj <- dba.count(dba_obj, bUseSummarizeOverlaps=TRUE)

# 3. Extract the consensus coordinates
print("Extracting consensus peak coordinates...")
consensus_peaks <- dba.peakset(dba_obj, bRetrieve=TRUE)

# Convert to a standard dataframe (chr, start, end)
df_consensus <- as.data.frame(consensus_peaks)[, c("seqnames", "start", "end")]

# 4. Export to standard BED format for CREsted
print(paste("Saving", nrow(df_consensus), "consensus peaks to:", OUTPUT_BED))
write.table(df_consensus, file=OUTPUT_BED, sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)

print("Consensus peak generation complete!")
