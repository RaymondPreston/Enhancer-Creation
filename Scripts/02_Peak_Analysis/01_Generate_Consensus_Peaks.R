#import required packages
install.packages("ggrepel", repos="http://cran.us.r-project.org")
library(DiffBind)
library(DESeq2)
library(rtracklayer)
library(ggplot2)
library(clusterProfiler)
library(ChIPseeker)
library(TxDb.Mmusculus.UCSC.mm10.knownGene)
library(org.Mm.eg.db)
library(ComplexHeatmap)
library(dplyr)

#cluster all KPC samples together, compare metastatic vs. nonmetastatic
#import sample sheet
samples_all = read.csv("diffbind_sample_sheet_KPC_all_no_H10.csv")

#read in peaksets
KPC_all = dba(sampleSheet=samples_all)

#generate insertion counts matrix
KPC_all_counts = dba.count(KPC_all, summits = 250)

# Extract the full consensus peak set (all peaks, not just DA)
if (!file.exists("input/consensus_peaks_all.bed")) {
    print("Generating consensus_peaks_all.bed...")
    consensus_peaks <- dba.peakset(KPC_all_counts, bRetrieve=TRUE)
    export(consensus_peaks, "input/consensus_peaks_all.bed")
} else {
    print("consensus_peaks_all.bed already exists in input/, skipping export.")
}

#normalize by total reads in peaks
KPC_all_counts_norm_trip = dba.normalize(KPC_all_counts, library=DBA_LIBSIZE_PEAKREADS)

#perform differential accessibility analysis comparing metastatic to non-metastatic samples using matrix normalized by trip
KPC_all_counts_norm_trip_DA = dba.contrast(KPC_all_counts_norm_trip, design="~Tissue + Condition")
KPC_all_counts_norm_trip_DA = dba.analyze(KPC_all_counts_norm_trip_DA)

dba.show(KPC_all_counts_norm_trip_DA, bContrasts=TRUE)

met_vs_nonmet_trip.DB = dba.report(KPC_all_counts_norm_trip_DA, th=0.05, contrast=2)
met_vs_nonmet_trip.DB = met_vs_nonmet_trip.DB[order(met_vs_nonmet_trip.DB$Fold, decreasing=TRUE),]
met_vs_nonmet_trip.DB_up = met_vs_nonmet_trip.DB[met_vs_nonmet_trip.DB$Fold>0,]
met_vs_nonmet_trip.DB_down = met_vs_nonmet_trip.DB[met_vs_nonmet_trip.DB$Fold<0,]
met_vs_nonmet_trip.DB_complete <- dba.report(KPC_all_counts_norm_trip_DA, th=1, contrast=2)

#annotate peaks
dir.create("output", showWarnings = FALSE)

anno = annotatePeak(met_vs_nonmet_trip.DB,  TxDb = TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb="org.Mm.eg.db")
anno_df = as.data.frame(anno)
anno_df = anno_df[order(anno_df$Fold),]
saveRDS(anno_df, file = "output/annotated_differentially_accessible_peaks_no_H10.RDS")

anno_complete = annotatePeak(met_vs_nonmet_trip.DB_complete,  TxDb = TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb="org.Mm.eg.db")
anno_complete_df <- as.data.frame(anno_complete)
anno_complete_df = anno_complete_df[order(-anno_complete_df$Fold),]
saveRDS(anno_complete_df, file = "output/annotated_complete_peakset_no_H10.RDS")

#generate heatmap
report_df = as.data.frame(met_vs_nonmet_trip.DB)
report_df <- report_df[order(-report_df$Fold),]
counts <- dba.peakset(KPC_all_counts_norm_trip_DA, bRetrieve=T, DataType=DBA_DATA_FRAME)
differential_peaks = as.numeric(rownames(report_df))
differential_peak_counts = counts[differential_peaks,]
differential_peak_counts <- differential_peak_counts[,-c(1,2,3)]

mat = data.matrix(differential_peak_counts)
log2mat = log2(1+mat)
log2mat = t(scale(t(log2mat)))

pdf("output/KPC_Consensus_Heatmap.pdf")
Heatmap(log2mat, show_row_names = FALSE, cluster_rows = FALSE)
dev.off()

#Export the differentially accessible peaks to .tsv file for use in generating finetune dataset

# Assign da_class
anno_df$da_class <- ifelse(anno_df$Fold < 0, "met_high", "met_low")

cat(sprintf("\nda_class breakdown:\n  met_high (Fold < 0): %d\n  met_low  (Fold > 0): %d\n",
    sum(anno_df$da_class == "met_high"),
    sum(anno_df$da_class == "met_low")))

# Build 2114 bp peak ID (0-based, centered on Tn5 summit)
# DiffBind with summits=250 produces 501 bp peaks (1-based GRanges coordinates).
# Summit position (0-based) = (start_1based - 1) + 250
# 2114 bp window = summit - 1057 to summit + 1057
summit_0based <- (anno_df$start - 1) + 250
anno_df$peak_id_2114bp <- paste0(
    anno_df$seqnames, ":",
    summit_0based - 1057, "-",
    summit_0based + 1057
)

# Export
out_df <- anno_df[, c(
    "peak_id_2114bp",
    "seqnames", "start", "end",
    "Fold", "FDR", "da_class",
    "Conc_Hi", "Conc_Lo",
    "SYMBOL", "annotation", "distanceToTSS"
)]

write.table(
    out_df,
    file      = "output/DA_peaks_for_finetune_02.tsv",
    sep       = "\t",
    quote     = FALSE,
    row.names = FALSE
)

cat(sprintf("\nSaved: output/DA_peaks_for_finetune.tsv  (%d peaks)\n", nrow(out_df)))
           
