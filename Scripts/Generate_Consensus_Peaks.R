#import required packages
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
consensus_peaks <- dba.peakset(KPC_all_counts, bRetrieve=TRUE)
export(consensus_peaks, "consensus_peaks_all.bed")

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
anno = annotatePeak(met_vs_nonmet_trip.DB,  TxDb = TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb="org.Mm.eg.db")
anno_df = as.data.frame(anno)
anno_df = anno_df[order(anno_df$Fold),]
saveRDS(anno_df, file = "annotated_differentially_accessible_peaks_no_H10.RDS")

anno_complete = annotatePeak(met_vs_nonmet_trip.DB_complete,  TxDb = TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb="org.Mm.eg.db")
anno_complete_df <- as.data.frame(anno_complete)
anno_complete_df = anno_complete_df[order(-anno_complete_df$Fold),]
saveRDS(anno_complete_df, file = "annotated_complete_peakset_no_H10.RDS")

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
Heatmap(log2mat, show_row_names = FALSE, cluster_rows = FALSE)
           
