library(readxl)
library(GenomicRanges)
library(dplyr)

# Load data
my_peaks_df <- readRDS("output/annotated_differentially_accessible_peaks_no_H10.RDS")
pi_peaks_df <- read_excel("Handler_Differential_Peaks.xlsx")

# Convert to GRanges objects
my_gr <- GRanges(
  seqnames = my_peaks_df$seqnames,
  ranges = IRanges(start = my_peaks_df$start, end = my_peaks_df$end)
)
mcols(my_gr) <- my_peaks_df

pi_gr <- GRanges(
  seqnames = pi_peaks_df$seqnames,
  ranges = IRanges(start = pi_peaks_df$start, end = pi_peaks_df$end)
)
mcols(pi_gr) <- pi_peaks_df

# Find overlaps
overlaps <- findOverlaps(my_gr, pi_gr)

# Overlapping peaks counts
my_overlapping_count <- length(unique(queryHits(overlaps)))
pi_overlapping_count <- length(unique(subjectHits(overlaps)))

cat("Total peaks in our analysis:", length(my_gr), "\n")
cat("Total peaks in PI's analysis:", length(pi_gr), "\n")
cat("Number of our peaks overlapping PI's peaks:", my_overlapping_count, "\n")
cat("Number of PI's peaks overlapping our peaks:", pi_overlapping_count, "\n")

# Non-overlapping peaks
my_unique_gr <- my_gr[-queryHits(overlaps)]
pi_unique_gr <- pi_gr[-subjectHits(overlaps)]

cat("Number of peaks unique to our analysis:", length(my_unique_gr), "\n")
cat("Number of peaks unique to PI's analysis:", length(pi_unique_gr), "\n")

# Save unique peaks
write.csv(as.data.frame(my_unique_gr), "output/unique_to_our_analysis.csv", row.names = FALSE)
write.csv(as.data.frame(pi_unique_gr), "output/unique_to_pi_analysis.csv", row.names = FALSE)

cat("Unique peak lists saved to output/unique_to_our_analysis.csv and output/unique_to_pi_analysis.csv\n")
