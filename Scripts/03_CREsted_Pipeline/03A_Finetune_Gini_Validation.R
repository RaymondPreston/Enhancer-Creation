#!/usr/bin/env Rscript
# =============================================================================
# Gini_QC_vs_DESeq.R
#
#Based on crested.pp.filter_regions_on_specificity(adata, gini_std_threshold=1.0, inplace=False)
#Goal is to validate that the peaks I pull out of the above function are the peaks I want to be training on.
#The peaks that I want to be pulled out are Met-high and Met-low specific peaks.
#
# QC: Do the Gini-filtered peaks (adata_specific) capture the
#     differentially accessible met-high and met-low peaks from DESeq2?
#
# Inputs:
#   1. /scratch/rprest2/Enhancer-Creation/input/training_inputs/01_a_ft_training_set.h5ad
#      (Gini-filtered AnnData — var index = peak IDs)
#   2. /scratch/rprest2/Enhancer-Creation/output/annotated_differentially_accessible_peaks_no_H10.RDS
#      (DESeq2 DA peaks with Fold, FDR, coordinates, annotations)
#
# Outputs (saved to output/Gini_QC/):
#   - gini_deseq_overlap_summary.csv   : per-peak classification table
#   - gini_deseq_overlap_barplot.svg   : stacked bar of Gini peak composition
#   - volcano_gini_highlight.svg       : DESeq2 volcano with Gini peaks highlighted
#   - fold_distribution_gini.svg       : fold-change distribution of Gini DA peaks
#   - gini_qc_stats.txt                : summary statistics
# =============================================================================

Sys.setenv(RETICULATE_PYTHON = "/users/rprest2/.conda/envs/Crested_QC/bin/python")

suppressPackageStartupMessages({
  library(anndata)
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(tidyr)
  library(svglite)
  library(GenomicRanges)
})

# ── Output directory ──────────────────────────────────────────────────────────
out_dir <- "/scratch/rprest2/Enhancer-Creation/output/Gini_QC"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Gini QC vs DESeq2 DA Peaks ===\n\n")

# =============================================================================
# 1. Load Gini-filtered AnnData (adata_specific)
# =============================================================================
cat("Loading Gini-filtered AnnData...\n")
adata <- read_h5ad("/scratch/rprest2/Enhancer-Creation/input/training_inputs/01_a_ft_training_set.h5ad")

# var index contains peak IDs — extract as a data frame
gini_var <- as.data.frame(adata$var)
gini_var$peak_id_raw <- rownames(gini_var)
cat(sprintf("  Gini-filtered peaks: %d\n", nrow(gini_var)))
cat(sprintf("  Gini peak ID format (first 3): %s\n",
            paste(head(rownames(gini_var), 3), collapse = ", ")))

# =============================================================================
# 2. Load DESeq2 DA peaks
# =============================================================================
cat("\nLoading DESeq2 DA peaks...\n")
da_peaks <- readRDS("/scratch/rprest2/Enhancer-Creation/output/annotated_differentially_accessible_peaks_no_H10.RDS")
da_peaks <- as.data.frame(da_peaks)

cat(sprintf("  Total DESeq2 peaks: %d\n", nrow(da_peaks)))
cat(sprintf("  Columns: %s\n", paste(colnames(da_peaks), collapse = ", ")))
cat(sprintf("  DESeq2 peak coord columns (first 3 rows):\n"))
print(head(da_peaks[, c("seqnames", "start", "end", "Fold", "FDR")], 3))

# =============================================================================
# 3. Harmonize peak ID formats using GenomicRanges Overlaps
# =============================================================================
library(GenomicRanges)

# Build GRanges for DESeq2 peaks
# (DiffBind uses 1-based coordinates in R)
da_gr <- GRanges(
  seqnames = da_peaks$seqnames,
  ranges = IRanges(start = da_peaks$start, end = da_peaks$end)
)
mcols(da_gr) <- da_peaks

# Parse Gini peak IDs (format: chr:start-end, 0-based start, 1-based end)
gini_parts <- strsplit(gini_var$peak_id_raw, "[:-]")
gini_seqnames <- sapply(gini_parts, `[`, 1)
# Add 1 to start to convert 0-based BED to 1-based R/GRanges coordinates
gini_start <- as.numeric(sapply(gini_parts, `[`, 2)) + 1
gini_end <- as.numeric(sapply(gini_parts, `[`, 3))

gini_gr <- GRanges(
  seqnames = gini_seqnames,
  ranges = IRanges(start = gini_start, end = gini_end)
)
mcols(gini_gr)$peak_id_raw <- gini_var$peak_id_raw

cat(sprintf("\nFinding spatial overlaps between %d Gini peaks and %d DESeq2 peaks...\n",
            length(gini_gr), length(da_gr)))

# Find overlaps (Gini peaks vs DESeq2 peaks)
overlaps <- findOverlaps(gini_gr, da_gr)

cat(sprintf("  Found overlaps for %d / %d Gini peaks\n",
            length(unique(queryHits(overlaps))), length(gini_gr)))

# =============================================================================
# 4. Classify DESeq2 peaks: met-high, met-low, or not-DA
# =============================================================================
# Fold > 0 = increased in Lo (met-low enriched)
# Fold < 0 = increased in Hi (met-high enriched)
# FDR threshold: 0.05

fdr_thresh <- 0.05

da_peaks <- da_peaks %>%
  mutate(
    da_class = case_when(
      FDR < fdr_thresh & Fold < 0 ~ "met_high",   # negative fold = higher in Hi
      FDR < fdr_thresh & Fold > 0 ~ "met_low",    # positive fold = higher in Lo
      TRUE                         ~ "not_DA"
    )
  )
# Update the GRanges object with the new classifications
mcols(da_gr)$da_class <- da_peaks$da_class

cat(sprintf("\nDESeq2 peak classification (FDR < %.2f):\n", fdr_thresh))
print(table(da_peaks$da_class))

# =============================================================================
# 5. Annotate Gini peaks with DESeq2 classification
# =============================================================================
# Initialize with not_in_DESeq2
gini_annotated <- gini_var
gini_annotated$da_class <- "not_in_DESeq2"
gini_annotated$Fold <- NA
gini_annotated$FDR <- NA
gini_annotated$p.value <- NA
gini_annotated$Conc <- NA
gini_annotated$Conc_Hi <- NA
gini_annotated$Conc_Lo <- NA
gini_annotated$annotation <- NA
gini_annotated$SYMBOL <- NA
gini_annotated$distanceToTSS <- NA

# Map the metadata from DESeq2 peaks to the overlapping Gini peaks
# If a Gini peak overlaps multiple DESeq2 peaks, we take the first one (or the most significant one ideally, but first is fine for this check)
overlap_df <- data.frame(
  gini_idx = queryHits(overlaps),
  da_idx = subjectHits(overlaps)
)

# Optional: if multiple overlaps, keep the one with lowest FDR
overlap_df$FDR <- mcols(da_gr)$FDR[overlap_df$da_idx]
overlap_df <- overlap_df %>%
  arrange(gini_idx, FDR) %>%
  distinct(gini_idx, .keep_all = TRUE)

# Assign values
matched_gini_idx <- overlap_df$gini_idx
matched_da_idx <- overlap_df$da_idx

gini_annotated$da_class[matched_gini_idx] <- mcols(da_gr)$da_class[matched_da_idx]
gini_annotated$Fold[matched_gini_idx] <- mcols(da_gr)$Fold[matched_da_idx]
gini_annotated$FDR[matched_gini_idx] <- mcols(da_gr)$FDR[matched_da_idx]
gini_annotated$p.value[matched_gini_idx] <- mcols(da_gr)$p.value[matched_da_idx]
gini_annotated$Conc[matched_gini_idx] <- mcols(da_gr)$Conc[matched_da_idx]
gini_annotated$Conc_Hi[matched_gini_idx] <- mcols(da_gr)$Conc_Hi[matched_da_idx]
gini_annotated$Conc_Lo[matched_gini_idx] <- mcols(da_gr)$Conc_Lo[matched_da_idx]
gini_annotated$annotation[matched_gini_idx] <- mcols(da_gr)$annotation[matched_da_idx]
gini_annotated$SYMBOL[matched_gini_idx] <- mcols(da_gr)$SYMBOL[matched_da_idx]
gini_annotated$distanceToTSS[matched_gini_idx] <- mcols(da_gr)$distanceToTSS[matched_da_idx]

cat(sprintf("\nGini peak classification:\n"))
print(table(gini_annotated$da_class))

# Compute percentages
gini_class_pct <- gini_annotated %>%
  count(da_class) %>%
  mutate(pct = round(100 * n / sum(n), 1))
cat("\nGini peak composition (%):\n")
print(gini_class_pct)

# =============================================================================
# 6. Save summary table
# =============================================================================
write.csv(gini_annotated,
          file.path(out_dir, "gini_deseq_overlap_summary.csv"),
          row.names = FALSE)
cat(sprintf("\nSaved: gini_deseq_overlap_summary.csv\n"))

# =============================================================================
# 7. Plot 1: Stacked bar — composition of Gini peaks
# =============================================================================
color_map <- c(
  "met_high"     = "#E64B35",   # red
  "met_low"      = "#4DBBD5",   # blue
  "not_DA"       = "#B0B0B0",   # grey
  "not_in_DESeq2" = "#F0E442"   # yellow (should be near-zero)
)

p_bar <- ggplot(gini_class_pct, aes(x = "Gini peaks", y = pct, fill = da_class)) +
  geom_col(width = 0.5, color = "white", linewidth = 0.3) +
  geom_text(aes(label = paste0(pct, "%\n(n=", n, ")")),
            position = position_stack(vjust = 0.5),
            size = 3.5, color = "white", fontface = "bold") +
  scale_fill_manual(values = color_map,
                    labels = c("met_high" = "Met-high DA",
                               "met_low"  = "Met-low DA",
                               "not_DA"   = "Not DA (FDR ≥ 0.05)",
                               "not_in_DESeq2" = "Not in DESeq2")) +
  labs(title = "Composition of Gini-filtered peaks",
       subtitle = sprintf("n = %d total Gini peaks | FDR threshold = %.2f",
                          nrow(gini_annotated), fdr_thresh),
       x = NULL, y = "Percentage (%)", fill = "DESeq2 class") +
  theme_classic(base_size = 13) +
  theme(
    text = element_text(family = "Arial"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "right"
  )

svglite(file.path(out_dir, "gini_deseq_overlap_barplot.svg"), width = 5, height = 6)
print(p_bar)
dev.off()
cat("Saved: gini_deseq_overlap_barplot.svg\n")

# =============================================================================
# 8. Plot 2: Volcano plot of ALL DESeq2 peaks, Gini peaks highlighted
# =============================================================================
da_peaks_plot <- da_peaks %>%
  mutate(
    in_gini = row_number() %in% subjectHits(overlaps),
    neg_log10_fdr = -log10(pmax(FDR, 1e-300)),
    point_group = case_when(
      in_gini & da_class == "met_high" ~ "Gini + met-high",
      in_gini & da_class == "met_low"  ~ "Gini + met-low",
      in_gini & da_class == "not_DA"   ~ "Gini + not DA",
      !in_gini & da_class == "met_high" ~ "Not Gini + met-high",
      !in_gini & da_class == "met_low"  ~ "Not Gini + met-low",
      TRUE                              ~ "Background"
    )
  )

volcano_colors <- c(
  "Gini + met-high"      = "#E64B35",
  "Gini + met-low"       = "#4DBBD5",
  "Gini + not DA"        = "#F5A623",
  "Not Gini + met-high"  = "#FFAAAA",
  "Not Gini + met-low"   = "#AAE0EE",
  "Background"           = "#DDDDDD"
)

volcano_sizes <- c(
  "Gini + met-high"      = 1.5,
  "Gini + met-low"       = 1.5,
  "Gini + not DA"        = 1.2,
  "Not Gini + met-high"  = 0.8,
  "Not Gini + met-low"   = 0.8,
  "Background"           = 0.4
)

# Plot background first, then highlighted points on top
p_volcano <- ggplot() +
  geom_point(data = filter(da_peaks_plot, point_group == "Background"),
             aes(x = Fold, y = neg_log10_fdr, color = point_group, size = point_group),
             alpha = 0.3) +
  geom_point(data = filter(da_peaks_plot, point_group != "Background"),
             aes(x = Fold, y = neg_log10_fdr, color = point_group, size = point_group),
             alpha = 0.7) +
  geom_hline(yintercept = -log10(fdr_thresh), linetype = "dashed",
             color = "black", linewidth = 0.5) +
  geom_vline(xintercept = 0, linetype = "solid",
             color = "black", linewidth = 0.3) +
  scale_color_manual(values = volcano_colors) +
  scale_size_manual(values = volcano_sizes) +
  annotate("text", x = min(da_peaks_plot$Fold, na.rm = TRUE) * 0.9,
           y = -log10(fdr_thresh) + 0.3,
           label = sprintf("FDR = %.2f", fdr_thresh),
           size = 3, hjust = 0) +
  annotate("text", x = -2, y = max(da_peaks_plot$neg_log10_fdr, na.rm = TRUE) * 0.95,
           label = "← Met-high enriched", size = 3.5, color = "#E64B35") +
  annotate("text", x = 2, y = max(da_peaks_plot$neg_log10_fdr, na.rm = TRUE) * 0.95,
           label = "Met-low enriched →", size = 3.5, color = "#4DBBD5") +
  labs(
    title = "DESeq2 volcano: Gini-filtered peaks highlighted",
    subtitle = sprintf("Gini peaks: %d met-high | %d met-low | %d not-DA",
                       sum(gini_annotated$da_class == "met_high"),
                       sum(gini_annotated$da_class == "met_low"),
                       sum(gini_annotated$da_class == "not_DA")),
    x = "Log2 fold-change (positive = met-low enriched)",
    y = "-log10(FDR)",
    color = "Peak group",
    size = "Peak group"
  ) +
  theme_classic(base_size = 13) +
  theme(
    text = element_text(family = "Arial"),
    legend.position = "right"
  )

svglite(file.path(out_dir, "volcano_gini_highlight.svg"), width = 9, height = 6)
print(p_volcano)
dev.off()
cat("Saved: volcano_gini_highlight.svg\n")

# =============================================================================
# 9. Plot 3: Fold-change distribution of Gini DA peaks
# =============================================================================
gini_da_only <- gini_annotated %>%
  filter(da_class %in% c("met_high", "met_low"))

p_fold <- ggplot(gini_da_only, aes(x = Fold, fill = da_class)) +
  geom_histogram(bins = 60, alpha = 0.8, position = "identity", color = "white", linewidth = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  scale_fill_manual(values = c("met_high" = "#E64B35", "met_low" = "#4DBBD5"),
                    labels = c("met_high" = "Met-high DA", "met_low" = "Met-low DA")) +
  labs(
    title = "Fold-change distribution of Gini DA peaks",
    subtitle = "Gini-filtered peaks that are also differentially accessible",
    x = "Log2 fold-change (positive = met-low enriched)",
    y = "Count",
    fill = "Class"
  ) +
  theme_classic(base_size = 13) +
  theme(text = element_text(family = "Arial"))

svglite(file.path(out_dir, "fold_distribution_gini.svg"), width = 7, height = 5)
print(p_fold)
dev.off()
cat("Saved: fold_distribution_gini.svg\n")

# =============================================================================
# 10. Summary statistics text file
# =============================================================================
sink(file.path(out_dir, "gini_qc_stats.txt"))
cat("=== Gini QC vs DESeq2 Summary ===\n\n")
cat(sprintf("Gini-filtered peaks (adata_specific): %d\n", nrow(gini_annotated)))
cat(sprintf("DESeq2 total peaks tested:             %d\n", nrow(da_peaks)))
cat(sprintf("FDR threshold used:                    %.2f\n\n", fdr_thresh))

cat("--- Gini peak composition ---\n")
print(gini_class_pct)

cat(sprintf("\n--- Recall: what fraction of DA peaks are captured by Gini? ---\n"))
n_da_methigh_total <- sum(da_peaks$da_class == "met_high")
n_da_metlow_total  <- sum(da_peaks$da_class == "met_low")
n_gini_methigh     <- sum(gini_annotated$da_class == "met_high")
n_gini_metlow      <- sum(gini_annotated$da_class == "met_low")

cat(sprintf("Met-high DA peaks in Gini set: %d / %d (%.1f%%)\n",
            n_gini_methigh, n_da_methigh_total,
            100 * n_gini_methigh / n_da_methigh_total))
cat(sprintf("Met-low DA peaks in Gini set:  %d / %d (%.1f%%)\n",
            n_gini_metlow, n_da_metlow_total,
            100 * n_gini_metlow / n_da_metlow_total))

cat(sprintf("\n--- Precision: what fraction of Gini peaks are DA? ---\n"))
cat(sprintf("Gini peaks that are met-high DA: %d / %d (%.1f%%)\n",
            n_gini_methigh, nrow(gini_annotated),
            100 * n_gini_methigh / nrow(gini_annotated)))
cat(sprintf("Gini peaks that are met-low DA:  %d / %d (%.1f%%)\n",
            n_gini_metlow, nrow(gini_annotated),
            100 * n_gini_metlow / nrow(gini_annotated)))
cat(sprintf("Gini peaks that are any DA:      %d / %d (%.1f%%)\n",
            n_gini_methigh + n_gini_metlow, nrow(gini_annotated),
            100 * (n_gini_methigh + n_gini_metlow) / nrow(gini_annotated)))

cat(sprintf("\n--- Peak ID matching ---\n"))
cat(sprintf("DESeq2 peaks overlapped with Gini peaks: %d / %d\n",
            length(unique(subjectHits(overlaps))), nrow(da_peaks)))
sink()
cat("Saved: gini_qc_stats.txt\n")

cat("\n=== Done. All outputs in:", out_dir, "===\n")
