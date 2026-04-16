# Set mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

pkgs <- c(
    "DiffBind", 
    "DESeq2", 
    "rtracklayer", 
    "ggplot2", 
    "clusterProfiler", 
    "ChIPseeker", 
    "TxDb.Mmusculus.UCSC.mm10.knownGene", 
    "org.Mm.eg.db", 
    "ComplexHeatmap", 
    "dplyr"
)

# Identify missing packages
missing_pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]

if (length(missing_pkgs)) {
    print(paste("Installing missing packages:", paste(missing_pkgs, collapse = ", ")))
    BiocManager::install(missing_pkgs, update = FALSE, ask = FALSE)
} else {
    print("All required packages are already installed.")
}
