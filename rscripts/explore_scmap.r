# Author: Fabio Zanini
# Date: 2020-05-27
# Description: explore scMap
# Ah, namespace pollution starts immediately
library(SingleCellExperiment)
library(scmap)


sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(yan)), colData = ann)

# this is needed to calculate dropout rate for feature selection
# important: normcounts have the same zeros as raw counts (fpkm)
counts(sce) <- normcounts(sce)
logcounts(sce) <- log2(normcounts(sce) + 1)

# use gene names as feature symbols
rowData(sce)$feature_symbol <- rownames(sce)
isSpike(sce, 'ERCC') <- grepl('^ERCC-', rownames(sce))

# remove features with duplicated names
sce <- sce[!duplicated(rownames(sce)), ]
sce <- selectFeatures(sce)
sce <- indexCell(sce)

##############################################
# run scmapCell to map the cells back to atlas
##############################################
scmapCell_results <- scmapCell(sce, list(metadata(sce)$scmap_cell_index))
##############################################
