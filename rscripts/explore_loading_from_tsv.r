# Author: Fabio Zanini
# Date: 2020-05-27
# Description: learn to load tsv files into SCE
# Ah, namespace pollution starts immediately
library(SingleCellExperiment)


fn_atlas <- '../data/for_scmap/TBS_kidney_atlas_subsample_20_counts.tsv'

df <- read.table(file = fn_atlas, sep = '\t', header = TRUE)

#sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(yan)), colData = ann)
