# vim: fdm=indent
'''
author:     Fabio Zanini
date:       19/05/20
content:    Compare our Fig 2 with the autism atlas by Velmeshev et al:

https://science.sciencemag.org/content/364/6441/685.long
'''
import os
import sys
import pickle
import time
import gzip
from collections import Counter
import numpy as np
import pandas as pd
import anndata
import loompy
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/home/fabio/university/postdoc/singlet')
import singlet


if __name__ == '__main__':

    print('Load autism data (subsampled)')
    fdn = '../data/Autism/'
    fn_loom = fdn+'subsample.loom'

    ds = singlet.Dataset(
        dataset={
            'path': fn_loom,
            'index_samples': 'CellID',
            'index_features': 'GeneName',
            })
    features = ds.feature_selection.overdispersed_within_groups('sample')
    dsf = ds.query_features_by_name(features)
    dsc = dsf.dimensionality.pca(n_dims=30, return_dataset='samples')
    vs = dsc.dimensionality.umap()

    cus = ds.samplesheet['cluster'].unique()
    cmap = dict(zip(cus, sns.color_palette('husl', n_colors=len(cus))))
    fig, ax = plt.subplots(figsize=(6, 4))
    for cu in cus:
        x, y = vs.loc[ds.samplesheet['cluster'] == cu].values.T
        ax.scatter(x, y, s=30, color=cmap[cu], alpha=0.6, label=cu)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.01), bbox_transform=ax.transAxes,
        )
    fig.tight_layout()

    plt.ion()
    plt.show()
