# vim: fdm=indent
'''
author:     Fabio Zanini
date:       21/05/20
content:    Combine scVI harmonization with northstar clustering.
'''
import os
import sys
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import anndata
import scvi
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/home/fabio/university/postdoc/northstar/build/lib')
import northstar


if __name__ == '__main__':

    print('Load autism data (subsampled)')
    fdn = '../data/Autism/'
    fn_loom = fdn+'subsample_raw.loom'
    adata = anndata.read_loom(fn_loom, sparse=False)
    adata.var_names = adata.var['GeneName']
    adata.obs['CellType'] = adata.obs['cluster']
    adata.obs['CellType'].replace({
        'L2/3': 'Neuron',
        'L5/6': 'Neuron',
        'L4': 'Neuron',
        'L5/6-CC': 'Neuron',
        'IN-VIP': 'Neuron',
        'IN-PV': 'Neuron',
        'IN-SV2C': 'Neuron',
        'IN-SST': 'Neuron',
        'Neu-NRGN-I': 'Neuron',
        'Neu-NRGN-II': 'Neuron',
        'Neu-mat': 'Neuron',
        'AST-PP': 'Astrocyte',
        'AST-FB': 'Astrocyte',
    }, inplace=True)
    # scVI wants absolute counts
    adata.X = adata.X.astype(np.int64)

    print('Load GBM data')
    fdn_gbm = '../data/GBM_data_and_metadata/'
    fn_loom_gbm = fdn_gbm+'GBM_data.loom'
    adata_gbm = anndata.read_loom(fn_loom_gbm, sparse=False)
    adata_gbm.var_names = adata_gbm.var['GeneName']
    # scVI wants absolute counts
    adata_gbm = (adata_gbm.X.T / adata_gbm.X.sum(axis=1) * adata_gbm.obs['Unique_reads'].values).T.astype(np.int64)
