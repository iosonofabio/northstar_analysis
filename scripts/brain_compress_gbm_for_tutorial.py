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

sys.path.insert(0, '/home/fabio/university/postdoc/northstar/build/lib')
import northstar


if __name__ == '__main__':

    print('Load GBM data')
    fdn_gbm = '../data/GBM_data_and_metadata/'
    fn_loom_gbm = fdn_gbm+'GBM_data.loom'
    adata_gbm = anndata.read_loom(fn_loom_gbm, sparse=False)
    adata_gbm.var_names = adata_gbm.var['GeneName']
    adata_gbm.X = 1e6 * (adata_gbm.X.T / adata_gbm.X.sum(axis=1)).T
    adata_gbm.obs['CellType'] = adata_gbm.obs['Cell_type']

    print('Compress GBM data')
    adata_gbm = northstar.subsample_atlas(
        adata_gbm,
        n_cells=50,
        )
    adata_gbm.write_loom(
            '../data/GBM_data_and_metadata/GBM_data_for_tutorial.loom',
            )

