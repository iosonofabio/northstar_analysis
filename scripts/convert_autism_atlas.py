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

    # FIXME: umap shows this is messed up

    print('Subsample Velmeshev dataset')
    fdn = '../data/Autism/'
    fn = fdn+'exprMatrix.tsv.gz'
    fn_meta = fdn+'meta.tsv'
    fns_loom = [
        #fdn+'subsample.loom',
        fdn+'subsample_large.loom',
        fdn+'subsample_control.loom',
        fdn+'subsample_large_ASD.loom',
        #fdn+'subsample_raw.loom',
        ]
    for fn_loom in fns_loom:
        print('Get gene IDs')
        if not os.path.isfile(fdn+'gids.tsv'):
            with gzip.open(fn, 'rt') as f:
                f.readline()
                gids = []
                for il, line in enumerate(f):
                    print(il+1, end='\r')
                    line = line[:100]
                    line = line.split('\t')[0]
                    gids.append(line)
                print()
            with open(fdn+'gids.tsv', 'wt') as f:
                f.write('\n'.join(gids))
        else:
            with open(fdn+'gids.tsv', 'rt') as f:
                gids = f.read().split('\n')

        print('Get metadata')
        meta = pd.read_csv(fn_meta, sep='\t', index_col='cell')

        print('Get cell names')
        with gzip.open(fn, 'rt') as f:
            cells = np.array(f.readline().rstrip('\n').split()[1:])
        #(cells == meta.index).all() yes, same order

        # Select subsample
        if 'large' in fn_loom:
            nsub = 500
        else:
            nsub = 20
        cts_counts = meta['cluster'].value_counts()
        inds = {}
        for ct, nct in cts_counts.items():
            ind = meta['cluster'] == ct
            if 'control' in fn_loom:
                ind &= meta['diagnosis'] == 'Control'
            elif 'ASD' in fn_loom:
                ind &= meta['diagnosis'] == 'ASD'
            ind = ind.values.nonzero()[0]
            if nct > nsub:
                ind = np.random.choice(ind, size=nsub, replace=False)
            inds[ct] = ind
        inds_array = np.concatenate([v for v in inds.values()])
        inds_array.sort()
        cellsf = cells[inds_array]
        metaf = meta.iloc[inds_array]

        print('Read counts (normalized by UMIs, log-transformed)')
        matrix = np.empty((len(gids), len(inds_array)), np.float32)
        with gzip.open(fn, 'rt') as f:
            cells_tsv = f.readline().rstrip('\n').split('\t')[1:]
            cellsf_tsv = np.array(cells_tsv)[inds_array]
            for il, line in enumerate(f):
                if ((il + 1) % 100) == 0:
                    print(il+1, end='\r')
                # The first column is gene ID, so skip it
                fields = line.split('\t')[1:]
                fields[-1] = fields[-1].rstrip('\n')
                for ii, i in enumerate(inds_array):
                    matrix[il, ii] = np.float32(fields[i])
            print()
        # It's log2 transformed, pseudocounted, and normalized to 10,000
        matrix = 2**matrix - 1
        if 'raw' in fn_loom:
            matrix = np.round((matrix / matrix.sum(axis=0) * metaf['UMIs'].values), 0).astype(np.int64)

        counts = pd.DataFrame(
                matrix,
                index=gids,
                columns=cellsf_tsv,
                )

        print('Convert gene IDs into gene names')
        fn_conv = fdn+'mart_export.tsv'
        genes = pd.read_csv(fn_conv, sep='\t', squeeze=True, index_col=0)
        gids_int = np.intersect1d(gids, genes.index.values)
        genes = genes.loc[gids_int]
        gn_counts = genes.value_counts()
        gn_single = list(gn_counts.index[gn_counts == 1])
        gn_multiple = list(gn_counts.index[gn_counts > 1])
        gid_simp = list(genes.index[genes.isin(gn_single)])
        tmp = set(gn_multiple)
        missing_ids = []
        for gid in genes.index[genes.isin(gn_multiple)]:
            gn = genes.loc[gid]
            if gn not in tmp:
                missing_ids.append((gid, gn))
            else:
                gid_simp.append(gid)
                tmp.remove(gn)
        gn_simp = genes.loc[gid_simp]
        counts_gn = counts.loc[gid_simp]
        counts_gn.index = gn_simp
        for (gid, gn) in missing_ids:
            counts_gn.loc[gn] += counts.loc[gid]

        print('Export to loom file')
        col_attrs = {col: metaf[col].values for col in metaf.columns}
        col_attrs['CellID'] = metaf.index.values
        row_attrs = {'GeneName': counts_gn.index.values}
        loompy.create(
                fn_loom,
                counts_gn.values,
                col_attrs=col_attrs,
                row_attrs=row_attrs,
            )

        print('Load back loom file to check umap')
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
