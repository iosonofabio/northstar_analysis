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

    print('Load Darmanis atlas')
    af = northstar.AtlasFetcher()
    adata_dmnf = af.fetch_atlas('Darmanis_2015_nofetal', kind='subsample')

    print('Load Velmeshev autism atlas labndmarks')
    fdn = '../data/Autism/'
    fn_loom = fdn+'subsample.loom'
    adata = anndata.read_loom(fn_loom, sparse=False)
    adata.X = adata.X * 100
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

    atlases = {
        'Velmeshev': adata,
        'Darmanis_2015_nofetal': adata_dmnf,
        }

    print('Run northstar (subsample)')
    ress = []
    for aname, atlas in atlases.items():
        if aname.endswith('avg'):
            continue
        print('Atlas: {:}'.format(aname))
        ns = northstar.Subsample(
            atlas,
            #n_features_per_cell_type=70,
            #n_features_overdispersed=500,
            #n_pcs=25,
            #resolution_parameter=0.003,
            #n_neighbors=30,
            #n_neighbors_external=0,
            #external_neighbors_mutual=True,
            )
        ns.fit(adata_gbm)

        ct_orig = ns.new_data.obs['Cell_type'].astype(str)
        gof = (ct_orig == ns.membership).mean()
        identity = ct_orig.to_frame()
        identity['northstar_assignment'] = ns.membership

        vs = ns.embed('umap')

        resd = {
            'atlas': aname,
            'gof': gof,
            'identity': identity,
            'embedding': vs,
            'class': 'subsample',
            'n_atlas': ns.n_atlas,
            }
        ress.append(resd)

    if False:
        print('Plot results')
        fig, axs = plt.subplots(3, 1, figsize=(6.8, 10.5))
        order = ['Darmanis_2015_nofetal', 'Velmeshev']
        ressd = {val['atlas']: val for val in ress}
        for ax, aname in zip(axs, order):
            resd = ressd[aname]
            identity = resd['identity']
            tmp = identity.copy()
            tmp['c'] = 1
            data = tmp.groupby(['Cell_type', 'northstar_assignment']).sum()['c'].unstack(fill_value=0)

            datan = 100. * (data.T / data.sum(axis=1)).T
            rows = [
                'Astrocyte',
                'Neuron',
                'Endothelial',
                'OPC',
                'Oligodendrocyte',
                'Immune cell',
                'Neoplastic',
               ]
            datan = datan.loc[rows]

            ind = []
            missing = set(list(range(datan.shape[1])))
            for row in datan.values:
                isrt = np.argsort(row)[::-1]
                imax = isrt[0]
                if datan.columns[imax].isdigit():
                    for i in isrt[1:]:
                        if (not datan.columns[i].isdigit()) and (row[i] > 2):
                            imax = i
                            break
                if imax not in ind:
                    ind.append(imax)
                    missing.remove(imax)
            ind += list(missing)
            datan = datan.iloc[:, ind]

            sns.heatmap(
                datan,
                ax=ax,
                xticklabels=True,
                yticklabels=True,
                cmap='plasma',
                )
            ax.set_title(aname)
        fig.suptitle('GBM data (Darmanis et al. 2017)\n[Percent within each correct cell type]')
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    print('Plot results (debugging only)')
    for resd in ress:
        identity = resd['identity']
        tmp = identity.copy()
        tmp['c'] = 1
        data = tmp.groupby(['Cell_type', 'northstar_assignment']).sum()['c'].unstack(fill_value=0)

        datan = 100. * (data.T / data.sum(axis=1)).T
        rows = [
            'Astrocyte',
            'Neuron',
            'Endothelial',
            'OPC',
            'Oligodendrocyte',
            'Immune cell',
            'Neoplastic',
           ]
        datan = datan.loc[rows]

        ind = []
        missing = set(list(range(datan.shape[1])))
        for row in datan.values:
            isrt = np.argsort(row)[::-1]
            imax = isrt[0]
            if datan.columns[imax].isdigit():
                for i in isrt[1:]:
                    if (not datan.columns[i].isdigit()) and (row[i] > 2):
                        imax = i
                        break
            if imax not in ind:
                ind.append(imax)
                missing.remove(imax)
        ind += list(missing)
        datan = datan.iloc[:, ind]

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        sns.heatmap(
            datan,
            ax=axs[0],
            xticklabels=True,
            yticklabels=True,
            cmap='plasma',
            )
        axs[0].set_title(resd['atlas'])

        vs = resd['embedding']
        n_atlas = resd['n_atlas']

        # Embedding with original cell types
        vs['CellType_original'] = vs['CellType'].copy()
        vs.loc[adata_gbm.obs_names, 'CellType_original'] = adata_gbm.obs['CellType']

        for ico, colname in enumerate(['CellType_original', 'CellType']):
            cols = vs[colname].unique()
            cmap = dict(zip(cols, sns.color_palette('husl', n_colors=len(cols))))
            for col in cols:
                x0, y0 = vs.iloc[:n_atlas].loc[vs[colname] == col].values.T[:2]
                x1, y1 = vs.iloc[n_atlas:].loc[vs[colname] == col].values.T[:2]
                axs[1+ico].scatter(
                    x0, y0, color=cmap[col],
                    s=45,
                    marker='*',
                    alpha=0.6,
                    )
                axs[1+ico].scatter(
                    x1, y1, color=cmap[col],
                    s=30,
                    marker='o',
                    alpha=0.6,
                    )
            hs = [axs[-1].scatter([], [], marker='o', color=cmap[cu]) for cu in cols]
            axs[1+ico].legend(
                hs, cols,
                loc='upper left', ncol=2,
                bbox_to_anchor=(-0.2, -0.1), bbox_transform=axs[1+ico].transAxes,
                fontsize=9,
                )

        fig.tight_layout()


    plt.ion()
    plt.show()
