# vim: fdm=indent
'''
author:     Fabio Zanini
date:       12/05/20
content:    Test northstar on a large dataset from Tabula Muris Senis
'''
import os
import sys
import time
from multithreading import Pool
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/home/fabio/university/postdoc/northstar/build/lib')
import northstar


def run_northstar(
            adata,
            asub,
            nct,
            rep,
            ):
    print('Subsample with {:} cells per type'.format(nct))
    asub2 = northstar.subsample_atlas(
            adata,
            cell_type_column='cell_ontology_class',
            n_cells=nct,
            )
    ntot = asub2.X.shape[0]

    print('Run northstar')
    t0 = time.time()
    ns = northstar.Subsample(
        asub,
        )
    ns.fit(asub2)
    t1 = time.time()
    t = t1 - t0

    ct_orig = ns.new_data.obs['cell_ontology_class'].astype(str)
    gof = (ct_orig == ns.membership).mean()

    resd = {
        'time': t,
        'gof': gof,
        'ntot': ntot,
        'rep': rep,
        'tissue': tissue,
        }

    return resd


if __name__ == '__main__':

    tissues = [
            'Kidney',
            'Bladder',
            'Fat',
            'Heart',
            'Marrow',
            'Skin',
            'Spleen',
            ]

    ress = []
    ncts = [5, 10, 30, 100, 300, 1000]
    nreps = 5
    ncts = [300, 1000]
    for tissue in tissues:
        print('Test northstar on tissue: {:}'.format(tissue))
        adata = anndata.read_h5ad('../data/TabulaMurisSenis/{:}_droplet.h5ad'.format(tissue))

        asub = northstar.subsample_atlas(
                adata,
                cell_type_column='cell_ontology_class',
                n_cells=20,
                )
        asub.obs['CellType'] = asub.obs['cell_ontology_class']

        res = []
        for nct in ncts:
            for rep in range(nreps):
                resd = run_northstar(adata, asub, nct, rep)
                res.append(resd)
        res = pd.DataFrame(res)
        ress.append(res)

    resa = pd.concat(ress)

    print('Plot results')
    colors = sns.color_palette('Dark2', n_colors=len(tissues))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    for it, tissue in enumerate(tissues):
        res_avg = resa.loc[resa['tissue'] == tissue].groupby('ntot').mean()[['time', 'gof']]
        res_std = resa.loc[resa['tissue'] == tissue].groupby('ntot').std()[['time', 'gof']]

        ax.errorbar(
                res_avg.index, res_avg['time'],
                yerr=res_std['time'],
                fmt='-o', label=tissue,
                lw=2, color=colors[it],
                )

        ax2.errorbar(
                res_avg.index, 100 * res_avg['gof'],
                yerr=100 * res_std['gof'],
                fmt='-o', label=tissue,
                lw=2, color=colors[it],
                )
    ax.set_xlabel('Number of cells\nin new data set')
    ax.set_ylabel('Runtime [s]')
    ax.set_ylim(0.1, 60)
    ax.set_xlim(14.298061502263902, 19878.288366859615)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_xlabel('Number of cells\nin new data set')
    ax2.set_ylabel('Correct assignments [%]')
    ax2.set_ylim(0, 100)
    ax2.set_xscale('log')
    ax2.legend(
            loc='upper left', title='Tissue:',
            bbox_to_anchor=(1.01, 1.01),
            bbox_transform=ax2.transAxes,
            )
    fig.tight_layout()
    fig.savefig('../figures/scaling_newdata_size_TMS_alltissues.svg')
    fig.savefig('../figures/scaling_newdata_size_TMS_alltissues.png', dpi=600)

    plt.ion()
    plt.show()
