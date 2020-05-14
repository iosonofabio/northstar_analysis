# vim: fdm=indent
'''
author:     Fabio Zanini
date:       12/05/20
content:    Test northstar on a large dataset from Tabula Muris Senis
'''
import os
import sys
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/home/fabio/university/postdoc/northstar/build/lib')
import northstar


if __name__ == '__main__':

    tissue = 'Kidney'
    adata = anndata.read_h5ad('../data/TabulaMurisSenis/{:}_droplet.h5ad'.format(tissue))

    asub = northstar.subsample_atlas(
            adata,
            cell_type_column='cell_ontology_class',
            n_cells=20,
            )
    asub.obs['CellType'] = asub.obs['cell_ontology_class']

    import time
    ncts = [5, 10, 30, 100, 300, 1000]
    nreps = 5
    res = []
    for nct in ncts:
        for rep in range(nreps):
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

            res.append({
                'time': t,
                'gof': gof,
                'ntot': ntot,
                'rep': rep,
                })

    res = pd.DataFrame(res)

    res_avg = res.groupby('ntot').mean()[['time', 'gof']]
    res_std = res.groupby('ntot').std()[['time', 'gof']]

    print('Plot results')
    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.errorbar(
            res_avg.index, res_avg['time'],
            yerr=res_std['time'],
            fmt='-o',
            lw=2, color=colors[0],
            )
    ax.set_xlabel('Number of cells\nin new data set')
    ax.set_ylabel('Runtime [s]')
    ax.scatter(
            [-0.16], [0.24], color=colors[0], clip_on=False,
            transform=ax.transAxes,
            )
    ax.set_ylim(0, 60)

    ax2 = ax.twinx()
    ax2.errorbar(
            res_avg.index, 100 * res_avg['gof'],
            yerr=100 * res_std['gof'],
            fmt='-o',
            lw=2, color=colors[1],
            )
    ax2.scatter(
            [1.17], [0.03], color=colors[1], clip_on=False,
            transform=ax.transAxes,
            )
    ax2.set_ylabel('Correct assignments [%]')
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig('../figures/scaling_newdata_size_TMS.svg')
    fig.savefig('../figures/scaling_newdata_size_TMS.png', dpi=600)

    plt.ion()
    plt.show()
