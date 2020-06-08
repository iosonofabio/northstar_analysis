# vim: fdm=indent
'''
author:     Fabio Zanini
date:       12/05/20
content:    Test scvi on a large dataset from Tabula Muris Senis
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

    resa = {'scvi': {}, 'northstar': {}, 'scmap': {}}

    print('Load northstar')
    with open('../data_for_figures/results_kidney_various_number_of_training_cell_types.pkl', 'rb') as f:
        res = pickle.load(f)

    res = res.loc[res['na'] >= 14]

    resa['northstar']['avg'] = res[['na', 'gof', 'time']].groupby('na').mean()
    resa['northstar']['std'] = res[['na', 'gof', 'time']].groupby('na').std()

    print('Load scvi')
    reps = [1, 2, 3, 4, 5]
    nat = [14, 15, 16, 17, 18]
    ress = []
    for na in nat:
        for rep in reps:
            fn = '../data_for_figures/scvi_kidney/result_na_{:}_rep_{:}.pkl'.format(
                na, rep,
                )
            with open(fn, 'rb') as f:
                resd = pickle.load(f)
            ress.append(resd)
    keys = ['na', 'rep', 'ntot', 'gof', 'time']
    resd = [{key: x[key] for key in keys} for x in ress]
    res = pd.DataFrame(resd)

    resa['scvi']['avg'] = res[['na', 'gof', 'time']].groupby('na').mean()
    resa['scvi']['std'] = res[['na', 'gof', 'time']].groupby('na').std()

    print('Load scmap')
    reps = [1, 2, 3, 4, 5]
    nat = [14, 15, 16, 17, 18]
    ress = []
    for na in nat:
        for rep in reps:
            fn = '../data_for_figures/scmap_kidney/result_na_{:}_rep_{:}.pkl'.format(
                na, rep,
                )
            with open(fn, 'rb') as f:
                resd = pickle.load(f)
            ress.append(resd)

    keys = ['na', 'rep', 'ntot', 'gof', 'time']
    resd = [{key: x[key] for key in keys} for x in ress]
    res = pd.DataFrame(resd)

    resa['scmap']['avg'] = res[['na', 'gof', 'time']].groupby('na').mean()
    resa['scmap']['std'] = res[['na', 'gof', 'time']].groupby('na').std()

    print('Plot results')
    colors = sns.color_palette('Dark2', n_colors=4)
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    ax = axs[0]
    ax.errorbar(
            resa['northstar']['avg'].index,
            resa['northstar']['avg']['time'],
            yerr=resa['northstar']['std']['time'],
            fmt='-o',
            lw=2, color=colors[1],
            )
    ax.errorbar(
            resa['scvi']['avg'].index,
            resa['scvi']['avg']['time'],
            yerr=resa['scvi']['std']['time'],
            fmt='-o',
            lw=2, color=colors[2],
            )
    ax.errorbar(
            resa['scmap']['avg'].index,
            resa['scmap']['avg']['time'],
            yerr=resa['scmap']['std']['time'],
            fmt='-o',
            lw=2, color=colors[3],
            )
    ax.set_xlabel('Number of cells types\nin atlas (out of 18)')
    ax.set_ylabel('Runtime [s]')
    ax.set_ylim(0, 720)

    ax2 = axs[1]
    ax2.errorbar(
            resa['northstar']['avg'].index,
            100 * resa['northstar']['avg']['gof'],
            yerr=100 * resa['northstar']['std']['gof'],
            fmt='-o',
            lw=2, color=colors[1],
            label='northstar',
            )
    ax2.errorbar(
            resa['scvi']['avg'].index,
            100 * resa['scvi']['avg']['gof'],
            yerr=100 * resa['scvi']['std']['gof'],
            fmt='-o',
            lw=2, color=colors[2],
            label='scVI',
            )
    ax2.errorbar(
            resa['scmap']['avg'].index,
            100 * resa['scmap']['avg']['gof'],
            yerr=100 * resa['scmap']['std']['gof'],
            fmt='-o',
            lw=2, color=colors[3],
            label='scmap',
            )
    ax2.set_ylabel('Correct assignments [%]')
    ax2.set_ylim(0, 100)
    ax2.legend(
        title='Algorithm:',
        loc='upper left',
        bbox_to_anchor=(1.01, 1.01), bbox_transform=ax2.transAxes,
        )
    fig.tight_layout()
    fig.savefig('../figures/northstar_vs_scvi_vs_scmap_tms_kidney_performance_various_cell_type_numbers.png')
    fig.savefig('../figures/northstar_vs_scvi_vs_scmap_tms_kidney_performance_various_cell_type_numbers.svg')



