# vim: fdm=indent
'''
author:     Fabio Zanini
date:       11/05/20
content:    Fig. 1E on number of new cell types in increasingly large atlases.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    fn_tm = '../data/TabulaMuris/annotations_facs.csv'
    df_tm = pd.read_csv(
            fn_tm,
            index_col='cell',
            low_memory=False)['cell_ontology_class']

    fn_tms = '../data/TabulaMurisSenis/TabulaMurisSenis_droplet_cell_types_tissue_age.tsv'
    df_tms = pd.read_csv(
            fn_tms,
            sep='\t',
            index_col=0,
            low_memory=False)['cell_ontology_class']

    ns = [
        200, 300, 500,
        1000, 2000, 3000, 5000,
        10000, 20000, 30000, 50000,
        100000, 200000,
        ]

    nct = {}
    for name, df in zip(['TabulaMuris', 'TabulaMurisSenis'], [df_tm, df_tms]):
        for n in ns:
            if n > len(df):
                continue
            tmp = []
            for i in range(10):
                nc = np.random.choice(np.arange(len(df)), size=n, replace=False)
                nc = (df[nc].value_counts() >= 20).sum()
                tmp.append(nc)
            nct[(name, n)] = np.mean(tmp)
        nct[(name, len(df))] = (df.value_counts() >= 20).sum()
    nct = pd.Series(nct).unstack(0)

    print('Plot scaling')
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharex=True)
    ax = axs[0]
    ytm = nct['TabulaMuris'].dropna()
    xtm = ytm.index

    ytms = nct['TabulaMurisSenis'].dropna()
    xtms = ytms.index

    ax.set_title('Scaling of tabula muris\n(>= 20 cells/type)')
    ax.plot(xtm, ytm, 'o-', lw=2, color='deeppink', alpha=0.7, label='3 mo FACS')
    ax.plot(xtms, ytms, 'o-', lw=2, color='grey', alpha=0.7, label='3/18/24 mo 10X')

    # final scaling is about 8 / decade
    scal = 5
    yp = ytms.iloc[-1] + scal
    xp = xtms[-1] * 10
    ax.plot([xtms[-1], xp], [ytms.iloc[-1], yp], '--', lw=2, color='grey')

    ax.grid(True)
    ax.set_xlim(1000, 1e6)
    ax.set_xscale('log')
    ax.set_xlabel('N. atlas cells')
    ax.set_ylabel('N. cell types')
    ax.legend(loc='lower right', title='Atlas:')

    ax = axs[1]

    # Northstar uses one float32 = 4 bytes per gene per cell. The number of
    # genes scales almost with the number of cell types by 30/cell type + 500
    # constant overdispersed (default parameters). However to compute the peak
    # memory requirements, let's use all genes. In MBytes:
    y1 = 4 * 20000 * ytms / 1024 / 1024
    y2 = 4 * 20000 * ytms * 20 / 1024 / 1024
    y3 = 4 * 20000 * xtms / 1024 / 1024

    # After feature selection
    #y1 = 4 * (500 + 30 * ytms) * ytms / 1024 / 1024
    #y2 = 4 * (500 + 30 * ytms) * ytms * 20 / 1024 / 1024
    #y3 = 4 * (500 + 30 * ytms) * xtms / 1024 / 1024

    # Per gene
    #y1 = ytms
    #y2 = 20 * ytms
    #y3 = xtms
    ys = [y1, y2, y3]
    colors = sns.color_palette('muted', n_colors=3)
    labels = ['Average', 'Subsample', 'Full atlas']
    markers = ['o', '^', 's']
    for i in [2, 1, 0]:
        ax.plot(xtms, ys[i], '-'+markers[i], color=colors[i], label=labels[i])

        # final scaling
        nc_proj = ytms.iloc[-1] + scal
        if i == 0:
            yp = 4 * 20000 * nc_proj / 1024 / 1024
        elif i == 1:
            yp = 4 * 20000 * nc_proj * 20 / 1024 / 1024
        elif i == 2:
            yp = 4 * 20000 * xtms[-1] * 10 / 1024 / 1024
        xp = xtms[-1] * 10
        ax.plot([xtms[-1], xp], [ys[i].values[-1], yp], '--', lw=2, color=colors[i])

    ax.grid(True)
    ax.set_xlabel('N. atlas cells')
    ax.set_xlim(1000, 10 * n)
    #ax.set_ylim(3, 5e5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Atlas memory [MBytes]')
    ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.01, 1.01), bbox_transform=ax.transAxes)
    ax.set_title('Scaling of northstar')

    fig.tight_layout()
    fig.savefig('../figures/scaling_tabula_muris.png')
    fig.savefig('../figures/scaling_tabula_muris.svg')

    plt.ion()
    plt.show()
