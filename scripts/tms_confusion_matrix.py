# vim: fdm=indent
'''
author:     Fabio Zanini
date:       12/05/20
content:    Test northstar on a large dataset from Tabula Muris Senis
'''
import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/home/fabio/university/postdoc/northstar/build/lib')
import northstar


if __name__ == '__main__':

    tissue = 'Kidney'

    print('Plot confusion matrix')
    import pickle
    with open('../data_for_figures/northstar_kidney_14cell_types.pkl', 'rb') as f: 
        resd = pickle.load(f)
    identity = resd['identity']
    cats_map = resd['cats_map']

    # Select a run
    rows = list(resd['csts'])
    # Swap two cell types
    idx = rows.index('fibroblast')
    rows[idx], rows[idx - 1] = rows[idx - 1], rows[idx]
    tmp = set(identity['northstar'].unique())
    tmp2 = Counter(cats_map.values())
    cols = []
    col_names = []
    for ct in rows:
        if ct in tmp:
            cols.append(ct)
            col_names.append(ct)
            tmp.remove(ct)
        elif tmp2[ct] == 1:
            ict = [x for x, v in cats_map.items() if v == ct][0]
            cols.append(ict)
            col_names.append('{:} ({:})'.format(ict, ct))
            tmp.remove(ict)
    cols += list(tmp)
    col_names += list(tmp)

    cm = pd.DataFrame(
            np.zeros((len(rows), len(cols)), np.int64),
            index=rows, columns=cols,
            )
    for r in rows:
        for c in cols:
            cm.at[r, c] = ((identity['cell_ontology_class'] == r) & (identity['northstar'] == c)).sum()

    cs_sets = [
        ['T cell', 'NK cell', 'lymphocyte'],
        ['kidney capillary endothelial cell', 'fenestrated cell'],
        ['fibroblast', 'kidney mesangial cell'],
        ['kidney proximal convoluted tubule epithelial cell', 'kidney distal convoluted tubule epithelial cell'],
        ]
    fig, ax = plt.subplots(figsize=(10, 7.8))
    sfun = lambda x: 300. * x / cm.values.max()
    s = sfun(cm.values)
    color = np.full(s.shape, 'tomato', dtype=object)
    for i, r in enumerate(rows):
        for j, c in enumerate(col_names):
            if r in c:
                color[i, j] = 'mediumseagreen'
            else:
                seti = set([r])
                if '(' in c:
                    seti.add(c.split('(')[1][:-1])
                else:
                    seti.add(c)
                for cs_set in cs_sets:
                    if seti.issubset(set(cs_set)):
                        color[i, j] = 'orange'
                        break

    xm, ym = np.meshgrid(np.arange(len(cols)), np.arange(len(rows)))
    ax.scatter(ym.ravel(), xm.ravel(), s.ravel(), c=color.ravel(), zorder=10)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(rows, rotation=60, ha='right')
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(col_names)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(len(cols) - 0.5, -0.5)
    ax.grid(True)
    ax.set_xlabel('Original cell Type')
    ax.set_ylabel('Northstar assignment')
    n_sizes = [0, 5, 20, 50, 100]
    handles = []
    for ns in n_sizes:
        handles.append(ax.scatter([], [], s=sfun(ns), color='grey'))
    ax.legend(
            handles, list(map(str, n_sizes)), title='# cells:',
            bbox_to_anchor=(1.01, 1.01), bbox_transform=ax.transAxes,
            labelspacing=1.3,
            )

    fig.tight_layout()
    #fig.savefig('../figures/confusion_matrix_kidney_TabulaMurisSenis_14cell_types.png')
    #fig.savefig('../figures/confusion_matrix_kidney_TabulaMurisSenis_14cell_types.svg')

    plt.ion()
    plt.show()
