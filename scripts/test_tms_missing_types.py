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
    adata = anndata.read_h5ad('../data/TabulaMurisSenis/{:}_droplet.h5ad'.format(tissue))

    asub = northstar.subsample_atlas(
            adata,
            cell_type_column='cell_ontology_class',
            n_cells=20,
            )
    asub.obs['CellType'] = asub.obs['cell_ontology_class']

    print('Include different subset of cell types in atlas')
    csts = adata.obs['cell_ontology_class'].value_counts()
    nat = [12, 13, 14, 15, 16, 17, 18]
    nct = 100
    reps = 5
    res_pars = [0.003]
    res = []
    plot_embed = False
    for na in nat:
        print('Restricting atlas to the most common {:} cell types'.format(na))
        csti = csts.index[:na]
        idx = asub.obs['cell_ontology_class'].isin(csti).values.nonzero()[0]
        asubr = asub[idx]

        for res_par in res_pars:
            for rep in range(reps):
                print('Subsample with {:} cells per type'.format(nct))
                asub2 = northstar.subsample_atlas(
                        adata,
                        cell_type_column='cell_ontology_class',
                        n_cells=nct,
                        )
                ntot = asub2.X.shape[0]

                print('Run northstar')
                import time
                t0 = time.time()
                ns = northstar.Subsample(
                    asubr,
                    # NOTE: seems like this has to go down with more cell types
                    resolution_parameter=0.005,
                    )
                ns.fit(asub2)
                t1 = time.time()
                t = t1 - t0

                ct_orig = ns.new_data.obs['cell_ontology_class'].astype(str)
                identity = ct_orig.to_frame()
                identity['northstar'] = ns.membership
                identity['correct'] = (ct_orig == ns.membership)

                atlas_cts = asubr.obs['CellType'].unique()
                new_cats = list(set(ns.membership) - set(csti))
                cats_map = {}
                for nc in new_cats:
                    idx = ns.membership == nc
                    ctx = ct_orig[idx]
                    ct_most = ctx.value_counts().index[0]
                    cats_map[nc] = ct_most
                    if ct_most not in atlas_cts:
                        idxt = idx & (ct_orig == ct_most)
                        identity.loc[idxt, 'correct'] = True
                    else:
                        # FIXME: not really fine, but ok for now
                        pass
                gof = identity['correct'].mean()

                res.append({
                    'na': na,
                    'time': t,
                    'gof': gof,
                    'ntot': ntot,
                    'rep': rep,
                    'res_par': res_par,
                    'identity': identity,
                    'cats_map': cats_map,
                    })

                if plot_embed:
                    vs = ns.embed(method='umap')
                    fig, axs = plt.subplots(1, 2, figsize=(9, 8), sharex=True, sharey=True)
                    vs['orig'] = pd.concat([
                        asubr.obs['cell_ontology_class'],
                        asub2.obs['cell_ontology_class']])
                    vs['northstar'] = pd.concat([
                        asubr.obs['cell_ontology_class'],
                        pd.Series(ns.membership, index=asub2.obs_names)])
                    cou = csts.index
                    cmau = dict(zip(cou, sns.color_palette('husl', n_colors=len(cou))))
                    tmp = set(vs['northstar'].unique())
                    from collections import Counter
                    tmp2 = Counter(cats_map.values())
                    tmp_add = []
                    con = []
                    for ct in cou:
                        if ct in tmp:
                            con.append(ct)
                            tmp.remove(ct)
                        elif (ct in tmp2) and (tmp2[ct] == 1):
                            ict = [x for x, v in cats_map.items() if v == ct][0]
                            tmp_add.append(ict)
                            con.append(ict)
                            tmp.remove(ict)
                    con += list(tmp)
                    cman = {}
                    nm = 0
                    for cn in con:
                        if cn in cmau:
                            cman[cn] = cmau[cn]
                        elif cn in tmp_add:
                            cman[cn] = cmau[cats_map[cn]]
                        else:
                            cman[cn] = tuple(np.mod(np.ones(3) * 0.111 * nm, 0.7))
                            nm += 1

                    cu = [cmau[x] for x in vs['orig'].values]
                    axs[0].scatter(
                            vs.values[:ns.n_atlas, 0], vs.values[:ns.n_atlas, 1],
                            s=70, marker='*', alpha=0.6, c=cu[:ns.n_atlas])
                    axs[0].scatter(
                            vs.values[ns.n_atlas:, 0], vs.values[ns.n_atlas:, 1],
                            s=30, marker='o', alpha=0.4, c=cu[ns.n_atlas:])
                    handles = []
                    for c in cou:
                        handles.append(axs[0].scatter([], [], color=cmau[c]))
                    axs[0].legend(
                            handles, cou, loc='upper center', fontsize=8,
                            bbox_to_anchor=(0.5, -0.1), bbox_transform=axs[0].transAxes)

                    cn = [cman[x] for x in vs['northstar'].values]
                    axs[1].scatter(
                            vs.values[:ns.n_atlas, 0], vs.values[:ns.n_atlas, 1],
                            s=70, marker='*', alpha=0.6, c=cn[:ns.n_atlas])
                    axs[1].scatter(
                            vs.values[ns.n_atlas:, 0], vs.values[ns.n_atlas:, 1],
                            s=30, marker='o', alpha=0.4, c=cn[ns.n_atlas:])
                    handles = []
                    for c in con:
                        handles.append(axs[1].scatter([], [], color=cman[c]))
                    axs[1].legend(
                            handles, con, loc='upper center', fontsize=8,
                            bbox_to_anchor=(0.5, -0.1), bbox_transform=axs[1].transAxes)
                    fig.suptitle('GOF: {:.0%}'.format(gof))
                    fig.tight_layout(rect=(0, 0, 1, 0.95))


    res = pd.DataFrame(res)

    # Save a specific run to file for the figure
    #resd = res.iloc[0].to_dict()
    #resd['csts'] = csts.index
    #with open('../data_for_figures/northstar_kidney_14cell_types.pkl', 'wb') as f: 
    #    pickle.dump(res, f)

    res_avg = res[['na', 'time', 'gof', 'ntot', 'rep']].groupby(['na', 'ntot']).mean()[['time', 'gof']]
    res_std = res[['na', 'time', 'gof', 'ntot', 'rep']].groupby(['na', 'ntot']).std()[['time', 'gof']]

    print('Plot results')
    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(figsize=(4, 3))
    for ntot in np.sort(res['ntot'].unique()):
        resi = res.loc[res['ntot'] == ntot]
        res_avg = resi[['na', 'time', 'gof', 'ntot', 'rep']].groupby(['na']).mean()[['time', 'gof']]
        res_std = resi[['na', 'time', 'gof', 'ntot', 'rep']].groupby(['na']).std()[['time', 'gof']]
        ax.errorbar(
                res_avg.index, res_avg['time'],
                yerr=res_std['time'],
                fmt='-o',
                lw=2, color=colors[0],
                )
        ax.set_xlabel('Number of cells types\nin atlas (out of 18)')
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
    fig.savefig('../figures/tms_kidney_performance_various_cell_type_numbers.png')
    fig.savefig('../figures/tms_kidney_performance_various_cell_type_numbers.svg')

    plt.ion()
    plt.show()
