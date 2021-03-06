# vim: fdm=indent
'''
author:     Fabio Zanini
date:       03/03/20
content:    Explore a few more datasets available online.
'''
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import loompy

from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure you can import leidenalg with fixed nodes
sys.path.insert(0, os.path.abspath('../..')+'/leidenalg/build/lib.linux-x86_64-3.8') # Fabio's laptop


# Make sure you can import northstar by adding its parent folder to your PYTHONPATH aka sys.path
sys.path.insert(0, os.path.abspath('../..')+'/northstar') # Fabio's laptop
import northstar


def ingest_melanoma_data():
    fdn = '../data/HCA/melanoma/E-EHCA-2-quantification-raw-files/'
    if False:
        from scipy.io import mmread
        cells = pd.read_csv(fdn+'E-EHCA-2.aggregated_filtered_counts.mtx_cols', sep='\t', squeeze=True, header=None).values
        geneIds = pd.read_csv(fdn+'E-EHCA-2.aggregated_filtered_counts.mtx_rows', sep='\t', squeeze=True, header=None).values[:, 0]
        matrix = mmread(fdn+'E-EHCA-2.aggregated_filtered_counts.mtx').astype(np.float32).todense()

        meta = pd.read_csv(fdn+'ExpDesign-E-EHCA-2.tsv', sep='\t', index_col=0)
        meta = meta.loc[meta.index.isin(cells)]
        meta = meta.loc[cells]

        print('Convert gene Ids into gene names')
        conv = pd.read_csv(fdn+'mart_export.txt', sep='\t', index_col=0)

        print('Exclude genes that have no gene name')
        ind = pd.Index(geneIds).isin(conv.index)
        geneIds = geneIds[ind]
        matrix = matrix[ind]

        conv = conv.loc[geneIds]

        from collections import Counter
        gene_d = Counter()
        for gid, gname in conv['Gene name'].items():
            gene_d[gname] += 1
        good_genes = [g for g, c in gene_d.items() if c == 1]
        ind = conv['Gene name'].isin(good_genes)
        conv = conv.loc[ind]

        ind = pd.Index(geneIds).isin(conv.index)
        geneIds = geneIds[ind]
        matrix = matrix[ind]

        genes = conv['Gene name'].values

        col_attrs = {'CellID': cells}
        col_attrs['Cell_type'] = meta['Sample Characteristic[inferred cell type]'].values
        col_attrs['Individual'] = meta['Sample Characteristic[individual]'].values
        col_attrs['Sampling site'] = meta['Sample Characteristic[sampling site]'].values
        col_attrs['Tissue'] = meta['Sample Characteristic[organism part]'].values
        col_attrs['FACS markers'] = meta['Sample Characteristic[facs marker]'].values
        loompy.create(
            fdn+'melanoma.loom',
            layers={'': matrix},
            col_attrs=col_attrs,
            row_attrs={'GeneName': genes, 'EnsemblID': geneIds},
            )

    with loompy.connect(fdn+'melanoma.loom') as dsl:
        cells = dsl.ca['CellID']
        genes = dsl.ra['GeneName']
        counts = pd.DataFrame(
                dsl[:, :],
                index=genes,
                columns=cells,
                )
        meta = pd.DataFrame([], index=cells)
        for col in dsl.ca.keys():
            if col == 'CellID':
                continue
            meta[col] = dsl.ca[col]

        # Exclude cells that have no inferred type
        ind = meta['Cell_type'] != 'not available'
        meta = meta.loc[ind]
        counts = counts.loc[:, ind]

        mela = {
            'counts': counts,
            'meta': meta,
            }

    return mela


def define_accuracy(true_types, cell_types):
    match = np.zeros(len(true_types), bool)

    tt = pd.Index(true_types)
    pt = pd.Index(cell_types)

    cats = ['T cell', 'B cell', 'natural killer cell', 'monocyte']
    for cat in cats:
        match |= tt.str.contains(cat) & pt.str.contains(cat)

    cat_pairs = [('dendritic cell', 'progenitor cell')]
    for cat1, cat2 in cat_pairs:
        match |= tt.str.contains(cat1) & pt.str.contains(cat2)

    ind_known = np.ones(len(true_types), bool)
    cats = ['fibroblast', 'endothelial']
    for cat in cats:
        ind_known &= ~tt.str.contains(cat)

    match = match[ind_known].mean()

    return match


def subsample_dataset(cancer_data, n):
    if (n == 'all') or (n == cancer_data['meta'].shape[0]):
        return cancer_data
    ind = np.arange(cancer_data['meta'].shape[0])
    np.random.shuffle(ind)
    ind = ind[:n]
    tmp = {
        'counts': cancer_data['counts'].iloc[:, ind],
        'meta': cancer_data['meta'].iloc[ind],
        }
    return tmp


def test_conditions(all_names, conditions, atlas_sub, cancer_data, repeats=1):
    import time
    results = []
    ncomb = len(conditions)
    for ic, comb in enumerate(conditions):
        print('{:} / {:}: {:}'.format(ic + 1, ncomb, comb))
        for ir in range(repeats):
            kwargs = dict(zip(all_names, comb))
            if 'n' in kwargs:
                n = kwargs.pop('n')
                tmp = subsample_dataset(cancer_data, n)
            else:
                tmp = cancer_data
            t0 = time.time()
            no = northstar.Subsample(
                atlas=atlas_sub,
                **kwargs,
                )
            cell_types = no.fit_transform(tmp['counts'])
            t1 = time.time()
            acc = define_accuracy(tmp['meta']['Cell_type'].values, cell_types)
            kwargs['accuracy'] = acc
            kwargs['runtime'] = t1 - t0
            kwargs['repeat'] = ir + 1
            kwargs['ncells'] = tmp['meta'].shape[0]
            results.append(kwargs)

    return pd.DataFrame(results)


def get_combinations(params_dict):
    import itertools as it
    all_names = sorted(params_dict)
    combinations = list(it.product(*(params_dict[x] for x in all_names)))
    return (all_names, combinations)


def chunker(seq, n_chunks, constants=None):
    groups = [{'conditions': []} for i in range(n_chunks)]
    i = 0
    for x in seq:
        groups[i]['conditions'].append(x)
        i = (i + 1) % n_chunks
    for g in groups:
        g['constants'] = constants
    return groups


def worker_scan_function(group):
    conditions = group['conditions']
    all_names, atlas_sub, cancer_data, repeats = group['constants']
    return test_conditions(
            all_names, conditions, atlas_sub, cancer_data, repeats=repeats)


def test_conditions_parallel(
        all_names, conditions, atlas_sub, cancer_data, repeats=1, n_procs=None):
    import multiprocessing
    from multiprocessing import Pool

    if n_procs is None:
        n_cpus = multiprocessing.cpu_count()
        n_procs = min(n_cpus - 1, 40)

    groups = chunker(
            conditions, n_procs,
            constants=[all_names, atlas_sub, cancer_data, repeats])

    print('Starting pool of {:} CPUs'.format(n_procs))
    p = Pool(n_procs)
    results = p.map(worker_scan_function, groups)
    results = pd.concat(results, axis=0)

    return results


def test_combinations(params_dict, atlas_sub, cancer_data, repeats=1, n_procs=None):
    (all_names, combinations) = get_combinations(params_dict)

    if n_procs != 1:
        df = test_conditions_parallel(
            all_names, combinations, atlas_sub, cancer_data, repeats=repeats, n_procs=n_procs)
    else:
        df = test_conditions(
            all_names, combinations, atlas_sub, cancer_data, repeats=repeats)
    return df


if __name__ == '__main__':


    print('Load cancer data')
    cancer_data = ingest_melanoma_data()

    af = northstar.fetch_atlas.AtlasFetcher()
    atlas_sub = af.fetch_atlas('TabulaMuris_2018_marrow', kind='subsample')


    sys.exit()

    print('Classify and cluster cells with default parameters')
    no = northstar.Subsample(
        atlas=atlas_sub,
        )
    cell_types = no.fit_transform(cancer_data['counts'])
    acc = define_accuracy(cancer_data['meta']['Cell_type'].values, cell_types)
    print(acc)

    if True:
        print('Measure runtime as a function of cell numbers')
        params_dict = dict(
            n_features_overdispersed=[200, 600, 1200],
            n=[100, 600, 1500, 3000, 'all'],
            )
        df = test_combinations(params_dict, atlas_sub, cancer_data, repeats=3)
        results = df[['n_features_overdispersed', 'ncells', 'runtime']].groupby(['n_features_overdispersed', 'ncells']).mean()
        results['runtime_std'] = df[['n_features_overdispersed', 'ncells', 'runtime']].groupby(['n_features_overdispersed', 'ncells']).std()['runtime']

        fig, ax = plt.subplots(figsize=(5, 2.5))
        cmap = dict(zip(params_dict['n_features_overdispersed'], sns.color_palette(n_colors=10)))
        for nfea, datum in results.groupby('n_features_overdispersed'):
            x = datum.index.get_level_values('ncells')
            y = datum['runtime']
            dy = datum['runtime_std']
            ax.plot(x, y, lw=2, label=str(nfea), color=cmap[nfea])
            ax.fill_between(
                    x, y - dy, y + dy,
                    lw=1, color=cmap[nfea], alpha=0.2,
                    )
        ax.grid(True)
        ax.set_xlabel('Number of cells')
        ax.set_ylabel('Runtime [s]')
        ax.legend(
                loc='upper left',
                title='Number of\noverdisperded\nfeatures:',
                bbox_to_anchor=(1.01, 1.01),
                bbox_transform=ax.transAxes,
                )
        fig.tight_layout()
        fig.savefig('../figures/melanoma_ncells_vs_runtime.svg')
        fig.savefig('../figures/melanoma_ncells_vs_runtime.png')


    if False:
        ct1 = np.unique(cell_types)
        ct2 = np.unique(cancer_data['meta']['Cell_type'].values)
        outp = cancer_data['meta'][['Cell_type']].copy()
        outp['Predicted_type'] = cell_types
        outp['c'] = 1

        print('Tsne')
        vs = no.embed()
        ctall = np.union1d(ct1, ct2)
        cmap = dict(zip(ctall, sns.color_palette('husl', n_colors=len(ctall))))
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        axs = axs.ravel()
        for i in range(2):
            ax = axs[i]
            ct = outp.values[:, i]
            c = [cmap[x] for x in ct]
            vsi = vs.iloc[-len(c):]
            ax.scatter(vsi['Dimension 1'], vsi['Dimension 2'], s=15, c=c, alpha=0.5)
            ax.set_title(outp.columns[i])
        import matplotlib.lines as mlines
        labels = sorted(cmap)
        handles = []
        for key in labels:
            h = mlines.Line2D(
                [], [], color=cmap[key], marker='o', lw=0,
                markersize=5,
                )
            handles.append(h)
        axs[-1].legend(
                handles, labels,
                loc='upper left',
                bbox_to_anchor=(1.01, 1.01),
                bbox_transform=ax.transAxes,
                ncol=2,
                fontsize=6,
                )
        fig.tight_layout()

        print('Figure out where they go')
        counts = outp.groupby(['Cell_type', 'Predicted_type']).sum()['c'].unstack(fill_value=0)

        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, leaves_list

        pd1 = pdist(counts.values)
        pd2 = pdist(counts.values.T)
        Z1 = linkage(pd1)
        Z2 = linkage(pd2)
        ind1 = leaves_list(Z1)
        ind2 = leaves_list(Z2)
        counts_plot = counts.iloc[ind1].T.iloc[ind2].T

        fig, ax = plt.subplots(figsize=(13, 9))
        sns.heatmap(counts_plot, ax=ax)
        fig.tight_layout()

    if False:
        print('Vice versa, modify only the two key parameters and see')
        if False:
            params_dict3 = dict(
                n_features_per_cell_type=[15],
                n_pcs=[15],
                n_neighbors=[20],
                distance_metric=['correlation'],
                threshold_neighborhood=[0.8],
                resolution_parameter=[1e-5, 3e-5, 6e-5, 0.0001, 0.0002, 0.0003, 0.0006, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05],
                n_features_overdispersed=[50, 75, 100, 150, 200, 250, 300, 600, 1000, 1500, 2000],
                )
            df3 = test_combinations(params_dict3, atlas_sub, cancer_data, repeats=5)
            fdn = '../data/HCA/melanoma/northstar_predictions/'
            df3.to_csv(fdn+'results_grid3.tsv', sep='\t', index=True)
        df3 = pd.read_csv(fdn+'results_grid3.tsv', sep='\t', index_col=0)
        accs = (df3[['n_features_overdispersed', 'resolution_parameter', 'accuracy']]
                .groupby(['n_features_overdispersed', 'resolution_parameter'])
                .mean()
                ['accuracy']
                .unstack('resolution_parameter'))

        print('Plot heatmap for those two params only')
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            accs,
            ax=ax,
            vmin=0.1,
            vmax=1,
            )
        for tk in ax.get_yticklabels():
            tk.set_rotation(0)
        fig.tight_layout()

    if False:
        print('Plot runtime versus accuracy')
        data = df3.loc[(df3['resolution_parameter'] <= 0.008) & (df3['resolution_parameter'] >= 0.0008), ['accuracy', 'runtime']]
        data['accuracy'] *= 100
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.kdeplot(data['accuracy'], data['runtime'], ax=ax, alpha=0.8)
        ax.scatter(data['accuracy'], data['runtime'], alpha=0.1, s=10, zorder=15, color='k')
        ax.grid(True)
        ax.set_xlabel('% correct assignments')
        ax.set_ylabel('Runtime [s]')
        fig.tight_layout()

    if False:
        print('Print runtime for accurate runs')
        data = df3.loc[df3['accuracy'] > 0.8, 'runtime']
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.kdeplot(data, ax=ax, alpha=0.8, lw=2)
        ax.legend().remove()
        ax.grid(True)
        ax.set_xlabel('Runtime [s]')
        ax.set_ylabel('Density')
        fig.tight_layout()
        fig.savefig('../figures/melanoma_runtime_goodruns.svg')
        fig.savefig('../figures/melanoma_runtime_goodruns.png')

    plt.ion()
    plt.show()
