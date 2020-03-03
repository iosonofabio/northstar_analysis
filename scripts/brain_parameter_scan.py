# vim: fdm=indent
'''
author:     Fabio Zanini
date:       02/03/20
content:    Scan parameters for GBM analysis.
'''
import os
import sys
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


def ingest_gbm_data():
    # GBM data folder
    savedir_init = os.path.abspath('..')+'/data/'
    gbmdir = savedir_init+'GBM_data_and_metadata/'
    gbm_loomfn = gbmdir+'GBM_data.loom'

    # only download if this is the first time running this notebook:
    if not os.path.exists(gbmdir):
        print('Downloading GBM raw data...')

        #darmanis glioblastoma (GBM) dataset (to annotate based on brain atlas)
        url = 'http://storage.googleapis.com/gbmseqrawdata/rawData.zip'

        from io import BytesIO
        from zipfile import ZipFile
        from urllib.request import urlopen

        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))
        files = zipfile.namelist()
        print('Avalable glioblastoma files:')
        for f in files:
            print(f)

        print('Extracting GBM data from ZIP archive...')
        for item in [1,2,3,4]:
            zipfile.extract(files[item],path=savedir_init)

        print('Reading GBM data from CSV files')
        GBM_count_path = savedir_init+files[4]
        GBM_counttable = pd.read_csv(GBM_count_path,sep=' ',index_col=0)
        GBM_meta = pd.read_csv(savedir_init+files[2],sep=' ',index_col=0)
        GBMtsne = pd.read_csv(savedir_init+files[1],sep=' ',index_col=0)
        # add annotation from separate file
        GBMmeta2= pd.read_excel(savedir_init+files[3],index_col=0,header=19,sep=' ')[:3589] # read from xls file

        print('Standardize annotations, add original t-SNE coordinates, and other minor manipulations...')
        GBM_meta = GBM_meta.join(GBMtsne)
        GBM_meta = GBM_meta.join(GBMmeta2['characteristics: cell type'])
        GBM_meta.rename(
                index=str,
                columns={'characteristics: cell type': 'Cell_type2'},
                inplace=True)

        # correct spelling and systematize cell types
        name_lut = {
            'Astocyte': 'Astrocyte',
            'microglia': 'Microglia',
            'Vascular': 'Endothelial',
            'Astrocytes': 'Astrocyte',
            'Oligodendrocyte': 'Oligodendrocyte',
            'Neurons': 'Neuron',
            'Neoplastic': 'Neoplastic',
            'Immune cell': 'Immune cell',
            'OPC': 'OPC',
            'Neuron': 'Neuron',
        }
        GBM_meta['Cell_type'] = GBM_meta['Cell_type2'].map(name_lut)

    elif os.path.isfile(gbm_loomfn):
        print('Reading GBM data from loom file')
        with loompy.connect(gbm_loomfn) as dsl:
            cells = dsl.ca['CellID']
            genes = dsl.ra['GeneName']
            GBM_meta = pd.DataFrame([], index=cells)
            for col in dsl.ca:
                GBM_meta[col] = dsl.ca[col]
            GBM_counttable = pd.DataFrame(
                data=dsl[:, :],
                index=genes,
                columns=cells,
                )

    else:
        print('Reading GBM data from CSV files')
        GBMtsne = pd.read_csv(gbmdir+'GBM_TSNE.csv',sep=' ',index_col=0)
        GBM_meta = pd.read_csv(gbmdir+'GBM_metadata.csv',sep=' ',index_col=0)
        GBMmeta2= pd.read_excel(gbmdir+'GEO_upload/spyros.darmanis_metadata_GBM.xls',index_col=0,header=19,sep=' ')[:3589]
        GBM_counttable = pd.read_csv(gbmdir+'GBM_raw_gene_counts.csv',sep=' ',index_col=0)

        print('Standardize annotations, add original t-SNE coordinates, and other minor manipulations...')
        GBM_meta = GBM_meta.join(GBMtsne)
        GBM_meta = GBM_meta.join(GBMmeta2['characteristics: cell type'])
        GBM_meta.rename(index=str,columns={'characteristics: cell type':'Cell_type2'},inplace=True)

        # correct spelling and systematize cell types
        name_lut = {
            'Astocyte': 'Astrocyte',
            'microglia': 'Microglia',
            'Vascular': 'Endothelial',
            'Astrocytes': 'Astrocyte',
            'Oligodendrocyte': 'Oligodendrocyte',
            'Neurons': 'Neuron',
            'Neoplastic': 'Neoplastic',
            'Immune cell': 'Immune cell',
            'OPC': 'OPC',
            'Neuron':'Neuron',
        }
        GBM_meta['Cell_type'] = GBM_meta['Cell_type2'].map(name_lut)

        matrix = GBM_counttable.values.astype(np.float32)
        col_attrs = {'CellID': GBM_counttable.columns.values}
        for col in GBM_meta:
            col_attrs[col] = GBM_meta[col].values
        row_attrs = {'GeneName': GBM_counttable.index.values}
        loompy.create(
            gbm_loomfn,
            layers={'': matrix},
            row_attrs=row_attrs,
            col_attrs=col_attrs,
            )

    return {
        'counts': GBM_counttable,
        'meta': GBM_meta,
        }


def define_accuracy(true_types, cell_types):
    ind_known = ~pd.Index(true_types).isin(['Immune cell', 'Neoplastic'])

    # TODO: implement better classification
    match = (true_types == cell_types)[ind_known].mean()

    return match


def test_conditions(all_names, conditions, atlas_sub, gbm, repeats=1):
    import time
    results = []
    ncomb = len(conditions)
    for ic, comb in enumerate(conditions):
        print('{:} / {:}: {:}'.format(ic + 1, ncomb, comb))
        for ir in range(repeats):
            kwargs = dict(zip(all_names, comb))
            t0 = time.time()
            no = northstar.Subsample(
                atlas=atlas_sub,
                **kwargs,
                )
            cell_types = no.fit_transform(gbm['counts'])
            t1 = time.time()
            acc = define_accuracy(gbm['meta']['Cell_type'].values, cell_types)
            kwargs['accuracy'] = acc
            kwargs['runtime'] = t1 - t0
            kwargs['repeat'] = ir + 1
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
    all_names, atlas_sub, gbm, repeats = group['constants']
    return test_conditions(
            all_names, conditions, atlas_sub, gbm, repeats=repeats)


def test_conditions_parallel(
        all_names, conditions, atlas_sub, gbm, repeats=1, n_procs=None):
    import multiprocessing
    from multiprocessing import Pool

    if n_procs is None:
        n_cpus = multiprocessing.cpu_count()
        n_procs = min(n_cpus - 1, 40)

    groups = chunker(
            conditions, n_procs,
            constants=[all_names, atlas_sub, gbm, repeats])

    print('Starting pool of {:} CPUs'.format(n_procs))
    p = Pool(n_procs)
    results = p.map(worker_scan_function, groups)
    results = pd.concat(results, axis=0)

    return results


def test_combinations(params_dict, atlas_sub, gbm, repeats=1, n_procs=None):
    (all_names, combinations) = get_combinations(params_dict)

    if n_procs != 1:
        df = test_conditions_parallel(
            all_names, combinations, atlas_sub, gbm, repeats=repeats, n_procs=n_procs)
    else:
        df = test_conditions(
            all_names, combinations, atlas_sub, gbm, repeats=repeats)
    return df



if __name__ == '__main__':

    af = northstar.fetch_atlas.AtlasFetcher()
    atlas_sub = af.fetch_atlas('Darmanis_2015_nofetal', kind='subsample')
    print('Rename a few cell types in the atlas')
    atlas_sub['cell_types'] = atlas_sub['cell_types'].map({
        'Oligodendrocyte': 'Oligodendrocyte',
        'Vascular': 'Endothelial',
        'Astrocyte': 'Astrocyte',
        'Neuron': 'Neuron',
        'OPC': 'OPC',
        'microglia': 'Immune cell'},
        )

    gbm = ingest_gbm_data()

    print('Classify and cluster cells with default parameters')
    no = northstar.Subsample(
        atlas=atlas_sub,
        )
    cell_types = no.fit_transform(gbm['counts'])
    acc = define_accuracy(gbm['meta']['Cell_type'].values, cell_types)
    print(acc)

    if False:
        print('Scan broad parameter space')
        params_dict = dict(
            n_features_per_cell_type=[10, 30, 50],
            n_features_overdispersed=[50, 300, 1000],
            n_pcs=[10, 20, 40],
            n_neighbors=[10, 20, 50],
            distance_metric=['correlation'],
            threshold_neighborhood=[0.8],
            resolution_parameter=[0.0001, 0.001, 0.01],
            )

        print('Perform parameter scan with parallelism')
        df = test_combinations(params_dict, atlas_sub, gbm, repeat=1, n_procs=None)

        #print('Classify and cluster cells with parameter scan')
        #df_np = test_combinations(params_dict, atlas_sub, gbm, repeats=1, n_procs=1)

        savedir_init = os.path.abspath('..')+'/data/'
        gbmdir = savedir_init+'GBM_data_and_metadata/'
        df.to_csv(gbmdir+'results_grid.tsv', sep='\t', index=True)

        print('Plot results')
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.hist(df['accuracy'], bins=np.linspace(0, 1, 21))
        ax.set_xlabel('Accuracy')
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
        ax.grid(False)
        ax.set_title('All parameter combinations')
        fig.tight_layout()

        print('Identify enrichments')
        threshold = 0.8
        for col in params_dict.keys():
            vc1 = df.loc[df['accuracy'] < threshold, col].value_counts()
            vc2 = df.loc[df['accuracy'] >= threshold, col].value_counts()

            ratio = 1.0 * vc2 / (vc1 + vc2)
            print(col)
            print(ratio)

        from scipy.stats import spearmanr, pearsonr
        print('Spearman rho btw runtime and accuracy: {:}'.format(spearmanr(df['accuracy'].values, df['runtime'].values)))
        print('Pearson r btw runtime and accuracy: {:}'.format(pearsonr(df['accuracy'].values, df['runtime'].values)))

        print('Print the top parameters')
        print(df.nlargest(10, 'accuracy'))

    if False:
        print('Scan parameter subspace of good rough sketches')
        params_dict2 = dict(
            n_features_per_cell_type=[5, 10, 20, 30, 50, 80],
            n_features_overdispersed=[300],
            n_pcs=[10, 20, 30, 40],
            n_neighbors=[5, 10, 20, 35, 50],
            distance_metric=['correlation'],
            threshold_neighborhood=[0.8],
            resolution_parameter=[0.0001],
            )
        df2 = test_combinations(params_dict2, atlas_sub, gbm)

        print('Print the top parameters')
        print(df2.nlargest(10, 'accuracy'))

        from scipy.stats import spearmanr, pearsonr
        print('Spearman rho btw runtime and accuracy: {:}'.format(spearmanr(df2['accuracy'].values, df2['runtime'].values)))
        print('Pearson r btw runtime and accuracy: {:}'.format(pearsonr(df2['accuracy'].values, df2['runtime'].values)))

        print('Plot results for subscan')
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.hist(df2['accuracy'], bins=np.linspace(0, 1, 51))
        ax.set_xlabel('Accuracy')
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
        ax.grid(False)
        for tk in ax.get_xticklabels():
            tk.set_rotation(90)
        ax.set_title('Subscan')
        fig.tight_layout()

    print('Vice versa, modify only the two key parameters and see')
    params_dict3 = dict(
        n_features_per_cell_type=[15],
        n_pcs=[15],
        n_neighbors=[20],
        distance_metric=['correlation'],
        threshold_neighborhood=[0.8],
        resolution_parameter=[1e-5, 2e-5, 3e-5, 6e-5, 8e-5, 0.0001, 0.0002, 0.0003, 0.0006, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05],
        n_features_overdispersed=[50, 75, 100, 150, 200, 250, 300, 400, 600, 800, 1000],
        )
    df3 = test_combinations(params_dict3, atlas_sub, gbm, repeats=2)
    accs = df3[['n_features_overdispersed', 'resolution_parameter', 'accuracy']].groupby(['n_features_overdispersed', 'resolution_parameter']).mean()['accuracy'].unstack('resolution_parameter')

    print('Plot heatmap for those two params only')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        np.maximum(accs, 0.9),
        ax=ax,
        vmin=0.9,
        vmax=1,
        )
    for tk in ax.get_yticklabels():
        tk.set_rotation(0)
    fig.tight_layout()
    fig.savefig('../figures/SupplFig4.svg')
    fig.savefig('../figures/SupplFig4.png')

    plt.ion()
    plt.show()
