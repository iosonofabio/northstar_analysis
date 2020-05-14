# vim: fdm=indent
'''
author:     Fabio Zanini
date:       12/05/20
content:    Test scvi on a large dataset from Tabula Muris Senis
'''
import os
import sys
import argparse
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

    pa = argparse.ArgumentParser()
    pa.add_argument('--reps', type=str, required=True)
    pa.add_argument('--nat', type=str, required=True)
    pa.add_argument('--collect', action='store_true')
    args = pa.parse_args()
    reps = [int(x) for x in args.reps.split(',')]
    nat = [int(x) for x in args.nat.split(',')]

    if not args.collect:
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
        nct = 100

        for na in nat:
            for rep in reps:
                print('Restricting atlas to the most common {:} cell types'.format(na))
                csti = csts.index[:na]
                idx = asub.obs['cell_ontology_class'].isin(csti).values.nonzero()[0]
                asubr = asub[idx]

                print('Subsample with {:} cells per type, rep {:}'.format(nct, rep))
                asub2 = northstar.subsample_atlas(
                        adata,
                        cell_type_column='cell_ontology_class',
                        n_cells=nct,
                        )
                ntot = asub2.X.shape[0]

                print('Run scvi')
                ##############################################################
                # SCVI
                ##############################################################
                import time
                from scvi.dataset import AnnDatasetFromAnnData
                from scvi.dataset.dataset import GeneExpressionDataset
                from scvi.inference import UnsupervisedTrainer
                from scvi.models import SCANVI, VAE
                from umap import UMAP
                import scanpy as sc

                # TODO: import the datasets into SCVI objects (sigh!)
                # scVI wants raw counts, but who knows about those TabulaMurisSenis data
                # quick and dirty solution for now
                asubr_scvi = asubr.copy()
                asubr_scvi.X.data = asubr_scvi.X.data.astype(np.int64)
                ds_atlas = AnnDatasetFromAnnData(asubr_scvi)

                asub2_scvi = asub2.copy()
                asub2_scvi.X.data = asub2_scvi.X.data.astype(np.int64)
                ds_new = AnnDatasetFromAnnData(asub2_scvi)

                all_dataset = GeneExpressionDataset()
                all_dataset.populate_from_datasets([ds_atlas, ds_new])

                ##############################################################
                t0 = time.time()
                print('Prepare some data structures')
                vae = VAE(
                    all_dataset.nb_genes,
                    n_batch=all_dataset.n_batches,
                    n_labels=all_dataset.n_labels,
                    n_hidden=128,
                    n_latent=30,
                    n_layers=2,
                    dispersion='gene',
                    )

                print('Prepare the trainer')
                trainer = UnsupervisedTrainer(vae, all_dataset, train_size=1.0)

                print('Train neural network')
                n_epochs = 100
                trainer.train(n_epochs=n_epochs)

                print('Get posteriors (latent space)')
                full = trainer.create_posterior(
                        trainer.model, all_dataset, indices=np.arange(len(all_dataset)),
                        )
                latent, batch_indices, labels = full.sequential().get_latent()
                batch_indices = batch_indices.ravel()

                print('Use scanpy and Leiden to cluster in latent space')
                adata_latent = sc.AnnData(latent)
                sc.pp.neighbors(adata_latent, use_rep='X', n_neighbors=30, metric='minkowski')
                sc.tl.leiden(adata_latent, resolution=0.8)
                clusters = adata_latent.obs.leiden.values.to_dense().astype(str)

                t1 = time.time()
                ##############################################################
                t = t1 - t0

                identity_atlas = asubr.obs[['CellType']].copy()
                identity_atlas['scvi_assignment'] = clusters[:asubr.shape[0]]
                cats_map_atlas = {}
                for ct in np.unique(asubr.obs['CellType']):
                    tmp = identity_atlas.loc[identity_atlas['CellType'] == ct, 'scvi_assignment']
                    tmpc = tmp.value_counts()
                    cats_map_atlas[ct] = tmpc.index[0]

                # Cell types that ended up in two or more clusters are merged, so there is no one-to-one
                tmp3 = Counter(cats_map_atlas.values())
                ctmap = {}
                for key, val in cats_map_atlas.items():
                    if tmp3[val] == 1:
                        ctmap[val] = key
                    else:
                        ctmap[val] = val
                identity_atlas['scvi_assignment_mapped'] = identity_atlas['scvi_assignment'].replace(ctmap)

                gof_atlas = (identity_atlas['scvi_assignment_mapped'] == identity_atlas['CellType']).mean()

                new_cats = []
                for cli in np.unique(clusters):
                    if cli not in ctmap:
                        new_cats.append(cli)
                        ctmap[cli] = cli

                identity = asub2.obs[['cell_ontology_class']].copy()
                identity['scvi_assignment'] = clusters[asubr.shape[0]:]
                identity['scvi_assignment_mapped'] = identity['scvi_assignment'].replace(ctmap)

                # Measure fraction of correct
                identity['correct'] = identity['cell_ontology_class'] == identity['scvi_assignment_mapped']

                # Extend to new identifiable clusters
                atlas_cts = asubr.obs['CellType'].unique()
                cats_map = {}
                for nc in new_cats:
                    idx = identity['scvi_assignment_mapped'] == nc
                    ctx = identity.loc[idx, 'cell_ontology_class']
                    ct_most = ctx.value_counts().index[0]
                    cats_map[nc] = ct_most
                    if ct_most not in atlas_cts:
                        idxt = idx & (identity['cell_ontology_class'] == ct_most)
                        identity.loc[idxt, 'correct'] = True
                    else:
                        # FIXME: not really fine, but ok for now
                        pass
                gof = identity['correct'].mean()

                print(gof)

                res = {
                    'na': na,
                    'time': t,
                    'gof': gof,
                    'ntot': ntot,
                    'rep': rep,
                    'identity': identity,
                    'cats_map_atlas': cats_map_atlas,
                    'new_cats': new_cats,
                    'cats_map': cats_map,
                    }
                import pickle
                fn = '../data_for_figures/scvi_kidney/result_na_{:}_rep_{:}.pkl'.format(
                    na, rep,
                    )
                with open(fn, 'wb') as f:
                    pickle.dump(res, f)


    else:

        import pickle
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

        res_avg = resd[['na', 'gof', 'time']].groupby('na').mean()
        res_std = resd[['na', 'gof', 'time']].groupby('na').std()

        print('Plot results')
        colors = sns.color_palette('Dark2', n_colors=2)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.errorbar(
                res_avg.index, res_avg['time'] / 60.,
                yerr=res_std['time'] / 60.,
                fmt='-o',
                lw=2, color=colors[0],
                )
        ax.set_xlabel('Number of cells types\nin atlas (out of 18)')
        ax.set_ylabel('Runtime [min]')
        ax.scatter(
                [-0.16], [0.2], color=colors[0], clip_on=False,
                transform=ax.transAxes,
                )
        ax.set_ylim(0, 12)

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
        fig.savefig('../figures/scvi_tms_kidney_performance_various_cell_type_numbers.png')
        fig.savefig('../figures/scvi_tms_kidney_performance_various_cell_type_numbers.svg')



