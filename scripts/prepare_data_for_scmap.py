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

sys.path.insert(0, '/home/fabio/university/postdoc/northstar/build/lib')
import northstar


if __name__ == '__main__':

    pa = argparse.ArgumentParser()
    pa.add_argument('--reps', type=str, default='1,2,3,4,5', required=False)
    args = pa.parse_args()
    reps = [int(x) for x in args.reps.split(',')]

    tissue = 'Kidney'
    adata = anndata.read_h5ad('../data/TabulaMurisSenis/{:}_droplet.h5ad'.format(tissue))
    adata.obs['CellType'] = adata.obs['cell_ontology_class']

    #print('Save initial subsample with all cell types')
    #asub = northstar.subsample_atlas(
    #        adata,
    #        cell_type_column='cell_ontology_class',
    #        n_cells=20,
    #        )
    #asub.to_df().to_csv(
    #        '../data/for_scmap/TBS_kidney_atlas_subsample_20_counts.tsv',
    #        sep='\t', index=True)
    #asub.obs[['CellType']].to_csv(
    #        '../data/for_scmap/TBS_kidney_atlas_subsample_20_metadata.tsv',
    #        sep='\t', index=True)

    #print('Save subsamples to be annotated')
    #nct = 100
    #for rep in reps:
    #    print('Subsample with {:} cells per type, rep {:}'.format(nct, rep))
    #    asub2 = northstar.subsample_atlas(
    #            adata,
    #            cell_type_column='cell_ontology_class',
    #            n_cells=nct,
    #            )

    #    asub2.to_df().to_csv(
    #            '../data/for_scmap/TBS_kidney_newdata_subsample_100_counts_rep_{:}.tsv'.format(rep),
    #            sep='\t', index=True)
    #    asub2.obs[['CellType']].to_csv(
    #            '../data/for_scmap/TBS_kidney_newdata_subsample_100_metadata_rep_{:}.tsv'.format(rep),
    #            sep='\t', index=True)


    print('Include different subset of cell types in atlas')
    nat = [14, 15, 16, 17]
    csts = adata.obs['cell_ontology_class'].value_counts()
    asub = anndata.read_text(
            '../data/for_scmap/TBS_kidney_atlas_subsample_20_counts.tsv',
            delimiter='\t',
            )
    asub.obs['CellType'] = pd.read_csv(
            '../data/for_scmap/TBS_kidney_atlas_subsample_20_metadata.tsv',
            sep='\t', index_col=0)

    for na in nat:
        csti = csts.index[:na]
        idx = asub.obs['CellType'].isin(csti).values.nonzero()[0]
        asubr = asub[idx]

        asubr.to_df().to_csv(
                '../data/for_scmap/TBS_kidney_atlas_subsample_20_counts_na_{:}.tsv'.format(na),
                sep='\t', index=True)
        asubr.obs[['CellType']].to_csv(
                '../data/for_scmap/TBS_kidney_atlas_subsample_20_metadata_na_{:}.tsv'.format(na),
                sep='\t', index=True)


