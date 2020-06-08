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

    ct_atlas = pd.read_csv(
            '../data/for_scmap/TBS_kidney_atlas_subsample_20_metadata.tsv',
            sep='\t', index_col=0, squeeze=True)
    cst = np.unique(ct_atlas)
    na = len(cst)

    print('First, evaluate on a complete atlas')
    print('Times by hand (copy/paste from R, in seconds)')
    times = np.array([2.964792, 3.048675, 3.101496, 3.108727, 3.036383])
    # Add time to prepare the atlas, because it is part of northstar's timing too
    time_atlas = 0.3800428
    times += time_atlas

    nct = 100
    ress = []
    for rep in reps:
        print('Read true cell types')
        ct_newdata = pd.read_csv(
                '../data/for_scmap/TBS_kidney_newdata_subsample_100_metadata_rep_{:}.tsv'.format(rep),
                sep='\t', index_col=0)


        print('Read scmap result')
        cmres = pd.read_csv(
                '../data/for_scmap/TBS_kidney_newdata_subsample_100_metadata_rep_{:}_output.tsv'.format(rep),
                sep='\t', index_col=0).T

        # R counts indices from 1
        cmres -= 1

        # Convert to cell types
        ct_res = pd.DataFrame(ct_atlas[cmres.values], index=cmres.index)

        from collections import Counter
        ct_rest = pd.DataFrame(np.zeros((len(cmres), len(cst)), int), index=cmres.index, columns=cst)
        for i, ct in enumerate(cst):
            ct_rest.iloc[:, i] += (ct_res == ct).sum(axis=1)

        ct_resf = []
        for ct, cou in ct_rest.iterrows():
            cou = cou.sort_values(ascending=False)
            ct_resf.append(cou.index[0])
        ct_resf = pd.Series(ct_resf, index=cmres.index)
        ct_resf = ct_resf.loc[ct_newdata.index]

        identity = ct_newdata.copy()
        identity.columns = ['CellType']
        identity['scmap_assignment'] = ct_resf

        gof = (identity['CellType'] == identity['scmap_assignment']).mean()

        res = {
            'na': na,
            'rep': rep,
            'identity': identity,
            'gof': gof,
            'ntot': identity.shape[0],
            'time': times[int(rep) - 1],
            }
        import pickle
        fn = '../data_for_figures/scmap_kidney/result_na_{:}_rep_{:}.pkl'.format(
            na, rep,
            )
        with open(fn, 'wb') as f:
            pickle.dump(res, f)

        ress.append(res)


    print('Include different subset of cell types in atlas')
    print('Times by hand (copy/paste from R, in seconds)')
    times_incomplete = {
        '17': [3.331619, 3.574516, 3.318047, 2.877976, 3.241976],
        '16': [3.14454, 2.76651, 3.233341, 3.0351, 3.155047],
        '15': [3.38759, 3.302768, 2.818339, 3.295036, 3.26331],
        '14': [3.123514, 2.901536, 3.275896, 3.392008, 3.354692],
        }
    # Add time to prepare the atlas, because it is part of northstar's timing too
    time_atlas_incomplete = {
        17: 0.4147904,
        16: 0.3446839,
        15: 0.3229709,
        14: 0.3133569,
        }
    for key, val in times_incomplete.items():
        times_incomplete[key] = np.array(val) + time_atlas_incomplete[key]

    nat = [17, 16, 15, 14]
    for na in nat:
        ct_atlas = pd.read_csv(
                '../data/for_scmap/TBS_kidney_atlas_subsample_20_metadata_na_{:}.tsv'.format(na),
                sep='\t', index_col=0, squeeze=True)
        cst = np.unique(ct_atlas)

        for rep in reps:
            print('Read true cell types')
            ct_newdata = pd.read_csv(
                    '../data/for_scmap/TBS_kidney_newdata_subsample_100_metadata_rep_{:}.tsv'.format(rep),
                    sep='\t', index_col=0)


            print('Read scmap result')
            cmres = pd.read_csv(
                    '../data/for_scmap/TBS_kidney_newdata_subsample_100_metadata_rep_{:}_na_{:}_output.tsv'.format(rep, na),
                    sep='\t', index_col=0).T

            # R counts indices from 1
            cmres -= 1

            # Convert to cell types
            ct_res = pd.DataFrame(ct_atlas[cmres.values], index=cmres.index)

            from collections import Counter
            ct_rest = pd.DataFrame(np.zeros((len(cmres), len(cst)), int), index=cmres.index, columns=cst)
            for i, ct in enumerate(cst):
                ct_rest.iloc[:, i] += (ct_res == ct).sum(axis=1)

            ct_resf = []
            for ct, cou in ct_rest.iterrows():
                cou = cou.sort_values(ascending=False)
                ct_resf.append(cou.index[0])
            ct_resf = pd.Series(ct_resf, index=cmres.index)
            ct_resf = ct_resf.loc[ct_newdata.index]

            identity = ct_newdata.copy()
            identity.columns = ['CellType']
            identity['scmap_assignment'] = ct_resf

            gof = (identity['CellType'] == identity['scmap_assignment']).mean()

            res = {
                'na': na,
                'rep': rep,
                'identity': identity,
                'gof': gof,
                'ntot': identity.shape[0],
                'time': times[int(rep) - 1],
                }
            import pickle
            fn = '../data_for_figures/scmap_kidney/result_na_{:}_rep_{:}.pkl'.format(
                na, rep,
                )
            with open(fn, 'wb') as f:
                pickle.dump(res, f)

            ress.append(res)
