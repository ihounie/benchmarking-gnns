#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:00:36 2020

@author: dnai

"""
from fermat import Fermat
from sklearn.metrics import euclidean_distances
import numpy as np

def fermat_distance_matrix(X, metric="euclidean", scale=True, d=3, k=10, landmarks=50):
    '''
    Parameters
    ----------
    X : GENOTYPE MATRIX (NxSNPs- rows are individuals)
    distance :  STR
        Distance used to calculate fermat distance. The default is "euclidean".
    scale : BOOL.
        Wether to scale distances prior to fermat estimation. The default is True.
    d : INT, optional
        Related to size of latent Space/Manifold dim.
    k : INT, optional
        Number of neighbours used in estimation. Larger is more exact but has more computational cost. The default is 10.
    landmarks : INT, optional
        See fermat docs. The default is 50.

    Returns
    -------
    None.

    '''
    if metric=="euclidean":
        dist = euclidean_distances(X.T, X.T)
    else:
        raise NotImplementedError
    if scale:
        avg = np.mean(dist)
        if  avg == 0:
            avg = avg+10e-10
        dist = dist / avg
    fermat = Fermat(d, path_method='L', k=k, landmarks=landmarks)
    print("calculating fermat approx distances")
    fermat.fit(dist)
    fermat_distances = fermat.get_distances()
    return fermat_distances
