# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:26:39 2020

@author: Juan
"""

from snp_correlation import corr
from sparsers import sparse_thresh, sparse_knn, sparse_cluster
from normalizers import normalize
from fermat_distance import fermat_distance_matrix
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
#%%

def geno_to_gso(X, dist = "corr", norm_method = "eigenvalue", sparse_method = "thresh", del_rate = 0.99, criterion = "maxclust", max_clusters = 35, k = 100):
    """

    Parameters
    ----------
    X : numpy array
        SNP Matrix.
    dist : str, optional
        Distance to be used. The default is "corr".
    norm_method : str, optional
        Normalization method. The default is "eigenvalue".
    sparse_method : str, optional
        Sparsification method. The default is "thresh".
    del_rate : float, optional
        Deletion rate for sparsification (only used with sparse_method = "thresh"). The default is None.
    criterion : TYPE, optional
        Clustering criterion (only used with sparse_method = "clust"). The default is "maxclust".
    max_clusters : TYPE, optional
        The default is 35.
    k : TYPE, optional
        Number of neighbors. Only used with sparse_method = "knn". The default is None.

    Returns
    -------
    gso: numpy array
        Graph Shift Operator
    """   
    if dist == "corr":
        gso = corr(X)
    elif dist == "fermat":
        gso = fermat_distance_matrix(X)
    else:
        print("Wrong distance dude.. choose either corr or fermat.")
    
    if sparse_method == "thresh":
        gso = sparse_thresh(gso, del_rate)
    elif sparse_method == "cluster":
        gso = sparse_cluster(gso, criterion, max_clusters)
    elif sparse_method == "knn":
        gso = sparse_knn(gso, k)
    else:
        print("Wrong sparsification method dude.. choose either thresh, clust or knn.")
    
    gso = normalize(gso, norm_method)
    
    return gso

    
if __name__ == "__main__":
    
    ### Load data
    print("Usage example: python geno_to_gso.py  data/yeast/geno_yeast_congored.npy data/yeast/pheno_yeast_congored.npy ")
    path_to_geno = sys.argv[1]
    path_to_pheno = sys.argv[2]
    X = np.load(path_to_geno, allow_pickle = True)
    y = np.load(path_to_pheno, allow_pickle = True)
    
    ### Train Val Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
    
    ### Build GSO
    gso = geno_to_gso(X_train)
    print(f"The shape of the GSO is: {gso.shape}")
    
    ### Build Bengio's dictionaries
    train_dict = {"GSO": gso, "X": X_train, "y": y_train}
    val_dict = {"GSO": gso, "X": X_val, "y": y_val}
    test_dict = {"GSO": gso, "X": X_test, "y": y_test}
    
    ### Dump pickle-Ricks
    print("Dumping pickles...")
    pickle_out = open("train.pickle","wb")
    pickle.dump(train_dict, pickle_out)
    pickle_out.close()
    pickle_out = open("val.pickle","wb")
    pickle.dump(val_dict, pickle_out)
    pickle_out.close()
    pickle_out = open("test.pickle","wb")
    pickle.dump(test_dict, pickle_out)
    pickle_out.close()
    
    
    
    