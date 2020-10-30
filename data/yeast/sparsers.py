import numpy as np 
from scipy.cluster import hierarchy

def sparse_cluster(dists, criterion = "maxclust", max_clusters = 35):

    '''
    Compute GSO
    Inputs:
    -------
    dists: PxP distance matrix between SNPs
    criterion: criterion used to split clusters
    max_clusters: max number of clusters computed

    Outputs:
    --------
    GSO: PxP (G)raph (S)hift (O)perator
    '''

    tri = np.zeros(np.sum([dists.shape[1] - i for i in range(1, dists.shape[1])]))
    l = 0
    for i, f in enumerate(dists):
        tri[l:l + f[i+1:].size] = f[i+1:]
        l = l + f[i+1:].size

    tri = 1 - tri

    # hierarchical clustering
    Z = hierarchy.linkage(tri, method = "average")
    fc = hierarchy.fcluster(Z, max_clusters, criterion = criterion)

    coords = [list(np.where(fc == i)[0]) for i in range(1, max_clusters + 1)]
    
    # GSO
    GSO = np.zeros(dists.shape)
    for c in coords:
        for i in c:
            aux = c.copy()
            aux.remove(i)
            GSO[i][aux] = dists[i][aux]

    return GSO 

def sparse_knn(dists, k):
    '''
    Compute GSO
    Inputs:
    -------
    dists: PxP distance matrix between SNPs
    k: int, number of neighboring nodes

    Outputs:
    --------
    GSO: PxP GSO 
    '''
    GSO = np.zeros(dists.shape)
    for i in range(GSO.shape[0]):
        closest = np.argsort(dists[i])[1:1+k]
        GSO[i][closest] = dists[i][closest]

    return GSO

def sparse_thresh(dists, del_rate):
    '''
    Compute GSO
    Inputs:
    -------
    dists: PxP distance matrix between SNPs
    del_rate: float

    Outputs:
    --------
    GSO: PxP GSO 
    '''

    """
    GSO = np.zeros(dists.shape)
    
    num_neighbors = []
    for i in range(GSO.shape[0]):
        closest = np.where(dists[i] > t)[0]
        num_neighbors.append(len(closest))
        GSO[i][closest] = dists[i][closest]
    """
    n_del = int(dists.size * del_rate)
    if (n_del%2 != 0):
        n_del += 1
    indxs = np.argsort(dists, axis = None)[:n_del]
    indxs = np.unravel_index(indxs, dists.shape)
    dists[ indxs ] = 0
    print(f"Sparsification.... DONE. Percentage of non zero values in gso: {np.count_nonzero(dists)*100/dists.size}")
    return dists