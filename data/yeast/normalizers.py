import numpy as np
from time import time
from scipy.linalg  import eigvalsh
def normalize(GSO, method = "eigenvalue"):

    if method == "min_max":
        for i in range(GSO.shape[0]):
            min_ = np.min(GSO[i])
            max_ = np.max(GSO[i])
            GSO[i] = (GSO[i] - min_)/(max_ - min_) 
    elif method == "std":
        for i in range(GSO.shape[0]):
            GSO[i] = (GSO[i] - np.mean(GSO[i]))/np.std(GSO[i])
    elif method == "eigenvalue":
        #l = eigvalsh(GSO, turbo = True, eigvals = (GSO.shape[0]-1, GSO.shape[0]))
        ti = time()
        print("Computing maximum eigenvalue, this may take a while.")
        l = np.linalg.eigvalsh(GSO)
        GSO /= l[-1]
        tf = time()
        #print(f'{tf - ti:.2f} seconds.')
        print(f"Maximum eigenvalue is {l[-1]}.")
        print(f"Computation lasted {tf - ti:.2f} seconds.")

    return GSO