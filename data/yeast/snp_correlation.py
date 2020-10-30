# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:18:22 2020

@author: Juan
"""

import numpy as np

def corr(X):
    """
    

    Parameters
    ----------
    X : numpy array
        Input Matrix, each row is an individual, each column a SNP.

    Returns
    -------
    R: numpy array
        Correlation matrix.

    """
    return np.corrcoef(X.T)

