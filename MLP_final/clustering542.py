from sklearn.mixture import GaussianMixture
import numpy as np
import random
import matplotlib.pyplot as plt

def clustering(data, max_components = 20):
    """
    :type input: np.array, int
    :rtype: int, tuple, GMM

    Author: Shawn Han

    This is the clustering function using Gaussian Mixture.

    It returns the optimal number of clusters as well as 
    the centorid, radius and frequency (sum up to 1 for all clusters)
    for the given data.

    The last thing it returns is GMM, which can be ignored if you don't need it.
    It can help you determine the cluster of a given data by using GMM.predict(data)
    
    """

    if data.shape != (-1, 1):
        data = data.reshape(-1, 1)
    candidate_model = None
    candidate_components_n = 0
    min_bayesian_inference_criterion = float('inf')
    
#    log_likehood_history = []
    for i in range(1, max_components):
        clf = GaussianMixture(i)
        clf.fit(data)
        tmp_bayesian_inference_criterion = clf.bic(data)
        if tmp_bayesian_inference_criterion < min_bayesian_inference_criterion:
            candidate_model = clf
            min_bayesian_inference_criterion = tmp_bayesian_inference_criterion
            candidate_components_n = i
            
            # list of all data, showing which clusterId it belongs to
            sample_labels = clf.predict(data) 
        else:
            break

    """ give IDs of valid clusters """

    threshold = 50
    valid = [-1,-1,-1,-1] # considering max 4 clusters
    for i in sample_labels:
        if( i == 0 ):
            valid[0] += 1
        if( i == 1 ):
            valid[1] += 1
        if( i == 2 ):
            valid[2] += 1
        if( i == 3 ):
            valid[3] += 1
            
    # change clusterIDs with < threshold items = -1
    for i in range(0, len(valid)):
        if valid[i] < threshold:
            valid[i] = -1
    
  
    return candidate_components_n, candidate_model.means_, \
                candidate_model.covariances_, candidate_model.weights_, valid

