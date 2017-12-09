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
    
    log_likehood_history = []
    for i in range(1, max_components):
        clf = GaussianMixture(i)
        clf.fit(data)
        tmp_bayesian_inference_criterion = clf.bic(data)
        if tmp_bayesian_inference_criterion < min_bayesian_inference_criterion:
            candidate_model = clf
            min_bayesian_inference_criterion = tmp_bayesian_inference_criterion
            candidate_components_n = i
        else:
            break
    def plot_model():
        x_min, x_max = np.min(data), np.max(data)
        x = np.arange(x_min, x_max, 0.1)
        y = candidate_model.score_samples(x.reshape(-1, 1))
        #plt.plot(x, y, 'o')
        plt.plot(log_likehood_history)
        plt.show()
    return candidate_components_n, (candidate_model.means_, 
                                    np.sqrt(candidate_model.covariances_), 
                                    candidate_model.weights_), candidate_model


if __name__ == "__main__":

    # A simple test, we feed a fake data formed by four Gaussian components and check the output.
    a = np.random.normal(10, 1, size = 10) # mean 10, sigma 1
    b = np.random.normal(100, 3, size = 40)
    c = np.random.normal(200, 2, size = 30)
    d = np.random.normal(240, 2, size = 30)
    res, params, _ = clustering(np.concatenate((a, b, c, d)), 20)
    cc = 0


    

