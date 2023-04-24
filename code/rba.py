"""AJA CONOETUMADRE"""
import numpy as np
# I will be importing pdist and squareform just like we did in Multivariate Statistical Analysis, 
# I hope in the near future to change this!
from scipy.spatial.distance import pdist, squareform
class relief(object):
    """author: cesar augusto sanchez-villalobos @cesarasa.

    The following code is a python implementation of the relief algorithm, and it will
    be used for learning and documentation purposes. The main goal of the code is not to
    replace the previous implementations, but provide a better understanding of the 
    algorithm. 

    n_features_to_keep: Number of features to keep. (self explanatory, tbh)
    X: The feature matrix, this shall be a numpy array (NOTE: add an assertion for this, and if not, convert it to a numpy array).
    y: The target vector, this shall be a numpy array (NOTE: add an assertion for this, and if not, convert it to a numpy array).

    Note that, just as skrebate, we won't use the m parameter, as the scores are more representative is m=N, so we will use the whole 
    dataset to compute the scores, always. 

    NOTE: we should be using the distance metric from the paper, but I guess the norm will do it for now.
    NOTE 2: Are there an explanation of why the distance is euclidean or so? Which is the best distance? How the metric affects the results?
    NOTE 3: It would be more computationally efficient to use a distance matrix. Maybe we can add this as an option in the future.
    """
    def __init__(self, n_features_to_keep=2):
        self.n_features_to_keep = n_features_to_keep


    def fit(self, X, y):
        # Get the information from the data: 
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        # Initialize the weights: 
        self.weights = np.zeros(self.D)
        for i in range(self.N):
            # Get the current instance: 
            x_i = self.X[i, :]
            y_i = self.y[i]
            # Get the nearest hit and miss: 
            hit = self.get_nearest_hits(x_i, y_i)
            miss = self.get_nearest_miss(x_i, y_i)
            # Update the weights: 
            self.weights -= ((x_i - hit)**2 - (x_i - miss)**2)/self.N
            print(self.weights)
            print(((x_i - hit)**2))
            # print('Instance: ')
            # print(x_i)
            # print('Hit: ')
            # print(hit)
            # print('Miss: ')
            # print(miss)
        # Get the indices of the features to keep:
        self.indices = np.argsort(self.weights)[::-1][:self.n_features_to_keep]
        # Resulting dataset is:
        self.X_new = self.X[:, self.indices]

    def get_nearest_hits(self, x_i, y_i):
        # Get the nearest hits: 
        hits = self.X[self.y == y_i]
        # Find x_i in the hits:
        hits = np.delete(hits, np.where((hits == x_i).all(axis=1))[0], axis=0)
        # Get the distance to the current instance: 
        distances = np.linalg.norm(hits - x_i, axis=1, ord=1)
        # Get the nearest hit: 
        nearest_hit = hits[np.argmin(distances)]
        return nearest_hit
    
    def get_nearest_miss(self, x_i, y_i):
        # Get the nearest miss: 
        miss = self.X[self.y != y_i]
        
        # Get the distance to the current instance: 
        distances = np.linalg.norm(miss - x_i, axis=1, ord=1)
        # Get the nearest miss: 
        nearest_miss = miss[np.argmin(distances)]
        return nearest_miss

class relief_f(object):
    """author: cesar augusto sanchez-villalobos @cesarasa.
    
    The following code is a python implementation of the relief-F algorithm, which is an 
    extension of the relief algorithm. 

    The main difference, is that relief-F uses more than k nearest hits and k nearest misses, 
    so we will need to compute slightly different the weights. 

    To follow this implementation, please refer to the paper: 
    https://www.sciencedirect.com/science/article/pii/S1532046418301400 
    Relief-based feature selection: Introduction and review, by Ryan J. Urbanowicz, 2018. 
    Journal of Biomedical Informatics, Volume 85, 2018, Pages 189-203. 
    https://doi.org/10.1016/j.jbi.2018.07.014 
    Algorithm 1.

    And Benchmarking relief-based feature selection methods for bioinformatics data mining, by
    Ryan J. Urbanowicz, 2018. 
    https://www.sciencedirect.com/science/article/pii/S1532046418301412?via%3Dihub
    Journal of Biomedical Informatics, Volume 85, 2018, Pages 168-188.
    https://doi.org/10.1016/j.jbi.2018.07.015
    Algorithm 2. 
    """
    def __init__(self, n_features_to_keep=2, k=5):
        """
        Inputs: 
        n_features_to_keep: Number of features to keep. (self explanatory, tbh)
        k: Number of nearest hits and misses to use.

        k will be the number of nearest hits AND the number of nearest misses, 
        this means we are NOT using the number of nearest neighbors, where
        k_nearest_neighbors = 2*k.
        """
        self.n_features_to_keep = n_features_to_keep
        self.k = k

    def fit(self, X, y):
        self.X = X
        n, p = self.X.shape
        self.y = y
        # Get the classes:
        classes = np.unique(y)
        # Compute all the distances:
        distances = self.compute_distances()
        # Initialize the weights: 
        self.weights = np.zeros(p).reshape(1,-1)
        for i in range(n):
            # Get the current instance:
            x_i = X[i, :].reshape(1, -1)
            y_i = y[i]
            # Get the nearest hits and misses:            
            hits = self.get_nearest_hits(x_i, y_i, distances, k=self.k, i = i).reshape(self.k, -1)
            misses = self.get_nearest_miss(x_i, y_i, distances, k=self.k, i = i).reshape(self.k, -1)
            hit_dis = (x_i - hits)**2
            avg_hits = np.mean(hit_dis, axis=0)/self.k
            miss_dis = (x_i - misses)**2
            avg_misses = np.mean(miss_dis, axis=0)/self.k
            # Update the weights:
            self.weights -= (avg_hits - avg_misses)/n
        # Get the indices of the features to keep:
        self.indices = np.argsort(self.weights)[::-1][:self.n_features_to_keep]
        # Resulting dataset is:
        self.X_new = self.X[:, self.indices]

        # NOTE: Now it works!

    
    def compute_distances(self):
        return squareform(pdist(self.X, 'cityblock'))
    
    def get_nearest_hits(self, x_i, y_i, distances, k, i):
        hits = self.X[self.y == y_i]
        distance = distances[i, :]
        distance = distance[self.y == y_i]
        hits = np.delete(hits, np.where((hits == x_i).all(axis=1))[0], axis=0)
        distance = distance[distance != 0]
        hits = hits[np.argsort(distance)]
        nearest_hits = hits[:k]
        return nearest_hits
    
    def get_nearest_miss(self, x_i, y_i, distances, k, i):
        miss = self.X[self.y != y_i]
        distance = distances[i, :]
        distance = distance[self.y != y_i]
        miss = np.delete(miss, np.where((miss == x_i).all(axis=1))[0], axis=0)
        distance = distance[distance != 0]
        miss = miss[np.argsort(distance)]
        nearest_miss = miss[:k]
        return nearest_miss
    

