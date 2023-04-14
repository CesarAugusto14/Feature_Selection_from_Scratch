import numpy as np

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
            self.weights -= (x_i - hit)**2/self.N - (x_i - miss)**2./self.N
        # Get the indices of the features to keep:
        self.indices = np.argsort(self.weights)[::-1][:self.n_features_to_keep]
        # Resulting dataset is:
        self.X_new = self.X[:, self.indices]

    def get_nearest_hits(self, x_i, y_i):
        # Get the nearest hits: 
        hits = self.X[self.y == y_i]
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

    The main difference, is that relief-F uses more than one nearest hit and miss to
    compute the weights.

    n_features_to_keep: Number of features to keep. (self explanatory, tbh)
    X: The feature matrix, this shall be a numpy array (NOTE: add an assertion for this, and if not, convert it to a numpy array).
    y: The target vector, this shall be a numpy array (NOTE: add an assertion for this, and if not, convert it to a numpy array).
    k: Number of nearest hits and misses to use.

    
    """
    def __init__(self, n_features_to_keep=2, k=5):
        self.n_features_to_keep = n_features_to_keep
        self.k = k

    def fit(self, X, y):
        pass
    

