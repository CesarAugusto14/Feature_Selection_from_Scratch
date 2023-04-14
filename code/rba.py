import numpy as np

class relief(object):
    """author: cesar augusto sanchez-villalobos @cesarasa.

    The following code is a python implementation of the relief algorithm, and it will
    be used for learning and documentation purposes. The main goal of the code is not to
    replace the previous implementations, but provide a better understanding of the 
    algorithm. 
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
            x_i = self.X[i]
            y_i = self.y[i]
            # Get the nearest hit and miss: 
            hit = self.get_nearest_hits(x_i, y_i)
            miss = self.get_nearest_miss(x_i, y_i)
            # Update the weights: 
            self.weights -= (x_i - hit)**2 - (x_i - miss)**2
            print(self.weights)
        # Get the indices of the features to keep:
        self.indices = np.argsort(self.weights)[::-1][:self.n_features_to_keep]
        # Resulting dataset is:
        self.X_new = self.X[:, self.indices]

    def get_nearest_hits(self, x_i, y_i):
        # Get the nearest hits: 
        hits = self.X[self.y == y_i]
        # Get the distance to the current instance: 
        distances = np.linalg.norm(hits - x_i, axis=1)
        # Get the nearest hit: 
        nearest_hit = hits[np.argmin(distances)]
        return nearest_hit
    
    def get_nearest_miss(self, x_i, y_i):
        # Get the nearest miss: 
        miss = self.X[self.y != y_i]
        # Get the distance to the current instance: 
        distances = np.linalg.norm(miss - x_i, axis=1)
        # Get the nearest miss: 
        nearest_miss = miss[np.argmin(distances)]
        return nearest_miss

