import matplotlib.pyplot as plt
import numpy as np
from code.rba import relief

# Create a synthetic dataset with two classes and 10 features
X = np.array([[1, 2],
             [1, 3],
             [1, 5],
             [2, 3],
             [2, 5],
             [2, 6]])

y = np.array([1, 1, 1, 0, 0, 0])

# Create the relief object
r = relief(n_features_to_keep=1)

# Fit the model
r.fit(X, y)

# Get the resulting dataset
X_new = r.X_new

# Print it:
print(X_new)