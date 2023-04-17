import matplotlib.pyplot as plt
import numpy as np
from code.rba import relief, relief_f
np.random.seed(0)
# Create a synthetic dataset with two classes and 10 features
X = np.array([[1, 2],
             [1, 3],
             [1, 5],
             [2, 3],
             [2, 5],
             [2, 6]])

# Let's add some noisy features:
X = np.hstack((X, np.random.rand(6, 100)))
y = np.array([1, 1, 1, 0, 0, 0])

# Create the relief object
r = relief(n_features_to_keep=2)

# Fit the model
r.fit(X, y)

# Get the resulting dataset
X_new = r.X_new

# Print it:
print(X_new)

# Plot importance of features
plt.figure()
plt.bar(range(len(r.weights)), r.weights, color='peru', ec='k', lw=2, align='center')
plt.title('Feature importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# # Create the relief object
# r = relief_f(n_features_to_keep=2, k = 3)

# # Fit the model
# r.fit(X, y)

# # Get the resulting dataset
# X_new = r.X_new

# # Print it:
# print(X_new)

# # Plot importance of features
# plt.figure()
# plt.bar(range(len(r.weights)), r.weights, color='peru', ec='k', lw=2, align='center')
# plt.title('Feature importance')
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.show()
