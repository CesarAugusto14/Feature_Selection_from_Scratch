import matplotlib.pyplot as plt
import numpy as np
from code.rba import relief, relief_f

"""
This code was developed by:
cesar augusto sanchez-villalobos @cesara

To check the results obtained by Urbanowicz et al. (2018) on their paper. The citations are
available through the rba.py file. 

The goal is to recreate the 4, 4, -8 result on Table 2. 
"""
plt.rcParams.update({'font.size': 14, 
                     'font.family': 'Times New Roman', 
                     'figure.figsize': (10, 10)})
X = np.array([[1, 0, 1],
              [1, 0, 0],
              [0, 1, 1],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [1, 1, 1],
              [1, 1, 0]])

N,p = X.shape
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

r = relief_f(n_features_to_keep=2, k = 1)
r.fit(X, y)
X_new = r.X_new


print(r.weights*N)
# Plot importance of features
# plt.figure()
# plt.bar(range(len(r.weights)), r.weights*N, color='peru', ec='k', lw=2, align='center')
# plt.title('Feature importance')
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# # plt.grid()
# # The xticks are f_1, f_2, f_3
# plt.xticks([0, 1, 2], ['$A_1$', '$A_2$', '$A_3$'])

# plt.savefig('./results/test_urbanowicz_example_table_2.png')

# plt.show()