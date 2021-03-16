#%% Spectral clustering

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering

# Generate sample data

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

labels = SpectralClustering(n_clusters=6, eigen_solver='arpack', affinity="nearest_neighbors").fit(X).labels_
plt.scatter(X[:,0], X[:,1], c=labels)