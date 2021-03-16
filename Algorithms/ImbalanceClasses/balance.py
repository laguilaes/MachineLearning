 #DOWNSAMPLING

#%% Otro metodo: RandomUnderSampler + fit_sample
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)
X_rus, y_rus = RandomUnderSampler().fit_sample(X, y)


#%% Downsampling using TomekLinks: TomekLinks + fit_sample

from imblearn.under_sampling import TomekLinks
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(
    n_classes=3, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=500, random_state=10
)

tl = TomekLinks()
X_tl, y_tl = tl.fit_sample(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.show()

plt.scatter(X_tl[:, 0], X_tl[:, 1], marker='o', c=y_tl, s=25, edgecolor='k')
plt.show()

#%% Downsampling using Cluster Centroids

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from imblearn.under_sampling import ClusterCentroids

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=500, random_state=10
)
cc = ClusterCentroids()
X_cc, y_cc = cc.fit_sample(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.show()

plt.scatter(X_cc[:, 0], X_cc[:, 1], marker='o', c=y_cc, s=25, edgecolor='k')
plt.show()















# UPSAMPLING

from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)
X_ros, y_ros = RandomOverSampler().fit_sample(X, y)


#%% Upsampling using SMOTE

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)

smote = SMOTE()
X_sm, y_sm = smote.fit_sample(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.show()

plt.scatter(X_sm[:, 0], X_sm[:, 1], marker='o', c=y_sm, s=25, edgecolor='k')
plt.show()

#%% Downsampling + Oversampling
from imblearn.combine import SMOTETomek
from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)

smt = SMOTETomek()
X_smt, y_smt = smt.fit_sample(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.show()

plt.scatter(X_smt[:, 0], X_smt[:, 1], marker='o', c=y_smt, s=25, edgecolor='k')
plt.show()