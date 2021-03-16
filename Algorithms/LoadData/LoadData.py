#%% DATASETS
#Precargados
from sklearn.datasets import load_boston, fetch_california_housing
X, y = load_boston(return_X_y = True)

X, y = fetch_california_housing(return_X_y = True)

#Descargar de openml
from sklearn.datasets import fetch_openml
mice = fetch_openml(name='miceprotein', version=4)
mice.url

#Generados
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles, make_multilabel_classification, make_regression
X, y = make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0, 
                  center_box=(-10.0, 10.0), shuffle=True, random_state=None)

X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2, 
                           n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, 
                           flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, 
                           shuffle=True, random_state=None)

X, y = make_gaussian_quantiles(mean=None, cov=1.0, n_samples=100, n_features=2,
                               n_classes=3, shuffle=True, random_state=None)

X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, 
                                      n_labels=2, length=50, allow_unlabeled=True, 
                                      sparse=False, return_indicator='dense', 
                                      return_distributions=False, random_state=None)

X, y = make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1, 
                       bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, 
                       shuffle=True, coef=False, random_state=None)