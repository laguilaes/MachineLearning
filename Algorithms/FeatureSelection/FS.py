#%% Quitar variables con poca variancia
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
X, y = load_iris(return_X_y=True)
X.shape
#For regression: f_regression, mutual_info_regression
#For classification: chi2, f_classif, mutual_info_classif
#Seleccionar los K mejores
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape
#Seleccionar los que caen dentro del percentil X
X_new = SelectPercentile(chi2, percentile = 100).fit_transform(X, y)
X_new.shape


#%% SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
X, y = load_iris(return_X_y=True)
X.shape

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape


#%%Hacer un PCA: PCA + fit_transform
# library(caret)
# preProcValues <- preProcess(iris[,-5], method = "pca")
# trainTransformed <- predict(preProcValues, iris[,-5])

import pandas as pd 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species'])

from sklearn.preprocessing import StandardScaler 
features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
x = df.loc[:, features].values
y = df.loc[:,['Species']].values
x = StandardScaler().fit_transform(x) #escalar

from sklearn.decomposition import PCA
df_pca = pd.DataFrame(data = PCA(n_components=2).fit_transform(x), columns = ['PC1', 'PC2']) #dos PC mas representativas
df_pca = pd.DataFrame(data = PCA(0.70).fit_transform(x), columns = ['PC1']) #nnumero de variables necesarias para alcanzar 0.7 de variancia
df_final = pd.concat([df_pca, df[['Species']]], axis = 1)



#%%Quitar variables correlacionadas
# library(caret)
# preProcValues <- preProcess(iris[,-5], method = "corr") #se puede incluir "nzv", "pca", "ica" (independent compnent analysis, to find linear combinations)
# trainTransformed <- predict(preProcValues, iris)

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
X, y = load_iris(return_X_y=True)
X=pd.DataFrame(X)
cors=X.corr(method='pearson')

plt.matshow(cors, cmap=plt.cm.RdYlGn)
plt.colorbar()



#%% DIMENSIONALITY REDUCTION
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
print(__doc__)

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# ----------------------------------------------------------------------
# Plot images of the digits
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')


# ----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits")


# ----------------------------------------------------------------------
# Projection on to the first 2 principal components
#PCA substracts de mean, unlike SVD. If your features are least sensitive (informative) 
#towards the mean of the distribution, then it makes sense to subtract the mean. 
#If the features are most sensitive towards the high values, then subtracting the mean does not make sense.

print("Computing PCA projection")
t0 = time()
X_svd = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
X_pca = decomposition.PCA(n_components=2).fit_transform(X) #center but doest not scale data before aplying SVD
X_ipca = decomposition.IncrementalPCA(n_components=2).fit_transform(X) #for large dataset that do not fit in memory
plot_embedding(X_svd, "Singular value decomposition projection of the digits ")
plot_embedding(X_pca, "Principal Components projection of the digits ")
plot_embedding(X_ipca, "Incremental Principal Components projection of the digits ")

# ----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components

print("Computing Linear Discriminant Analysis projection")
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible. Creo que no es necesario
t0 = time()
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
plot_embedding(X_lda,"Linear Discriminant projection of the digits (time %.2fs)" % (time() - t0))


# ----------------------------------------------------------------------
# Isomap projection of the digits dataset
print("Computing Isomap projection")
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding(X_iso, "Isomap projection of the digits (time %.2fs)" %(time() - t0))


# ----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle, "Locally Linear Embedding of the digits (time %.2fs)" %(time() - t0))


# ----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='modified')
t0 = time()
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_mlle, "Modified Locally Linear Embedding of the digits (time %.2fs)" %(time() - t0))


# ----------------------------------------------------------------------
# HLLE embedding of the digits dataset
#Requires: n_neighbors > n_components * (n_components + 3) / 2
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,  method='hessian')
t0 = time()
X_hlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_hlle, "Hessian Locally Linear Embedding of the digits (time %.2fs)" %(time() - t0))


# ----------------------------------------------------------------------
# LTSA embedding of the digits dataset
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,  method='ltsa')
t0 = time()
X_ltsa = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_ltsa, "Local Tangent Space Alignment of the digits (time %.2fs)" %
               (time() - t0))

# ----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds, "MDS embedding of the digits (time %.2fs)" % (time() - t0))

# ----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
print("Computing Totally Random Trees embedding")
RTE = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5).fit_transform(X)
X_reduced  = decomposition.TruncatedSVD(n_components=2).fit_transform(RTE)


plot_embedding(X_reduced, "Random forest embedding of the digits")

# ----------------------------------------------------------------------
# Spectral embedding of the digits dataset
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(X)

plot_embedding(X_se,"Spectral embedding of the digits (time %.2fs)" %(time() - t0))

# ----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne, "t-SNE embedding of the digits (time %.2fs)" %(time() - t0))

# ----------------------------------------------------------------------
# NCA projection of the digits dataset
print("Computing NCA projection")
nca = neighbors.NeighborhoodComponentsAnalysis(init='random', n_components=2, random_state=0)
t0 = time()
X_nca = nca.fit_transform(X, y)

plot_embedding(X_nca,"NCA embedding of the digits (time %.2fs)" %(time() - t0))


# ----------------------------------------------------------------------
# NCA projection of the digits dataset
print("Computing ICA projection")
ica = decomposition.FastICA(n_components=2)
X_ica = ica.fit_transform(X)

plot_embedding(X_nca,"ICA embedding of the digits ")

#-----------------------------------------------------------------------
# Feature Agglomeration
import numpy as np
from sklearn import datasets, cluster
digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
agglo = cluster.FeatureAgglomeration(n_clusters=32)
agglo.fit(X)

X_reduced = agglo.transform(X)
X_reduced.shape