# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:46:32 2020

@author: laguila
"""


#%% LOADING DATA 
#%% DictVectorizer
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vec.fit_transform(measurements).toarray()
vec.get_feature_names()


#%% Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
X.toarray()


#%% Image feature extraction
import numpy as np
from sklearn.feature_extraction import image

one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
one_image[:, :, 0]  # R channel of a fake RGB picture

patches = image.extract_patches_2d(one_image, (2, 2), max_patches=2,
    random_state=0)
patches.shape

patches[:, :, :, 0]

patches = image.extract_patches_2d(one_image, (2, 2))
patches.shape

patches[4, :, :, 0]

#Reconstruir la imagen
reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4, 3))
np.testing.assert_array_equal(one_image, reconstructed)


#Y para varias imagenes, 
five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
patches = image.PatchExtractor((2, 2)).transform(five_images)
patches.shape


#%%IMPUTE DATA
#SimpleImputer
import numpy as np
from sklearn.impute import SimpleImputer

X = [[-1, 2], [6, -1], [7, 6]] #-1 en vez de np.nan
imp = SimpleImputer(missing_values=-1, strategy='mean').fit(X)

print(imp.transform(X))

import pandas as pd
df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", "y"]], dtype="category")

imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(df))

#%%Iterative Imputer, usa valores de otras columnas
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])

X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
# the model learns that the second feature is double the first
print(np.round(imp.transform(X_test)))



#%% KNN imputer
import numpy as np
from sklearn.impute import KNNImputer
nan = np.nan
X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer.fit_transform(X)

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



 #%% Crear train y test
# library(rsample)
# set.seed(123)
# split_strat  <- initial_split(iris, prop = 0.7, strata = "Species")
# train_strat  <- training(split_strat)
# test_strat   <- testing(split_strat)
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])

#Division lineal
p_train = 0.80 # Porcentaje de train.
train = datos[:int((len(datos))*p_train)] 
test = datos[int((len(datos))*p_train):]

from sklearn.model_selection import train_test_split 
train, test = train_test_split(datos, test_size = 0.30, random_state = 123, shuffle = False)

#Division aleatoria
from sklearn.model_selection import train_test_split 
train, test = train_test_split(datos, test_size = 0.30, random_state = 123, shuffle = True, stratify=datos.Species)
from collections import Counter
Counter(train.Species)

#Formas alternativas de leer iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


import pandas as pd 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species'])


#%%One-hot encoding
# library(caret)
# data(iris)
# dummies <- dummyVars(formula = Sepal.Length ~ ., data = iris)
# as.data.frame(predict(dummies, newdata = iris))
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
values = array(['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot'])
print(values)
# integer encode: LabelEncoder + fit_transform
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode: OneHotEncoder + fit_transform
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1) #option1
integer_encoded = integer_encoded[:,np.newaxis] #option 2
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[2, :])])
print(inverted)


#Con pandas: pd.get_dummies
import pandas as pd
pd.get_dummies(values)

#%% Escalar y centrar

from sklearn.datasets import load_iris
iris = load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])

# library(caret)
# preProcValues <- preProcess(iris[,-5], method = "scale") 
# trainTransformed <- predict(preProcValues, iris[,-5])

# library(caret)
# preProcValues <- preProcess(iris[,-5], method = "center") 
# trainTransformed <- predict(preProcValues, iris[,-5])

#Escalado estandard: scale + fit_transform
from sklearn import preprocessing
datos_scaled=preprocessing.scale(datos.iloc[:,list(range(0,4))])
datos_scaled.mean(axis=0)
datos_scaled.std(axis=0)
#Escalado estandard 2 StandardScaler + fit_transform
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
datos_scaled = std_scaler.fit_transform(datos.iloc[:,list(range(0,4))])
#Escalado minimo maximo MinMaxScaler + fit_transform
from sklearn import preprocessing
minmaxscaler=preprocessing.MinMaxScaler(feature_range = (0,1))
datos_scaled=minmaxscaler.fit_transform(datos.iloc[:,list(range(0,4))])
#Escalado con outliers RobustScaler + fit_transform
from sklearn import preprocessing
robustscaler=preprocessing.RobustScaler()
datos_scaled=robustscaler.fit_transform(datos.iloc[:,list(range(0,4))])
#Mapping to an uniform distribution: QuantileTransformer + fit_transform
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]) 
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100])

import matplotlib.pyplot as plt
plt.hist(X_train[:, 0])
plt.hist(X_train_trans[:, 0])
plt.show()

#Mapping to a Gaussian distribution: 'box-cox' o 'Yeo-Johnson': PowerTransformer + fit_transform
from sklearn import preprocessing
gaussian_scaler = preprocessing.PowerTransformer(method='box-cox', standardize=False)
datos_gauss = gaussian_scaler.fit_transform(datos.iloc[:,np.arange(0,4)])
import matplotlib.pyplot as plt
plt.hist(datos.iloc[:, 0])
plt.hist(datos_gauss[:, 0])
plt.show()

#Normalizar: normalize
from sklearn import preprocessing
datos_normalized = preprocessing.normalize(datos.iloc[:,list(range(0,4))], norm='l2')

import matplotlib.pyplot as plt
plt.hist(datos.iloc[:, 0])
plt.hist(datos_scaled[:, 0])
plt.hist(datos_normalized[:, 0])
plt.show()

#Discretizar: KBinsDiscretizer + fit_transform
X = np.array([[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]])
est = preprocessing.KBinsDiscretizer(n_bins=2, encode='ordinal')
est.fit_transform(X)

#Binarizar: Binarizer + fit_transform
X = np.array([[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]])
est = preprocessing.Binarizer(threshold=1.1)
est.fit_transform(X)

#Polinomizar: PolynomialFeatures + fit_transform
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
#The features of X have been transformed from [x1, x2] to [1, x1, x2, x1^2, x2^2, x1*x2]
poly2 = PolynomialFeatures(2)
poly2.fit_transform(X)
#The features of X have been transformed from [x1, x2] to [1, x1, x2, x2x3]
poly2 = PolynomialFeatures(2, interaction_only=True)
poly2.fit_transform(X)


#%% Quitar variables con poca variancia
# library(caret)
# preProcValues <- preProcess(iris[,-5], method = "nzv")
# trainTransformed <- predict(preProcValues, iris[,-5])

# #Quitar variables sin varianza
# library(caret)
# preProcValues <- preProcess(iris[,-5], method = "zv") 
# trainTransformed <- predict(preProcValues, iris[,-5])

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

#With SVMs and logistic-regression, the parameter C controls the sparsity: 
#the smaller C the fewer features selected. With Lasso, the higher the alpha parameter, 
#the fewer features selected.



#%%Downsampling
# iris$objetivo=1
# iris$objetivo[1:10]=2
# iris$objetivo=as.factor(iris$objetivo)
# down_train <- downSample(x = iris[, -ncol(iris)],
#                          y = iris$objetivo)
import pandas as pd
import sklearn
iris=sklearn.datasets.load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])

datos=datos.iloc[40:]
count_class_0, count_class_1, count_class2 = datos.Species.value_counts()

# Divide by class
df_class_0 = datos[datos['Species'] == 'versicolor']
df_class_1 = datos[datos['Species'] == 'virginica']
df_class_2 = datos[datos['Species'] == 'setosa']

df_class_0_under = df_class_0.sample(count_class2)
df_class_1_under = df_class_1.sample(count_class2)
undersampled = pd.concat([df_class_0_under, df_class_1_under, df_class_2], axis=0)

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
#https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
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

#%%UpSampling
# down_train <- upSample(x = iris[, -ncol(iris)],
#                          y = iris$objetivo)
import pandas as pd
import sklearn
iris=sklearn.datasets.load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])

datos=datos.iloc[40:]
count_class_0, count_class_1, count_class2 = datos.Species.value_counts()

# Divide by class
df_class_0 = datos[datos['Species'] == 'versicolor']
df_class_1 = datos[datos['Species'] == 'virginica']
df_class_2 = datos[datos['Species'] == 'setosa']
df_class_2_over = df_class_2.sample(count_class_1, replace=True)
oversampled = pd.concat([df_class_2_over, df_class_1, df_class_2], axis=0)

#%% Otro metodo
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
# library(DMwR)
# smote_train <- SMOTE(objetivo ~ ., data  = iris)                         
# table(smote_train$objetivo)

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

#%% Kernel PCA
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot results

plt.figure()
plt.subplot(2, 2, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# projection on the first principal component (in the phi space)
Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

plt.subplot(2, 2, 2, aspect='equal')
plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.subplot(2, 2, 3, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.subplot(2, 2, 4, aspect='equal')
plt.scatter(X_back[reds, 0], X_back[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_back[blues, 0], X_back[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.tight_layout()
plt.show()







#%% Transform target in regression

import numpy as np
from sklearn.datasets import load_boston
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = load_boston(return_X_y=True)
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor,
                                  transformer=transformer)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regr.fit(X_train, y_train)

print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))

raw_target_regr = LinearRegression().fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))


# Tambien se pueden hacer transformaciones propias
def func(x):
    return np.log(x)
def inverse_func(x):
    return np.exp(x)

regr = TransformedTargetRegressor(regressor=regressor,
                                  func=func,
                                  inverse_func=inverse_func)
regr.fit(X_train, y_train)

print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))



#%% Feature Union
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()

X, y = iris.data, iris.target

# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)
print("Combined space has", X_features.shape[1], "features")

svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)


#%% Column Transformer

import pandas as pd
X = pd.DataFrame(
    {'city': ['London', 'London', 'Paris', 'Sallisaw'],
     'title': ["His Last Bow", "How Watson Learned the Trick",
               "A Moveable Feast", "The Grapes of Wrath"],
     'expert_rating': [5, 3, 4, 5],
     'user_rating': [4, 5, 4, 3]})

from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
column_trans = make_column_transformer(
    (OneHotEncoder(), ['city']),
    (CountVectorizer(), 'title'),
    remainder='drop')
X_new = column_trans.fit_transform(X)

X_new

#Otro ejemplo

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data from https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

numeric_features = ['age', 'fare']
categorical_features = ['embarked', 'sex', 'pclass']

cnts_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])
categ_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess_pipeline = ColumnTransformer([
    ('continuous', cnts_pipeline, numeric_features),
    ('cat', categ_pipeline, categorical_features)
    ])  ##remainder is used to get all the columns irrespective of transormation happened or not


#Recuperar los nombres
def get_transformer_feature_names(columnTransformer):

    output_features = []

    for name, pipe, features in columnTransformer.transformers_:
        if name!='remainder':
            for i in pipe:
                trans_features = []
                if hasattr(i,'categories_'):
                    trans_features.extend(i.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)

    return output_features


X_train_processed = pd.DataFrame(preprocess_pipeline.fit_transform(X), 
             columns=get_transformer_feature_names(preprocess_pipeline))

#PAra aplicar una transformation propia
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data from https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

numeric_features = ['age', 'fare']

def funcion():
    return np.log1p

cnts_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('log', FunctionTransformer(np.log1p))
     
])

preprocess_pipeline = ColumnTransformer([
    ('continuous', cnts_pipeline, numeric_features)
    ])

X_train_processed = preprocess_pipeline.fit_transform(X)

X_train_processed

#%% Ejemplo importancia escalar

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)

# Code source: Tyler Lanigan <tylerlanigan@gmail.com>
#              Sebastian Raschka <mail@sebastianraschka.com>

# License: BSD 3 clause

RANDOM_STATE = 42
FIG_SIZE = (10, 7)


features, target = load_wine(return_X_y=True)

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

# Fit to data and predict using pipelined GNB and PCA.
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
pred_test = unscaled_clf.predict(X_test)

# Fit to data and predict using pipelined scaling, GNB and PCA.
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

# Show prediction accuracies in scaled and unscaled data.
print('\nPrediction accuracy for the normal test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

print('\nPrediction accuracy for the standardized test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

# Extract PCA from pipeline
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# Show first principal components
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Use PCA without and with scale on X_train data for visualization.
X_train_transformed = pca.transform(X_train)
scaler = std_clf.named_steps['standardscaler']
X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

# visualize standardized vs. untouched dataset with PCA performed
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)


for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train_transformed[y_train == l, 0],
                X_train_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std_transformed[y_train == l, 0],
                X_train_std_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

ax1.set_title('Training dataset after PCA')
ax2.set_title('Standardized training dataset after PCA')

for ax in (ax1, ax2):
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()

plt.show()


#%% Distribuciones

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# `normed` is being deprecated in favor of `density` in histograms
if LooseVersion(matplotlib.__version__) >= '2.1':
    density_param = {'density': True}
else:
    density_param = {'normed': True}

# ----------------------------------------------------------------------
# Plot the progression of histograms to kernels
np.random.seed(1)
N = 20
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# histogram 1
ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', **density_param)
ax[0, 0].text(-3.5, 0.31, "Histogram")

# histogram 2
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', **density_param)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# tophat KDE
kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# Gaussian KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

for axi in ax.ravel():
    axi.plot(X[:, 0], np.full(X.shape[0], -0.01), '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)

for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')

for axi in ax[1, :]:
    axi.set_xlabel('x')

# ----------------------------------------------------------------------
# Plot all available kernels
X_plot = np.linspace(-6, 6, 1000)[:, None]
X_src = np.zeros((1, 1))

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)


def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x

for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
                            'exponential', 'linear', 'cosine']):
    axi = ax.ravel()[i]
    log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
    axi.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
    axi.text(-2.6, 0.95, kernel)

    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    axi.yaxis.set_major_locator(plt.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

ax[0, 1].set_title('Available Kernels')

# ----------------------------------------------------------------------
# Plot a 1D density example
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')
colors = ['navy', 'cornflowerblue', 'darkorange']
kernels = ['gaussian', 'tophat', 'epanechnikov']
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
            linestyle='-', label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()