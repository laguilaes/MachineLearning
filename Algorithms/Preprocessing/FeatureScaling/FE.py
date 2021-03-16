from sklearn.datasets import load_iris
iris = load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])

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