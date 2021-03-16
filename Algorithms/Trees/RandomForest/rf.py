#%% RANDOM FOREST 
#Bootstrap(samples, sqrt(features)) +  decision tree + aggregating(voting, mean...) (BAgging = Bootstrapping + Agregating data)
#OOB error(out of bag error) es el error al clasificar registros no usados en bagging

#Missing values in train
#Inicial: categorical: igual que la variable objetivo. Numerical: mean
#Refinamiento: Run decision tree para cada registro. Repetir en cada tree. 
#Matriz de similitud: registros similares en mismas hojas. Normalizar entre numero arboles

#Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))

#Regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

print(regr.feature_importances_)

print(regr.predict([[0, 0, 0, 0]]))