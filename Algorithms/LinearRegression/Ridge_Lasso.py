import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data
data = pd.read_csv("D:/MachineLearning/Algorithms/LinearRegression/Advertising.csv")
print(data.head())
data.drop(["Unnamed: 0"], axis=1, inplace=True)

#Visualize data
def scatter_plot(feature, target):
    plt.figure(figsize=(16, 18))
    plt.scatter(data[feature],
                data[target],
                c='black'
                )
    plt.xlabel("Money Spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()
scatter_plot("TV", "Sales")
scatter_plot("Radio", "Sales")
scatter_plot("Newspaper", "Sales")

#Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

xs = data.drop(["Sales"], axis=1)
y = data["Sales"].values.reshape(-1,1)
linreg = LinearRegression()
MSE = cross_val_score(linreg, xs, y, scoring="neg_mean_squared_error", cv=5)

mean_MSE = np.mean(MSE)
print(mean_MSE)

#Ridge regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()

parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regression = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regression.fit(xs, y)

print(ridge_regression.best_params_)
print(ridge_regression.best_score_)

#Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()

parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regression = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regression.fit(xs, y)

print(lasso_regression.best_params_)
print(lasso_regression.best_score_)







#%%
#%% Ridge Regression #Muchos predictores y pocas observaciones: Usado cuando todas las variables aportan algo
# alpha{float, ndarray of shape (n_targets,)}, default=1.0
# fit_interceptbool, default=True
# normalizebool, default=False
# copy_Xbool, default=True
# max_iterint, default=None
# tolfloat, default=1e-3
# solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}, default=’auto’
import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.alpha_
reg.intercept_, 
reg.coef_

reg2 = linear_model.Ridge(alpha=.5)
reg2.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg2.coef_
reg2.intercept_
reg2.n_iter_


#%% Lasso Regression: Muchos predictores y pocas observaciones:uando hay algunas variables inutiles totalmente
#Lasso: alpha, fit_intercept, normalize
#LassoCV: eps (alpha_min/alpha_max), n_alphas (numero alphas), fit_intercept, normalize, cv
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
X, y = make_regression(noise=4, random_state=0)

plt.scatter(X[:,0], y,  color='black')
plt.show()

reg = LassoCV(cv=5, random_state=0).fit(X, y)
reg.score(X, y)

pred=reg.predict(X[:,])

plt.scatter(X[:,0], y,  color='black')
plt.scatter(X[:,0], pred,  color='red')
plt.show()

#%%Elastic Net
#ElasticNet: l1_ratio (ratio l1-l2), alpha, fit_intercept, normalize
#ElasticNetCV: l1_ratio (ratio l1-l2), n_alphas, fit_intercept, normalize, eps(alpha_min/alpha_max)
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
X, y = make_regression(n_features=2, random_state=0)
regr = ElasticNetCV(cv=5, random_state=0)
regr.fit(X, y)
pred=regr.predict(X[:,])

print(regr.alpha_)
print(regr.intercept_)

plt.scatter(X[:,0], y,  color='black')
plt.scatter(X[:,0], pred,  color='red')
plt.show()

#%% Least Angle Regression LARS:
#Lars:fit_intercept, verbose, normalize
#LarsCV: fit_intercept, verbose, normalize, cv

from sklearn.linear_model import LarsCV, Lars
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
reg = LarsCV(cv=5).fit(X, y)
reg.score(X, y)
reg.alpha_
pred=reg.predict(X[:,])

plt.scatter(X[:,0], y,  color='black')
plt.scatter(X[:,0], pred,  color='red')
plt.show()

reg2 = Lars().fit(X, y)
reg2.score(X, y)
reg2.alpha_
pred=reg2.predict(X[:,])

#%% LassoLars: alpha, fit_intercept, normalize
#LassoLarsCV: alpha, fit_intercept, normalize, cv
from sklearn import linear_model
reg = linear_model.LassoLars(alpha=0.01)
reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])

print(reg.coef_)

reg2 = linear_model.LassoLarsCV()
reg2.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
