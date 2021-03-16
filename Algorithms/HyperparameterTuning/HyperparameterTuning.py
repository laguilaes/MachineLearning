#%%Hyperparameter Tuning

#Grid Search
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10]}
svc = svm.SVC()

clf = GridSearchCV(svc, parameters, cv=3, scoring = "accuracy")
clf.fit(iris.data, iris.target)
clf.cv_results_
clf.best_params_




#%%
#%%RandomizedSearch
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
iris = load_iris()
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,random_state=0)
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
clf = RandomizedSearchCV(logistic, distributions, random_state=1,n_iter = 100, n_jobs=-1)
search = clf.fit(iris.data, iris.target)
search.best_params_



#%%
#%%
from optuna import *
def fit_lgbm(trial, X_train, y_train,  X_test, y_test, seed=None, cat_features=cat_feats):
    """Train Light GBM model"""
 
    params = {
    'num_threads': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'learning_rate': trial.suggest_uniform('learning_rate', 0.05, 0.3),
    'num_leaves': trial.suggest_int('num_leaves', 100, 200), 
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 150),
    'num_iteration': 300, 
    'verbose': 1,
    'metric': "rmse",
    'lambda_l2': trial.suggest_uniform('lambda_l2', 0, 0.1)
    }
   

    params['seed'] = 13

    early_stop = 5
    verbose_eval = 20

    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    d_valid = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features)
    watchlist = [d_train, d_valid]

    print('Training model:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=100,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    # predictions
    y_pred_valid = model.predict(X_test, num_iteration=model.best_iteration)
    
    print('best_score', model.best_score)
    log = {'train/rmse': model.best_score['training']['rmse'],
           'valid/rmse': model.best_score['valid_1']['rmse']}
    return model, y_pred_valid, log

def objective(trial: Trial):
    models = []
    model, y_pred_valid, log = fit_lgbm(trial, X_train, y_train,  X_test, y_test)
    models.append(model)
    valid_score = log["valid/rmse"]
    gc.collect()
    return valid_score

study = create_study()
study.optimize(objective, n_trials=10)


print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
#Best trial: score 2.186941436074196, params {'learning_rate': 0.1284288977810651, 'num_leaves': 186, 'min_data_in_leaf': 130, 'lambda_l2': 0.037660303077688945}
study.trials_dataframe()

import plotly
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_intermediate_values(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study)
optuna.visualization.plot_parallel_coordinate(study)




#%%
#%% OPTIMIZACION PARAMETROS SKOPT


def f(x):
    print(x)
    params = {
        'num_threads': 8,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'poisson',
        'learning_rate': x[0],
        'num_leaves': x[1], 
        'min_data_in_leaf': x[2],
        'num_iteration': x[3], 
        'max_bin': x[4],
        'verbose': 1,
        'metric': "rmse",
        'lambda_l2': x[5]
    }
    
    gbm = lgb.LGBMRegressor(**params)

    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5) 
    
    print('score: ', mean_squared_error(gbm.predict(X_test), y_test))
    
    return mean_squared_error(gbm.predict(X_test), y_test)

# optimize params in these ranges
spaces = [
    (0.05, 0.10), #learning_rate.
    (100, 150), #num_leaves.
    (50, 150), #min_data_in_leaf
    (280, 300), #num_iteration
    (200, 220), #max_bin
    (0, 0.1) #max depth
    ]

# run optimization
from skopt import gp_minimize #bayesian optimization
res = gp_minimize(
    f, spaces,
    acq_func="EI", #expected improvement
    n_calls=20) # increase n_calls for more performance

# print tuned params
print(res.x)

# plot tuning process
from skopt.plots import plot_convergence
plot_convergence(res)