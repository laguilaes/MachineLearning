# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:10:49 2020

@author: PACO
"""
# Importamos librerias
import os                                        # Funciones del sistema operativo
import csv                                       # Tratar con archivos de datos csv
import pandas as pd                              # Trabajar con dataframes (Rstudio)
import numpy as np                               # Trabaja con matrices
import lightgbm as lgb                           # LightGBM
from sklearn.model_selection import KFold        # Cross-Validation
import psutil                                    # Averiguar número de nucleos


# Funciones auxiliares
def reduce_size(df):
    
    for c in df.columns:
        if df[c].dtypes == "float64":
            df[c] = df[c].astype(np.float16)

    
def treatment_training(train_features):
    reduce_size(train_features) # Reducimos tamaño
    train_features=train_features.loc[train_features.cp_type!="ctl_vehicle",] #Eliminamos ctl_vehicle para realizar el entrenamiento
    del(train_features['sig_id']) 
    train_features = pd.get_dummies(train_features)
    train_features=train_features.to_numpy()
    
    return train_features

def treatment_test(test_features):
     reduce_size(test_features) # Reducimos tamaño
     identifiers=test_features['sig_id'].copy()
     del(test_features['sig_id'])
     test_features = pd.get_dummies(test_features)
     test_features=test_features.to_numpy()
     
     return test_features,identifiers
 
def training_lightgbm(X_train,y_train,params):
    fake_valid_inds = np.random.choice(range(X_train.shape[0]), round(0.15*X_train.shape[0]), replace = False)
    train_inds = np.setdiff1d(range(X_train.shape[0]), fake_valid_inds)
    train_data = lgb.Dataset(X_train[train_inds,:] , label = y_train[train_inds], free_raw_data=False)
    fake_valid_data = lgb.Dataset(X_train[fake_valid_inds,:], label = y_train[fake_valid_inds],free_raw_data=False)
    m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) 
    return m_lgb

def predictor(test_features):
    pred=pd.DataFrame(m_lgb.predict(test_features))
    return pred

#################################
#             INDEX             #
#################################
# 1. GENERAL PARAMETERS
# 2. READING DATA
# 3. TREATING DATA (TRAINING AND TEST)
# 4. DEFINING PARAMETERS
# 5. PREDICTION LOOP
# 6. PREPARING SUBMISSION
#################################


########
# 1. GENERAL PARAMETERS
########
EJECUCION     = 'NO KAGGLE'
SEED          = 766
PATH          = 'C:\\Users\\PACO\\Desktop\\PROYECTOS\REGRESSION\lish-moa'
TRAIN_DTYPES  = {"cp_type": "category", "cp_dose": "category"}
nucleos       = psutil.cpu_count()



########
# 2. READING DATA
########
if EJECUCION=='KAGGLE':
    train_targets_scored      = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
    train_targets_non_scored  = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
    train_features            = pd.read_csv("/kaggle/input/lish-moa/train_features.csv", dtype = TRAIN_DTYPES)
    test_features             = pd.read_csv("/kaggle/input/lish-moa/test_features.csv", dtype = TRAIN_DTYPES)
    sample_submission         = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

else: 
    os.chdir(PATH)
    train_targets_scored      = pd.read_csv("train_targets_scored.csv")
    train_targets_non_scored  = pd.read_csv("train_targets_nonscored.csv")
    train_features            = pd.read_csv("train_features.csv", dtype = TRAIN_DTYPES)
    test_features             = pd.read_csv("test_features.csv")
    sample_submission         = pd.read_csv("sample_submission.csv")

########
# 3. TREATING DATA (TRAINING AND TEST)
########
train_features=treatment_training(train_features)
test_features,identifiers=treatment_test(test_features)




########
# 4. DEFINING PARAMETERS
########
params = {
        "objective" : "poisson",
        "metric" :"binary_logloss",
        "force_row_wise" : True,
        "learning_rate" : 0.073,
        "sub_feature" : 0.8,
        "sub_row" : 0.73,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "nthread" : nucleos,
        'verbosity': 1,
        'num_iterations' :20,
        'num_leaves': 124,
        "min_data_in_leaf": 100,
}


########
# 5. PREDICTION LOOP
########
tot=pd.DataFrame()
for i in range(train_targets_scored.shape[1]-1):
    i=i+1
    
    # 5.1. Choosing target variable
    y_train=train_targets_scored.iloc[:,i]
    if sum(y_train)>5:
    # 5.2. Training model
        m_lgb = training_lightgbm(train_features,y_train,params)
    # 5.3. Predicting
        pred=predictor(test_features)
    # 5.4. Renaming variables
        pred=pred.rename(columns={0:train_targets_scored.columns[i]})
    else: # case that the variable to predict is practically zero in all the rows
        pred=pred.rename(columns={train_targets_scored.columns[i-1]:train_targets_scored.columns[i]})
        pred.iloc[:,0]=0.000012*(1+sum(y_train)/100)
    # 5.5. concatenating predictions
    tot= pred if i==1 else pd.concat([tot,pred],axis=1)
    
    print(i)


########
# 6. PREPARING SUBMISSION
########
sub=pd.concat([identifiers,tot],axis=1)
sub=sub[sample_submission.columns]
sub.to_csv('submission.csv',index=False)
