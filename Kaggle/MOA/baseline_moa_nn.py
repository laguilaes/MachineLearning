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
import tensorflow.keras as tf
import tensorflow_addons as tfa
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
    test_features             = pd.read_csv("test_features.csv", dtype = TRAIN_DTYPES)
    sample_submission         = pd.read_csv("sample_submission.csv")

########
# 3. TREATING DATA (TRAINING AND TEST)
########
train_features             =treatment_training(train_features)
test_features,identifiers  =treatment_test(test_features)


########
# 4. DEFINING NN
########
def NN_Model(model, X_train, y_train):
    
    fake_valid_inds = np.random.choice(range(X_train.shape[0]), round(0.15*X_train.shape[0]), replace = False)
    train_inds = np.setdiff1d(range(X_train.shape[0]), fake_valid_inds)
    train_data , train_target = X_train[train_inds,:]   , y_train[train_inds]
    valid_data , valid_target = X_train[fake_valid_inds,:] , y_train[fake_valid_inds]
    
    
    model.add(tf.layers.BatchNormalization())
    

    model.add(tfa.layers.WeightNormalization(tf.layers.Dense(876, activation='relu')))
    model.add(tf.layers.BatchNormalization())
    model.add(tf.layers.Dropout(0.7))

    
    model.add(tfa.layers.WeightNormalization(tf.layers.Dense(876, activation='relu')))
    model.add(tf.layers.BatchNormalization())
    model.add(tf.layers.Dropout(0.6))

    model.add(tfa.layers.WeightNormalization(tf.layers.Dense(512, activation='relu')))
    model.add(tf.layers.BatchNormalization())

    
    model.add(tf.layers.Dense(206, activation='sigmoid'))
    
    model.compile(optimizer='adam'
                  ,loss='binary_crossentropy'
                 )

    #model.summary()

    model.fit(train_data, train_target, epochs=2, validation_data=(valid_data, valid_target))

########
# 5. PREDICTION LOOP
########

    
# 5.1. Choosing target variable
y_train=train_targets_scored.iloc[:,1:].values
# 5.2. Training model
model = tf.models.Sequential()
NN_Model(model, train_features, y_train)
# 5.3. Predicting
pred=model.predict(test_features)
# 5.4. Renaming variables
tot=pd.concat([identifiers,pd.DataFrame(pred)],axis=1)
tot.columns=sample_submission.columns
########
# 6. PREPARING SUBMISSION
########
tot.to_csv('submission.csv',index=False)
