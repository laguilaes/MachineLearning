# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 10:49:09 2020

@author: laguila
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

os.chdir("C:/Users\laguila\Desktop\MOA")
pd.set_option('display.max_columns', 50)

#%%
def reduce_size(df):
    
    for c in df.columns:
        if df[c].dtypes == "float64":
            df[c] = df[c].astype(np.float16)




#%%
#Lectura de archivos
train_targets_scored = pd.read_csv("train_targets_scored.csv")
train_targets_non_scored = pd.read_csv("train_targets_nonscored.csv")
train_features = pd.read_csv("train_features.csv")
test_features = pd.read_csv("test_features.csv")
sample_submission = pd.read_csv("sample_submission.csv")

#reducir tamaño 
reduce_size(train_features)
reduce_size(test_features)

#Meter columnas de la variable objetivo como una unica variable en train/test
train_targets_melted = pd.melt(train_targets_scored, id_vars=['sig_id'], var_name='Variable', value_name='valor').reset_index()
le = preprocessing.LabelEncoder()
le.fit(train_targets_melted.Variable)
train_targets_melted.Variable = le.transform(train_targets_melted.Variable)
train_features = pd.merge(train_features, train_targets_melted, on = "sig_id", how = 'outer')

test_features_melted = pd.melt(sample_submission, id_vars=['sig_id'], var_name='Variable', value_name='valor').reset_index()
le = preprocessing.LabelEncoder()
le.fit(test_features_melted.Variable)
test_features_melted.Variable = le.transform(test_features_melted.Variable)
test_features = pd.merge(test_features, test_features_melted, on = "sig_id", how = 'outer')

#dummy vars
train_features.set_index('sig_id', inplace = True)
train_features = pd.get_dummies(train_features)

test_features.set_index('sig_id', inplace = True)
test_features = pd.get_dummies(test_features)


#Crear train y test para entrenar
y = train_features.valor
del(train_features["valor"])
X_train, X_test, y_train, y_test = train_test_split(train_features, y, random_state=0)


