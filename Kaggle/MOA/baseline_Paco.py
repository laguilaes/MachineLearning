# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:10:49 2020

@author: PACO
"""
# Importamos librerias
import os
import csv
import pandas as pd
import numpy as np
import sys
import lightgbm as lgb
from sklearn.model_selection import KFold

# Definimos directorio de trabajo
os.getcwd()
os.chdir('C:\\Users\\PACO\\Desktop\\PROYECTOS\REGRESSION\lish-moa')

# Leemos datos brutos
TRAIN_DTYPES={"cp_type": "category", "cp_dose": "category"}

train_targets_scored      = pd.read_csv("train_targets_scored.csv")
train_targets_non_scored  = pd.read_csv("train_targets_nonscored.csv")
train_features            = pd.read_csv("train_features.csv", dtype = TRAIN_DTYPES)
test_features             = pd.read_csv("test_features.csv")
sample_submission         = pd.read_csv("sample_submission.csv")

# Funciones auxiliares
def reduce_size(df):
    
    for c in df.columns:
        if df[c].dtypes == "float64":
            df[c] = df[c].astype(np.float16)

# EDA
print ('DATOS DE ENTRENAMIENTO')
print('El dataset del train cuenta con',train_features.shape[0],'filas y ',train_features.shape[1],' columnas')
print('El dataset del test cuenta con',test_features.shape[0],'filas y ',test_features.shape[1],' columnas')

print ('VARIABLES OBJETIVO')
print('El dataset objetivo cuenta con',train_targets_scored.shape[0],'filas y ',train_targets_scored.shape[1],' columnas')



# Tratamiento de datasets

# -- reducir tamaño 
reduce_size(train_features)
reduce_size(test_features)
reduce_size(sample_submission)

# -- Meter columnas de la variable objetivo como una unica variable en train
train_targets_melted = pd.melt(train_targets_scored, id_vars=['sig_id'], var_name='Variable', value_name='valor').reset_index()
train_features = pd.merge(train_features, train_targets_melted, on = "sig_id", how = 'outer')
del(train_features["index"])
train_features.dtypes
cat_feats =['cp_type','cp_dose','Variable']
for col in cat_feats:
     train_features[col] = train_features[col].astype("category").cat.codes.astype("int16")
     train_features[col] -= train_features[col].min()
    
           

# #dummy vars
# y_train = train_features["valor"]
# del(train_features["valor"])
# # train_features = pd.concat([train_features['sig_id'], pd.get_dummies(train_features.drop("sig_id", axis = 1))], axis = 1)
for col, col_dtype in train_features.items():
    if col_dtype == "category" or col== 'Variable' and col!='sig_id':
        train_features[col] = train_features[col].cat.codes.astype("int16")
        train_features[col] -= train_features[col].min()
            
cat_feats =['cp_type','cp_dose','Variable']#list(train_targets_non_scored.columns[1:])
useless_cols = ["sig_id","index"]
train_cols = train_features.columns[~train_features.columns.isin(useless_cols)]
X_train = train_features[train_cols]
y_train = train_features["valor"]
del(X_train["valor"])

np.random.seed(767)

fake_valid_inds = np.random.choice(X_train.index.values, 1_000_000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 
                         categorical_feature=cat_feats, free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],
                              categorical_feature=cat_feats,
                 free_raw_data=False)



params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.073,
#         "sub_feature" : 0.8,
        "sub_row" : 0.73,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
    "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 140,
    'num_leaves': 124,
    "min_data_in_leaf": 100,
}


m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) 

m_lgb.save_model("model_baseline.lgb")
m_lgb.save_model('model.txt')

bst = lgb.Booster(model_file="model.txt")

test_targets_melted = pd.melt(sample_submission, id_vars=['sig_id'], var_name='Variable', value_name='valor').reset_index()
test_features = pd.merge(test_features, test_targets_melted, on = "sig_id", how = 'outer')

del(test_features["index"])
test_features.dtypes
indentfiers=test_features[['sig_id','Variable']]
cat_feats =['cp_type','cp_dose','Variable']
for col in cat_feats:
     test_features[col] = test_features[col].astype("category").cat.codes.astype("int16")
     test_features[col] -= test_features[col].min()



cat_feats =['cp_type','cp_dose','Variable']#list(train_targets_non_scored.columns[1:])
useless_cols = ["sig_id","index"]
test_cols = test_features.columns[~test_features.columns.isin(useless_cols)]
X_train = test_features[test_cols]
del(X_train["valor"])

Pkl_Filename = "bs_model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(LR_Model, file)


pred=m_lgb.predict(X_train)

l=pd.concat([indentfiers,pd.DataFrame(pred)], axis=1)
l.rename(columns={'sig_id':'sig_id',
                        'Variable':'Variable',
                        0:'pred'},
               inplace=True)

l.pred=np.clip(l.pred,0,1)
test_targets_melted = pd.melt(sample_submission, id_vars=['sig_id'], var_name='Variable', value_name='valor').reset_index()



spread=pd.pivot_table(l, values='pred',index='sig_id', columns='Variable').reset_index()

sample_submission.columns==sub.columns

sub=spread[sample_submission.columns]


sub.to_csv('bs_sub_1.csv',index=False)
os.

dfs = np.array_split(train_targets_non_scored, 2)
for i in dfs:
    print(i.shape)


np.split(train_targets_non_scored,)
