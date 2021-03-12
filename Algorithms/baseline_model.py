
import pandas as pd
import datatable as dt
import numpy as np
import datetime
import os
import random
import gc

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.rcParams.update({'font.size':20})
import seaborn as sns

import neptune


os.chdir('C:/Users/laguila/Google Drive/Programacion/Python')

'''
IDEAS
Modelo EDA
Modelo Preds
Regresion, clasificacion, imagenes, LSTM


'''

#%% Feature engineering

def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2 #memoria del dataframe inicial
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    gc.collect()
    return df

def submission_vars(data, var):
    df = data.copy()
    for v in var:
        df[v.upper()] = df[v]
    return df
        

def binaryToDecimal(binary): 
      
    decimal, i = 0, 0
    while(binary != 0): 
        dec = binary % 10
        decimal = decimal + dec * pow(2, i) 
        binary = binary//10
        i += 1
    return decimal
    
def label_encoder(data, variables, return_encoder = False):
    
    from sklearn.preprocessing import LabelEncoder
    df = data.copy()
    
    for variable in variables:
        label_encoder = LabelEncoder()
        df[variable] = label_encoder.fit_transform(df[variable].astype(str))
        
    if not return_encoder:
        return df
    else:
        return df, label_encoder


def binary_encoder(data, variables):
    df = data.copy()
    
    for variable in variables:
        df = label_encoder(df, [variable])
        aux = df[variable].map(lambda x: bin(x)[2:])
        max_len = max(aux.map(lambda x: len(x)))
        aux = aux.map(lambda x: '0' * (max_len - len(x)) + x)
        
        for i in range(max_len):
            df[variable + f"_{i}"] = aux.map(lambda x: int(x[i]))
            
        df.drop(variable, axis=1, inplace=True)
    
    return df

def dummies_encoder(data, variables):
    df = data.copy()
    
    for variable in variables:
        dummies = pd.get_dummies(df[variable])
        df = pd.concat([df, dummies], axis=1)
        df.drop(variable, axis=1, inplace=True)
    return df

def target_encoder(data, variables, target, m):
    df = data.copy()
    
    for variable in variables:
        mean = df[target].mean()
        agg = df.groupby(variable)[target].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        smooth = (counts * means + m * mean) / (counts + m)
        df[variable] = df[variable].map(smooth)
    return df
        
def scaler(data, submission_feat, exclude, scale):
    
    if scale != "no":
        from sklearn.preprocessing import MinMaxScaler
        
        df = data.copy()
        features = list(set(df.select_dtypes(include=["uint8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns) - set(exclude) - set([feat.upper() for feat in submission_feat]))
        no_action= list(set(df.columns) - set(features))
        sc = MinMaxScaler(feature_range = (0,1))
        values = sc.fit_transform(df[features])
        df_scaled = pd.DataFrame(data = values, columns = df[features].columns,
                                 index = df[features].index)
        df_scaled[no_action] = df[no_action]
    else:
        df_scaled = data.copy()
        sc = "not_scaled"
        features = "not_scaled"
    gc.collect()    
    return df_scaled, sc, features

def to_gaussian(data, submission_feat, exclude, gauss):
    
    if gauss != "no":
        from sklearn.preprocessing import PowerTransformer
        
        df = data.copy()
        features = list(set(df.select_dtypes(include=["uint8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns) - set(exclude) - set([feat.upper() for feat in submission_feat]))
        no_action= list(set(df.columns) - set(features))
        pt = PowerTransformer(method='yeo-johnson')
        values = pt.fit_transform(df[features])
        df_gaussian = pd.DataFrame(data = values, columns = df[features].columns,
                                 index = df[features].index)
        df_gaussian[no_action] = df[no_action]
    else:
        df_gaussian = data.copy()
        pt = "No PowerTransformer applied"
        features = pt
    gc.collect()
    return df_gaussian, pt, features

def best_features(data, target, submission_feat, num):
    
    if num>0:
        from sklearn.feature_selection import SelectKBest, SelectPercentile
        
        df = data.dropna().copy()
        y = df[target]
        df = df.drop(target, axis=1)
        features = list(set(df.select_dtypes(include=["uint8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns) - set([target]) - set([feat.upper() for feat in submission_feat]))
        no_action= list(set(df.columns) - set(features))
        
        if num>=1:
            X_new = SelectKBest(k=num)
        else:
            X_new = SelectPercentile(percentile = 100*num)
        
        values = X_new.fit_transform(df[features], y)
        df_k = pd.DataFrame(data = values, columns = df.iloc[:,X_new.get_support(indices=True)].columns,
                                 index = df[features].index)
        df_k[no_action] = df[no_action]
        df_k[target] = y
        gc.collect()
        return data[df_k.columns]
    else:
        return data
        
def proccess_nas(data, target, submission_feat, method):
    
    if method != "no":
    
        df = data.copy()
        
        features = list(set(df.select_dtypes(include=["uint8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns) - set([target]) - set([feat.upper() for feat in submission_feat]))
        no_action= list(set(df.columns) - set(features))
        
        if method == "omit":
            df_imputed = df.dropna(subset = list(features), how = "any")
        
        else:
            if type(method) == str:                   
                if method == "knn":
                    from sklearn.impute import KNNImputer
                    imp = KNNImputer(n_neighbors=6, weights="uniform")
                    values = imp.fit_transform(df[features])
                        
                elif (method ==  "mean") | (method == "most_frequent"):
                    from sklearn.impute import SimpleImputer
                    imp = SimpleImputer(strategy=method)
                    values = imp.fit_transform(df[features])
                else:
                    values = df[features].fillna(method = method)
                    
            else:
                values = df[features].fillna(method)
                
            df_imputed = pd.DataFrame(data = values, columns = df[features].columns,
                                         index = df[features].index)
            df_imputed[no_action] = df[no_action]
    else:
        df_imputed = data.copy()
    gc.collect()
        
    return df_imputed 
            
def perform_pca(data, target, submission_feat, k):
    
    if k != 0:
        from sklearn.decomposition import PCA
        
        df = data.copy()
        
        features = list(set(df.select_dtypes(include=["uint8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns) - set([target]) - set([feat.upper() for feat in submission_feat]))
        no_action= list(set(df.columns) - set(features))
        
        if k>1:
            df_pca = pd.DataFrame(data = PCA(n_components=k).fit_transform(df[features])) 
        else:
            df_pca = pd.DataFrame(data = PCA(k).fit_transform(df[features]))
        
        df_pca.columns = [f"PC_{n+1}" for n in range(df_pca.shape[1])]
        df_pca[no_action] = df[no_action]
        
    else:
        df_pca = data.copy()
    gc.collect()    
    return df_pca

def perform_lda(data, target, submission_feat, k):
    
    if k != 0:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        df = data.dropna().copy()
        y = df[target]
        features = list(set(df.select_dtypes(include=["uint8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns) - set([target]) - set([feat.upper() for feat in submission_feat]))
        no_action= list(set(df.columns) - set(features))
    
        values = LinearDiscriminantAnalysis(n_components=k).fit_transform(df[features], y)
    
        df_lda = pd.DataFrame(data = values, columns = [f"LD_{n+1}" for n in range(values.shape[1])], index = df.index)
        df_lda[no_action] = df[no_action]
    else:
        df_lda = data
    gc.collect()
    return df_lda

def create_statistics(data, stats):
    
    df = data.copy()
    if stats != "no":

        df=df.sort_values('Month_num')
        lags = [1, 2, 3, 4, 5, 6, 7, 8]
        lag_cols = [f"lag_{lag}" for lag in lags ]
    
        for lag, lag_col in zip(lags, lag_cols):
            df[lag_col] = df[["Brand","Country",'volume']].groupby(["Country",'Brand'])["volume"].shift(lag)
    
        wins = [2]
        for win in wins:
            for lag,lag_col in zip(lags, lag_cols):
                df[f"rmean_{lag}_{win}"] = df[["Country",'Brand', lag_col]].groupby(["Country",'Brand'])[lag_col].transform(lambda x : x.rolling(win).mean())
          
    gc.collect()    
    return df

def predict(test, models, train_cols, model_type, decoders, config):
    tst = test.copy()
    if type(models) == list:
        # if task == 'regression':
    
        #     preds = model[0].predict(tst[train_cols]) / len(model)
        #     for m in model[1:]:
        #         preds = preds + m.predict(tst[train_cols]) / len(model)
        # else:
        from statistics import mode
        all_preds = pd.DataFrame()
        for i, m in enumerate(models):
            if model_type == "keras":
                all_preds[i] = np.argmax(m.predict(tst[train_cols]), axis=1)
            elif model_type == "lgb":
                all_preds[i] = m.predict(tst[train_cols])
        
        preds = all_preds.apply(lambda x: mode(x), axis=1)
    else:
        preds = np.argmax(models.predict(tst[train_cols]),axis=1)
    
    sub_feats = config["submission_feat"]
    tst = tst[[feat.upper() for feat in sub_feats if config["target"] != feat]]
    tst[config["target"]] = decoders[0].inverse_transform(preds)

    gc.collect()    
    return tst


def kfold(X, X_y, train_cols, k, bayes, model_type, metric, config):
    
    from sklearn.model_selection import StratifiedKFold, KFold
    
    PARAMS = {"strat" : config['strat'],
          "label_encod" : config['label_encod'],
          "binary_encod" : config['binary_encod'],
          "one-hot_encod": config['oneHot_encod'],
          "target_encod" : config['target_encod'],
          "process_nas" : config['na_treat'],
          "scale":config['scale'],
          "gaussian":config['gauss'],
          "stats": config['stats'],
          "k_feat":config['k_feat'],
          "pca":config['pca'],
          "lda":config['lda'],
          "balance":config['balance'],
          "model": model_type,
          "cross_val": k if k>1 else 1,
          "bayes_opt": bayes if model_type == "lgb" else 0,
          "metric" : metric
    }
    
    neptune.create_experiment(params = PARAMS, upload_source_files = ["baseline_model.py"], tags = ["classification"])

    if k<=1:
        if k==1:
            q=0.2
        else:
            q=k
        from sklearn.model_selection import train_test_split
        if config['strat'] in X.columns:
            X_train, X_val, y_train, y_val = train_test_split(X, X_y, test_size = q, shuffle = True, stratify = X[config['strat']])
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, X_y, test_size = q, shuffle = True)
        model = fit_model(X_train, y_train, X_val, y_val, train_cols, bayes, model_type)
        models = model[0]
        history = model[1]
        
        if model_type == "keras":
            models.summary(print_fn = lambda x: neptune.log_text("model_summary", x))
  
    else:
        if config['strat'] in X.columns:
            kf = StratifiedKFold(n_splits=k, shuffle=True)
            y = X[config['strat']]
            models = []
            history = []
            for train_index, test_index in kf.split(X, y):
                X_train, X_val = X.iloc[train_index,:], X.iloc[test_index,:]
                y_train, y_val = X_y.iloc[train_index], X_y.iloc[test_index]
                m = fit_model(X_train, y_train, X_val, y_val, train_cols, bayes, model_type)
                models.append(m[0])
                history.append(m[1])
        else:
            
            kf = KFold(n_splits=k, shuffle=True)
            models = []
            history = []
            for train_index, test_index in kf.split(X):
                X_train, X_val = X.iloc[train_index,:], X.iloc[test_index,:]
                y_train, y_val = X_y.iloc[train_index], X_y.iloc[test_index]
                m = fit_model(X_train, y_train, X_val, y_val, train_cols, bayes, model_type)
                models.append(m[0])
                history.append(m[1])
        if model_type == "keras":
            m[0].summary(print_fn = lambda x: neptune.log_text("model_summary", x))
  
    gc.collect()    
    plot_training(history, model_type, metric)
    neptune.stop()         
    return models
        

def balance_classes(X, X_y, X_strat, target, train_cols, method):
    
    y = X_strat.astype(str)
    x = X.copy()
    # x[target] = X_y
    
    if method == "over":
        from imblearn.over_sampling import RandomOverSampler
        x, y = RandomOverSampler().fit_sample(x, y)
    
    elif method == "under":
        from imblearn.under_sampling import RandomUnderSampler
        x, y = RandomUnderSampler().fit_sample(x, y)
        
    elif method == "smote":
        from imblearn.over_sampling import SMOTE      
        features = list(set(x.select_dtypes(include=["uint8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns) - set(X_strat.name))
        no_action = list(set(x.columns) - set(features))
        x_sm = x[features].copy()
        x_no_action = x[no_action].copy()
        x, y = SMOTE().fit_sample(x_sm, y)
        x[no_action] = x_no_action[no_action]
        x[X_strat.name] = y
        
    
    X = x.copy()
    X_y = x[target]
    gc.collect()    
    return X, X_y

def plot_training(models, model_type, score):
    
    if type(models) != list:
        models = [models]
    data = pd.DataFrame()
    
    if model_type == "keras":
        for i, m in enumerate(models):        
            aux_train = pd.DataFrame()
            aux_val = pd.DataFrame()
            aux_train["model_" + str(i) + "_train"] = m.history["val_loss"]
            aux_val["model_" + str(i) + "_val"] = m.history["loss"]
            data = pd.concat([data, aux_train, aux_val], axis=1)        
    
    if model_type == "lgb":
        for i, m in enumerate(models):
            aux_train = pd.DataFrame()
            aux_val = pd.DataFrame()
            aux_train["model_" + str(i) + "_train"] = m.evals_result_["valid_1"][score]
            aux_val["model_" + str(i) + "_val"] = m.evals_result_["valid_0"][score]
            data = pd.concat([data, aux_train, aux_val], axis=1)
    
    val_cols = [col for col in data.columns if "val" in col]
    train_cols = [col for col in data.columns if "train" in col]
    data["val_" + score] = data[val_cols].apply("mean", axis=1)
    data["train_" + score] = data[train_cols].apply("mean", axis=1)
    
    fig = plt.figure(figsize = (12, 8))
    
    for i in val_cols:
        plt.plot(data[i], label = i, color = "red")
    for i in train_cols:
        plt.plot(data[i], label = i, color = "blue")
                
    
    plt.plot(data["val_" + score], color = "red", linewidth = 3)  
    plt.plot(data["train_" + score], color = "blue", linewidth = 3)
    
    plt.legend(loc="best") 
    plt.title("Train and validation " + score)       
        

    for n in range(len(data["val_" + score])):
        neptune.log_metric("val_" + score, data["val_" + score][n])
        neptune.log_metric("train_" + score, data["train_" + score][n])
        
    neptune.log_image("charts", fig)
    gc.collect()            


def bayesian_opt(estimator, X_train, y_train, X_val, y_val, fit_params, params, bayes):
    from skopt import BayesSearchCV 
    opt = BayesSearchCV(estimator = estimator, search_spaces = fit_params, fit_params = params, n_jobs=-1,cv=2,n_iter=bayes) 
    model = opt.fit(X_train, y_train)
    gc.collect()
    return model

    
def fit_model(X_train, y_train, X_val, y_val, train_cols, bayes, model_type = "lgb"):
    
    if model_type == "lgb":
        from lightgbm import LGBMRegressor, LGBMClassifier
        import lightgbm as lgb
        
    
        if bayes != 0:
            
            params = { 
                'early_stopping_rounds':10, #early stopping
                'eval_set': [(X_val[train_cols], y_val), (X_train[train_cols], y_train)]
            }
                        
            fit_params = {
                    'n_estimators': (100, 500),      #number of trees.
                    # 'num_leaves': (30, 50),
                    'learning_rate': (0.01, 0.05),
                    'num_boost_round' : (3000, 3500),
                    'subsample' : (0.7, 0.75),          #part of the dataset used for training on each round
                    'reg_alpha' : (0, 0.1),         #reg alpha
                    'reg_lambda' : (0, 0.1),         #reg lambda
                    'early_stopping_rounds':(10, 11), #early stopping
                    # 'min_data_in_leaf' :  (1000, 1001),  #important to prevent overfitting
                    }

            estimator = LGBMClassifier(**params, eval_metric=metric)
            model= bayesian_opt(estimator, X_train[train_cols], y_train, X_val, y_val, fit_params, params, bayes)
            model = model.best_estimator_                

        else:
            
            params = { 
                'objective': "multiclass",
                'boosting_type': "gbdt" ,  #boosting algorithm: gbdt, dart, goss
                # 'n_estimators': 100,      #number of trees.
                # 'num_leaves' :  64,      #number of leaves, to control the complexity of the tree. Either set this parameter of max_depth
                'learning_rate': 0.01,  #learning rate
                'num_boost_round' : 100, #max number of iterations
                'subsample' : 0.7,          #part of the dataset used for training on each round
                'reg_alpha' : 0.1,         #reg alpha
                'reg_lambda' : 0.1,         #reg lambda
                'verbose_eval':20,          #verbose
                'early_stopping_rounds':10, #early stopping
                # 'min_data_in_leaf' :  1000,  #important to prevent overfitting
                'n_jobs': -1
            }
 
              
            model = LGBMClassifier(**params, eval_metric = metric)
            
            model = model.fit(X=X_train[train_cols], y=y_train, eval_set = [(X_val[train_cols], y_val), (X_train[train_cols], y_train)])
        
        for k, v in model.get_params(False).items():
            if (type(v) == int) | (type(v) == float):
                neptune.log_text(k, str(v))
            elif (type(v) == str):
                neptune.log_text(k, v)
        history = model
               
    elif model_type == "keras":
        import tensorflow as tf  
        from tensorflow.keras.constraints import max_norm
        import tensorflow.keras.backend as K
           
        inp = tf.keras.layers.Input(shape = (len(train_cols),))
        x = tf.keras.layers.Dense(10, activation = 'relu')(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(10, activation = 'relu')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.4)(x)
        # x = tf.keras.layers.Dense(24, activation = 'relu')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(51, activation = 'relu')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.4)(x)
        out = tf.keras.layers.Dense(len(np.unique(y_train)), activation = 'softmax')(x)
        model = tf.keras.models.Model(inputs = inp, outputs = out)

        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                          mode = 'auto',
                                                          patience = 10,
                                                          restore_best_weights = True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                         mode = 'auto',
                                                         factor = 0.5,
                                                         patience = 5)

        adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
        neptune.log_text("init_lr", str(K.eval(adam.lr))) 
        model.compile(optimizer=adam, loss = metric, metrics=['accuracy'])

        
        history = model.fit(X_train[train_cols], pd.get_dummies(y_train),
                  validation_data = (X_val[train_cols] , pd.get_dummies(y_val)),
                  epochs = 100, 
                  batch_size = 4,
                  callbacks = [early_stopping, reduce_lr],
                  verbose = 1) 
        neptune.log_text("epochs", str(history.params["epochs"]))  
        neptune.log_text("steps", str(history.params["steps"]))  
        neptune.log_text("early_stopping_rounds", str(K.eval(early_stopping.patience))) 
        
        layers = [str(type(l)).split()[1].split(">")[0].split(".")[-1][:-1] for l in model.layers]
        neptune.log_text("Layers", "_".join(layers))

        
    gc.collect()
    return [model, history]
          


def pipeline(data, config):
    
    df = data.copy()
    #Reduce memory
    df = reduce_mem_usage(df)
    #Drop useless columns
    df.drop(config["useless_feat"], axis=1, inplace=True)
    #Copy of submission features
    df = submission_vars(df, config["submission_feat"])
    #Label encode objective
    df, target_decoder = label_encoder(df, [config['target']], True) 
    #Dealing with cathegorical variables
    df = label_encoder(df, config['label_encod'])                                               
    df = binary_encoder(df, config['binary_encod'])
    df = dummies_encoder(df, config['oneHot_encod'])
    # df = target_encoder(df, target_encod, target, 10)                                               
    #Process nas
    df = proccess_nas(df, config['target'], config['submission_feat'], config['na_treat'])
    #Scale
    df, sc, scale_feat = scaler(df, config['submission_feat'], [config['target']], config['scale'])
    #Map to a Gaussian distribution
    df, pt, gaussian_feat = to_gaussian(df, config['submission_feat'], [config['target']], config['gauss'])
    #Create Statistics
    df = create_statistics(df, config['stats'])
    #Select k features
    df = best_features(df, config['target'],config['submission_feat'],config['k_feat'])
    #Perform PCA
    df = perform_pca(df, config['target'], config['submission_feat'], config['pca'])
    #Perform LDA
    df = perform_lda(df, config['target'], config['submission_feat'], config['lda'])
    #Delete unused memory

    
    X = df[df["test"] == "train"].copy().reset_index().drop("index", axis=1)
    T = df[df["test"] == "test"].copy().reset_index().drop("index", axis=1)
    X.drop("test", axis = 1, inplace=True)
    T.drop("test", axis = 1, inplace=True)
    X_y = X[config['target']]
    T_y = X[config['target']]
    X_strat = X[config['strat']]
    train_cols = list(set(X.columns) - set([feat.upper() for feat in config['submission_feat']]) - set([config['target']]))
    
    #Deal with imbalanced classes: 
    X, X_y = balance_classes(X, X_y, X_strat, config['target'], train_cols, config['balance'])

    gc.collect()
    return X, X_y, T, T_y, train_cols, (target_decoder, sc, scale_feat, pt, gaussian_feat)    

#%%
neptune.init(project_qualified_name='laguila/sandbox', # change this to your `workspace_name/project_name`
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMWVjZjQ1NmEtMmY5Yi00YTJiLTk0NzMtNGI3Y2NiZWMwNjE5In0=', 
            )

#%% Read Data
# df= dt.fread("data/data.csv").to_pandas()
from sklearn.datasets import load_iris
iris = load_iris()
df=pd.DataFrame(iris.data)
df['species']=iris.target
df.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df.dropna(how="all", inplace=True) # remove any empty lines
df["Species"]=df["Species"].replace(0, iris.target_names[0])
df["Species"]=df["Species"].replace(1, iris.target_names[1])
df["Species"]=df["Species"].replace(2, iris.target_names[2])

test_inds = np.random.choice(df.index.values, round(0.3*df.shape[0]), replace = False)
train_inds = np.setdiff1d(df.index.values, test_inds)
df["test"] = "train"
df.loc[test_inds,"test"] = "test"
#%%
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(data = boston.data, columns = boston.feature_names)
df["MEDV"] = boston.target
test_inds = np.random.choice(df.index.values, round(0.3*df.shape[0]), replace = False)
train_inds = np.setdiff1d(df.index.values, test_inds)
df["test"] = "train"
df.loc[test_inds,"test"] = "test"

#%%
df.dtypes

#Set Features
config = {
    "useless_feat" : [],                                      #columns to be dropped from the beginning
    "submission_feat" : ["SepalLength", "Species"],    #columns to be copied and kept as they are, for submission, strat_k_fold, statistics, stratify, etc
    "target":'Species',                                        #target variable
    "strat" : "SPECIES",                               # strat: feature to perform stratified_k_fold or just stratified train_split. If not in columns, it wont be stratified
    "label_encod" : [],
    "binary_encod" : [],
    "oneHot_encod" : [],
    "target_encod" : [],
    "na_treat" : "no",                                 #omit, mean, most_frequent, knn, bfill, ffill or just the number
    "scale" : "yes",
    "gauss" : "no",
    "stats" : "no",
    "k_feat" : 0,
    "pca" : 0,
    "lda" : 0,
    "balance" : "smote"                               #under, over, smote(over)
}


X, X_y, T, T_y, train_cols, decoders = pipeline(df, config)
                   

model_type="keras"   #keras or lgb
k = 2                #k: if k<1, percentage of data for validation, if k=1, percentage = 0.2. If k>1 (integer), number of cross-validation splits.
bayes = 5           # bayes: set to 0 for no bayesian optimization. If >0, number of searches
metric = "categorical_crossentropy"   # lgb=multi_logloss     keras: categorical_crossentropy, binary_crossentropy

#%%Train model

models = kfold(X, X_y, train_cols, k, bayes, model_type, metric, config)

#%% Make predictions

submission = predict(T, models, train_cols, model_type, decoders, config) 

#%%Save file
submission.to_csv("submission.csv")
    



