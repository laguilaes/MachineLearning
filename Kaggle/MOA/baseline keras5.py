from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm
import random
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from bayes_opt import BayesianOptimization

os.chdir('C:/Users/laguila/Desktop\MOA')
train = pd.read_csv('train_features.csv')
train_targets = pd.read_csv('train_targets_scored.csv')
test = pd.read_csv('test_features.csv')
sample_submission = pd.read_csv('sample_submission.csv')


#%%
FOLDS = 10
# Number of epochs to train each model
EPOCHS = 80
# Batch size
BATCH_SIZE = 124
# Learning rate
LR = 0.001
# Verbosity
VERBOSE = 0
# Seed for deterministic results
SEEDS1 = [123, 321]
SEEDS2 = [67, 33]
SEEDS3 = [141, 16]
SEEDS4 = [54, 81]

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
def mapping_and_filter(train, train_targets, test):
    cp_type = {'trt_cp': 0, 'ctl_vehicle': 1}
    cp_dose = {'D1': 0, 'D2': 1}
    for df in [train, test]:
        df['cp_type'] = df['cp_type'].map(cp_type)
        df['cp_dose'] = df['cp_dose'].map(cp_dose)
    train_targets = train_targets[train['cp_type'] == 0].reset_index(drop = True)
    train = train[train['cp_type'] == 0].reset_index(drop = True)
    train_targets.drop(['sig_id'], inplace = True, axis = 1)
    return train, train_targets, test

# Function to scale our data
def scaling(train, test):
    features = train.columns[2:]
    scaler = RobustScaler()
    scaler.fit(pd.concat([train[features], test[features]], axis = 0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features

# Function to extract pca features
def fe_pca(train, test, n_components_g = 70, n_components_c = 10, SEED = 123):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    def create_pca(train, test, features, kind = 'g', n_components = n_components_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        pca = PCA(n_components = n_components,  random_state = SEED)
        data = pca.fit_transform(data)
        columns = [f'pca_{kind}{i + 1}' for i in range(n_components)]
        data = pd.DataFrame(data, columns = columns)
        train_ = data.iloc[:train.shape[0]]
        test_ = data.iloc[train.shape[0]:].reset_index(drop = True)
        train = pd.concat([train, train_], axis = 1)
        test = pd.concat([test, test_], axis = 1)
        return train, test
    
    train, test = create_pca(train, test, features_g, kind = 'g', n_components = n_components_g)
    train, test = create_pca(train, test, features_c, kind = 'c', n_components = n_components_c)
    return train, test

# Function to extract common stats features
def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in [train, test]:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test

def c_squared(train, test):
    
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f'{feature}_squared'] = df[feature] ** 2
    return train, test

# Function to calculate the mean log loss of the targets including clipping
def mean_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0015, 1 - 0.0015)
    metrics = []
    for target in range(206):
        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))
    return np.mean(metrics)

# Function to create our 5 layer dnn model
def create_model_5l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2560, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2048, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1524, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(780, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(206, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0020),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model

# Function to create our 4 layer dnn model
def create_model_4l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2048, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1524, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(206, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0020),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model

# Function to create our 3 layer dnn model
def create_model_3l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4914099166744246)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1159, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.18817607797795838)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(960, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.12542057776853896)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1811, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20175242230280122)(x)
    out = tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation = 'sigmoid'))(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.Lookahead(opt, sync_period = 10)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0015),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model

# Function to create our 2 layer dnn model
def create_model_2l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.2688628097505064)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1292, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4598218403250696)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(983, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4703144018483698)(x)
    out = tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation = 'sigmoid'))(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.Lookahead(opt, sync_period = 10)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0015),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model


# Function to train our dnn
def train_and_evaluate(train, train_targets, test, features, SEED = 123, MODEL = '3l'):
    seed_everything(SEED)
    oof_pred = np.zeros((train.shape[0], 206))
    test_pred = np.zeros((test.shape[0], 206))   
    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = FOLDS, 
                                                                        random_state = SEED, 
                                                                        shuffle = True)\
                                              .split(train_targets, train_targets)):
        K.clear_session()
        if MODEL == '5l':
            model = create_model_5l(len(features))
        elif MODEL == '4l':
            model = create_model_4l(len(features))
        elif MODEL == '3l':
            model = create_model_3l(len(features))
        elif MODEL == '2l':
            model = create_model_2l(len(features))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_binary_crossentropy',
                                                          mode = 'min',
                                                          patience = 10,
                                                          restore_best_weights = True,
                                                          verbose = VERBOSE)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_binary_crossentropy',
                                                         mode = 'min',
                                                         factor = 0.3,
                                                         patience = 3,
                                                         verbose = VERBOSE)

        x_train, x_val = train[features].values[trn_ind], train[features].values[val_ind]
        y_train, y_val = train_targets.values[trn_ind], train_targets.values[val_ind]

        model.fit(x_train, y_train,
                  validation_data = (x_val, y_val),
                  epochs = EPOCHS, 
                  batch_size = BATCH_SIZE,
                  callbacks = [early_stopping, reduce_lr],
                  verbose = VERBOSE)

        oof_pred[val_ind] = model.predict(x_val)
        test_pred += model.predict(test[features].values) / FOLDS


    oof_score = mean_log_loss(train_targets.values, oof_pred)
    print(f'Our out of folds mean log loss score is {oof_score}')
    
    return test_pred, oof_pred
    

# Function to train our model with multiple seeds and average the predictions
def run_multiple_seeds(train, test, train_targets, SEEDS = [123], MODEL = '3l'):
    
    test_pred = []
    oof_pred = []
    
    for SEED in SEEDS:
        print(f'Training model {MODEL} with seed {SEED}')
        train_, test_ = fe_pca(train, test, n_components_g = 70, n_components_c = 10, SEED = SEED)
        train_, test_, features = scaling(train_, test_)
        print(f'Training with {len(features)} features')
        test_pred_, oof_pred_ = train_and_evaluate(train_, train_targets, test_, features, SEED = SEED, MODEL = MODEL)
        test_pred.append(test_pred_)
        oof_pred.append(oof_pred_)
        print('-'*50)
        print('\n')
        
    test_pred = np.average(test_pred, axis = 0)
    oof_pred = np.average(oof_pred, axis = 0)
        
    seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
    print(f'Our out of folds log loss for our seed blend model is {seed_log_loss}')
    
    return test_pred, oof_pred

def submission(test_pred):
    sample_submission.loc[:, train_targets.columns] = test_pred
    sample_submission.loc[test['cp_type'] == 1, train_targets.columns] = 0
    sample_submission.to_csv('submission.csv', index = False)
    return sample_submission

#%%


train, train_targets, test = mapping_and_filter(train, train_targets, test)
train, test = fe_stats(train, test)
train, test = c_squared(train, test)
print('-'*50)
print('\n')



test_pred_5l, oof_pred_5l = run_multiple_seeds(train, test, train_targets, SEEDS = SEEDS1, MODEL = '5l')
test_pred_4l, oof_pred_4l = run_multiple_seeds(train, test, train_targets, SEEDS = SEEDS2, MODEL = '4l')
test_pred_3l, oof_pred_3l = run_multiple_seeds(train, test, train_targets, SEEDS = SEEDS3, MODEL = '3l')
test_pred_2l, oof_pred_2l = run_multiple_seeds(train, test, train_targets, SEEDS = SEEDS4, MODEL = '2l')

oof_pred = np.average([oof_pred_5l, oof_pred_4l, oof_pred_3l, oof_pred_2l], axis = 0)
seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
print(f'Our final out of folds log loss for our blended models is {seed_log_loss}')

test_pred = np.average([test_pred_5l, test_pred_4l, test_pred_3l, test_pred_2l], axis = 0)
sample_submission = submission(np.clip(test_pred, 0.0015, 1 - 0.0015))
sample_submission.head()