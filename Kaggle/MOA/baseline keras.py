import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import os
os.chdir("C:/Users\laguila\Desktop\MOA")
pd.set_option('display.max_columns', 50)

train_features = pd.read_csv('train_features.csv')
train_targets = pd.read_csv('train_targets_scored.csv')

#GetDummies manual
COLS = ['cp_type','cp_dose']
FE = []
for col in COLS:
    for mod in train_features[col].unique():
        FE.append(mod)
        train_features[mod] = (train_features[col] == mod).astype(int)
        
        
del train_features['sig_id']
del train_features['cp_type']
del train_features['cp_dose']
FE+=list(train_features.columns) 
del train_targets['sig_id']

# train_targets
# del train_targets['sig_id']
print(train_targets.columns)
print(np.array(train_features.to_numpy(), dtype=np.float).shape) #(23814, 877)
print(np.array(train_targets.to_numpy(), dtype=np.float).shape) #(23814, 206)

train_dataset = tf.data.Dataset.from_tensor_slices((np.array(train_features.to_numpy(), dtype=np.float), np.array(train_targets.to_numpy(), dtype=np.float)))
train_dataset = train_dataset.shuffle(100).batch(64)


model = tf.keras.Sequential([
    tf.keras.layers.Input(len(list(train_features.columns))),
    #tf.keras.layers.Flatten(input_shape=(len(list(train_features.columns)),1)),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(206, activation="softmax")
    ])
model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-6, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy","AUC"])
model.fit(train_dataset, epochs=2)


test_features = pd.read_csv('test_features.csv')

test_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_features.to_numpy(), dtype=np.float))
test_dataset = test_dataset.batch(64)

for col in COLS:
    for mod in test_features[col].unique():
        test_features[mod] = (test_features[col] == mod).astype(int)

sig_id = pd.DataFrame()
sig_id = test_features.pop('sig_id')
del test_features['cp_type']
del test_features['cp_dose']

columns = pd.read_csv('train_targets_scored.csv')
del columns['sig_id']


hist = model.predict(test_features)
hist.shape
sub = pd.DataFrame(data=hist, columns=columns.columns)


sample = pd.read_csv('sample_submission.csv')
sub.insert(0, column = 'sig_id', value=sample['sig_id'])


sub.to_csv('submission.csv', index=False)


def Diff(list1, list2): 
    return (list(list(set(list1)-set(list2)) + list(set(list2)-set(list1)))) 

Diff(sub.columns, pd.read_csv('sample_submission.csv').columns)