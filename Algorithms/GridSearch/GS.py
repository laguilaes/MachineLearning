from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
import sys
import pandas as pd
import numpy as np

#Read data
columns = ['num_pregnant', 'glucose_concentration', 'blood_pressure', 'skin_thickness',
           'serum_insulin', 'BMI', 'pedigree_function', 'age', 'class']

df = pd.read_csv("D:/MachineLearning/Algorithms/GridSearch/pima-indians-diabetes.csv", names=columns)

#Preprocess data
for col in columns:
    df[col].replace(0, np.NaN, inplace=True)

df.dropna(inplace=True)
dataset = df.values 

#Standard Scaler
X = dataset[:,0:8]
Y = dataset[:, 8].astype(int)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)

X_standardized = scaler.transform(X)

data = pd.DataFrame(X_standardized)

#Create model
def create_model(learn_rate, dropout_rate):
    # Create model
    model = Sequential()
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    adam = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# Create the model
model = KerasClassifier(build_fn=create_model, verbose=1)

learn_rate = [0.001, 0.02, 0.2]
dropout_rate = [0.0, 0.2, 0.4]
batch_size = [10, 20, 30]
epochs = [1, 5, 10]
seed = 42

# Make a dictionary of the grid search parameters
param_grid = dict(learn_rate=learn_rate, dropout_rate=dropout_rate, batch_size=batch_size, epochs=epochs )

# Build and fit the GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=KFold(random_state=seed), verbose=10)

grid_results = grid.fit(X_standardized, Y)

# Summarize the results in a readable format
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))