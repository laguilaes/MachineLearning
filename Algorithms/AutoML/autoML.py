import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('D:/MachineLearning/Algorithms/AutoML/Advertising.csv')
df.head()

#Init H2O
import h2o
h2o.init()

#Convert to h2o dataframe
adver_df = h2o.H2OFrame(df)
adver_df.describe()

#Train-test split
train, test = adver_df.split_frame(ratios=[.50])
x = train.columns
y = "sales"
x.remove(y)

#Develop model
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_runtime_secs=60,
                seed=1,
                balance_classes=False,
                project_name='Advertising'
)

#List with best models
lb = aml.leaderboard
lb.head()

#Analyze best model
se = aml.leader 
metalearner = h2o.get_model(se.metalearner()['name'])
metalearner.varimp()

model = h2o.get_model('DeepLearning_grid__1_AutoML_20200731_222821_model_1')
model.model_performance(test)

model.varimp_plot(num_of_features=3)

model.partial_plot(train, cols=["TV"], figsize=(5,5))