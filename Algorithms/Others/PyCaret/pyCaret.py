import numpy as np
import pandas as pd
import os

#Reading data
os.chdir("D:/MachineLearning/Algorithms/PyCaret")
data = pd.read_csv("train.csv")
data.head()

#Make classification with different models
from pycaret.classification import *
clf = setup(data, target = "Survived",
            ignore_features=["Ticket", "Name", "PassengerId"], 
            silent = True, session_id = 786)

#Compare models
compare_models()

#Train most accurate model
lightgbm = create_model('lightgbm')
test_data = pd.read_csv('test.csv')
predict = predict_model(lightgbm, data=test_data)
predict.head()