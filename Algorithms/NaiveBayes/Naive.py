# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 10:29:11 2021

@author: laguila
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix

#Load data
iris = datasets.load_iris()

#Gaussian: It is used when the dataset is normally distributed.
#Multinomial: It is used when the dataset contains discrete values.
#Bernoulli: It is used while working on binary classification problems.

gnb = GaussianNB()
mnb = MultinomialNB()
y_pred_gnb = gnb.fit(iris.data, iris.target).predict(iris.data)
cnf_matrix_gnb = confusion_matrix(iris.target, y_pred_gnb)
print(cnf_matrix_gnb)

#Make predictions
y_pred_mnb = mnb.fit(iris.data, iris.target).predict(iris.data)
cnf_matrix_mnb = confusion_matrix(iris.target, y_pred_mnb)
print(cnf_matrix_mnb)