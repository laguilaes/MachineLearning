import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Reading data
data = pd.read_csv('D:/MachineLearning/Algorithms/PassiveAggressiveClassifier/news.csv')
print(data.head())

#Initial data analysis
labels = data.label
print(labels.head())

target = data.label.value_counts()
print(target)

sns.countplot(data.label)
plt.title("Distribution of Real and Fake News")
plt.show()

#Train-test split

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data['text'], 
                                                labels, test_size=0.2, 
                                                random_state=7)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
train = tfidf.fit_transform(xtrain)
test = tfidf.transform(xtest)

#Traning the fake news detection model
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(train, ytrain)

ypred = pac.predict(test)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(ytest, ypred)
print(f'Accuracy Score of Passive Aggresive Scassifier: {round(accuracy*100,2)}%')
