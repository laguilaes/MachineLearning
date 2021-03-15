#Load data
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#TRain-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

#Visualize data
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8)

#Train KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#Make preds
prediction = knn.predict(X_test)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])