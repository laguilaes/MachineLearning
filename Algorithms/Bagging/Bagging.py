#%% BAGGING
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
bagging = BaggingClassifier(GaussianNB(),
                            max_samples=0.5, max_features=0.5)

# import some data to play with
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

gnb = BaggingClassifier(GaussianNB(), max_samples=0.5, max_features=0.5).fit(X_train, y_train)

y_pred = gnb.predict(X_test)


# Plot also the training points
#plt.scatter(X_train[:, 0], X_train[:, 1], c=np.array(["r", "g", "b"])[Z_pred], edgecolors=(0, 0, 0))
plt.scatter(X_train[:, 0], X_train[:, 1], c=np.array(["r", "g", "b"])[y_train], edgecolors=(0, 0, 0))
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xticks(())
plt.yticks(())

plt.tight_layout()
plt.show()