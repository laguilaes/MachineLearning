import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

#Load dataset
diabetes = datasets.load_diabetes()

#Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

#Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

#Make preds and visualize results
y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()