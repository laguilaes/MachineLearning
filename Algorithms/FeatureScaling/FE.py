import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("D:/MachineLearning/Algorithms/FeatureScaling/housing.csv")
# using these three features for simplicity
data = data.loc[:, ["median_income", "latitude", "longitude"]]

training, testing = train_test_split(data, test_size=0.2, random_state=42)
normalization = MinMaxScaler().fit_transform(training)
standardization = StandardScaler().fit_transform(training)
