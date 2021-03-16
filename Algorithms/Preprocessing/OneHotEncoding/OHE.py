import pandas as pd
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                       ['red', 'L', 13.5, 'class2'],
                       ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

#LabelEncoder
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

class_le.inverse_transform(y)

#OneHotEncoder
pd.get_dummies(df[['classlabel', 'color', 'size']])