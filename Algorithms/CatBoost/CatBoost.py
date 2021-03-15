import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
import catboost as cb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#Import data
data = pd.read_csv('D:/MachineLearning/Algorithms/CatBoost/titanic.csv')
data.head()

#Data preparation
data.dropna(subset=['Survived'],inplace=True)
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']
X['Pclass'] = X['Pclass'].astype('str')
X['Fare'].fillna(0,inplace=True)
X['Age'].fillna(0,inplace=True)

#Convert all data to categorical
def get_categorical_indicies(X):
    cats = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            pass
        else:
            cats.append(col)
    cat_indicies = []
    for col in cats:
        cat_indicies.append(X.columns.get_loc(col))
    return cat_indicies
categorical_indicies = get_categorical_indicies(X)

def convert_cats(X):
    cats = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            pass
        else:
            cats.append(col)
    cat_indicies = []
    for col in cats:
        X[col] = X[col].astype('category')
convert_cats(X)

X_train,X_test,y_train,y_test = train_test_split(X, 
                                                 y, 
                                                 test_size=0.2, 
                                                 random_state=101, 
                                                 stratify=y)

#Balancing data
train_df = pd.concat([X,y],axis=1)
survived = train_df[train_df['Survived']==1]
deceased = train_df[train_df['Survived']==0]
deceased = deceased.sample(n=len(survived), random_state=101)
train_df = pd.concat([survived,deceased],axis=0)
X_train = train_df.drop('Survived',axis=1)
y_train = train_df['Survived']


#Training model
train_dataset = cb.Pool(X_train,y_train, 
                        cat_features=categorical_indicies)                                                      
test_dataset = cb.Pool(X_test,y_test,           
                       cat_features=categorical_indicies)
                       
  
model = cb.CatBoostClassifier(loss_function='Logloss',  
                              eval_metric='Accuracy')
grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5,],
        'iterations': [50, 100, 150]}
        
 
model.grid_search(grid,train_dataset)
pred = model.predict(X_test)
print(classification_report(y_test, pred))

