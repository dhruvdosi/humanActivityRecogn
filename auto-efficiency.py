import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import tree

""" 
# Compariing model with the dummy classifier
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train)
yy_hat = dummy_clf.predict(X_test)


""" 
np.random.seed(42)

# HEre we will be reading the data as in the question

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn



# CLEANUP of the DATA
    # removing unclear rows
data.drop(data.index[data['horsepower'] == "?"], inplace=True)
    # removing unclear columns
data.drop('car name', axis=1, inplace=True)
for i in data.columns:
    data[i] = data[i].astype('float64') # we are converting columns to float 64
data.reset_index(drop=True, inplace=True)

#SPLITING our data in to desired fraction values ( training set & test set)
fractvalue = 0.7
n_value = int(fractvalue*len(data))
X_trainSET, X_testSET, y_trainSET, y_testSET = data.iloc[:n_value,1:], data.iloc[n_value+1:,1:], data.iloc[:n_value,0], data.iloc[n_value+1:,0]

#WE will be now spliting on the information gain
tree0 = DecisionTree(criterion='information_gain', max_depth=5)  # Split based on Inf. Gain
tree0.fit(X_trainSET, y_trainSET)
y_hat = tree0.predict(X_testSET)
tree0.plot()
print('root mean squar error value: ', rmse(y_hat, y_testSET))
print('MAE value: ', mae(y_hat, y_testSET))
print('<    =   =   =   =   =   =   =   =   =   =   =   =   =   >')

# now Traininga a decision tree model from scikit-learn

tree1 = DecisionTreeRegressor(max_depth=5)
tree1.fit(X_trainSET, y_trainSET)
main_text = tree.export_text(tree1)
print(main_text)
y_newhat = tree1.predict(X_testSET)
y_newhat = pd.Series(y_newhat)
print('Root mean square error: ', rmse(y_newhat, y_testSET))
print('MAE values: ', mae(y_newhat, y_testSET))
