import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from metrics import *
from tree.base import DecisionTree

np.random.seed(42)

# Read dataset

        # generating synthetic dataset
P, Q = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Below plotting the dataset
plt.scatter(P[:, 0], P[:, 1], c=Q)

    #conveting teh dataset in to the pandas dataphrame
dataset = pd.DataFrame(P)
dataset['Y'] = pd.Series(Q, dtype='category')
P = dataset.iloc[:, :-1]
Q = dataset['Y']

    #SPLIT and then training dataset for " test and train" 
frac = 0.7
split_val = int(frac*len(dataset))

train_data, test_data = dataset.iloc[:split_val, :], dataset.iloc[split_val+1:, :]
X_trainset = train_data.iloc[:,:-1]
y_trainset = train_data.iloc[:,-1]
X_testdataset = test_data.iloc[:,:-1]
y_testdataset = test_data.iloc[:,-1]


# Decision tree based on the information gain criterion
FirstTREE = DecisionTree(criterion="information_gain", max_depth = 5)  # Split based on Inf. Gain
FirstTREE.fit(X_trainset,y_trainset)
y_CAP = FirstTREE.predict(X_testdataset)

FirstTREE.plot()
print('Accuracy have the value: ', accuracy(y_CAP,  y_testdataset))
for cls in test_data.iloc[:, -1].unique():
    print('Precision have the value: ', precision(y_CAP, test_data.iloc[:, -1], cls))
    print('Recall have the value: ', recall(y_CAP, test_data.iloc[:, -1], cls))


# NOw Optimizing for depth of decision tree using the 5 fold method

def find_best_depth(X, y, folds=5, depths=[1]):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    max_depth = max(depths)
    trees = {}
    accuracies = {}
    # similar to 100/4 = 25, size of data in every fold
    subset_size = int(len(X)//folds)

    for fold in range(folds):
        # we are Training a seperate model for each fold
        # we will getting the first set of data
        sub_data_inddexes = range(fold*subset_size, (fold+1)*subset_size)
        c_fold = []
        for i in range(len(X)):
            if(i in sub_data_inddexes):
                c_fold.append(True)
            else:
                c_fold.append(False)
        c_fold = pd.Series(c_fold)
            
        X_trainset = X[~c_fold].reset_index(drop=True)
        y_trainset = y[~c_fold].reset_index(drop=True)
        X_testing = X[c_fold].reset_index(drop=True)
        y_testing = y[c_fold].reset_index(drop=True)

        tree = DecisionTree(criterion='information_gain', max_depth=max_depth)
        tree.fit(X_trainset, y_trainset)
        trees[fold+1] = tree

        for depth in depths:
            print("Current depth value: "+str(depth))
            tree = DecisionTree(criterion='information_gain', max_depth=depth)
            tree.fit(X_trainset, y_trainset)
            y_hat = tree.predict(X_testing)
            if fold+1 in accuracies:
                accuracies[fold+1][depth] = accuracy(y_hat, y_testing)
            else:
                accuracies[fold+1] = {depth: accuracy(y_hat, y_testing)}

    accuracies = pd.DataFrame(accuracies).transpose()
    accuracies.index.name = "Fold ID"
    accuracies.loc["mean"] = accuracies.mean(axis = 'rows')
    best_mean_acc = accuracies.loc["mean"].max()
    best_depth = accuracies.loc["mean"].idxmax()
    print(accuracies)
    print("Best Mean Accuracy have the value:" + str(best_mean_acc))
    print("Optimum Depth have the value:"+str(best_depth))

find_best_depth(P, Q, folds=5, depths=list(range(1, 11)))



