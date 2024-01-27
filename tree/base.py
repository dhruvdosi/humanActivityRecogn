from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion= "information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.isRealInput = False 
        self.isRealOutput = False 

    def check_isReal(self, X: pd.Series) -> bool:
        """
        Function to check if the input features are real-valued
        """
        return check_ifreal(X)


    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0):
        """
        Function to train and construct the decision tree
        """
        type_of_x = X.dtypes.iloc[0]
        if(type_of_x == 'category'):
            self.isRealInput = False 
        else : 
            self.isRealInput = True 
        self.isRealOutput = self.check_isReal(y)
        if depth == self.max_depth or len(y.unique()) == 1:
            # Stop splitting if max depth is reached or all labels are the same
            # y.mode() ==> will give majority elements 
            return {'leaf': y.mode().iloc[0]}

        if self.isRealInput == False :
            tree = self.fit_discrete(X, y, depth) 
            return tree
        else:
            return self.fit_real(X, y, depth)


    def fit_discrete(self, X: pd.DataFrame, y: pd.Series, depth=0):
        """
        Function to train and construct the decision tree for discrete output
        """

        optimal_feature, split_value = opt_split_attribute(X, y, self.criterion, X.columns, self.isRealInput, self.isRealOutput)

        if optimal_feature is None:
            return {'leaf': y.mode().iloc[0]}

        if depth == self.max_depth or len(y.unique()) == 1:
            return {'leaf': y.mode().iloc[0]}
        
        node = {'feature': optimal_feature, 'splits': {} }

        # Split the data
        splitted_data = split_data(X, y, optimal_feature, split_value, self.isRealInput)

        for attr_value, (sub_X, sub_y) in splitted_data.items():
            # Recursively build the tree for each split
            node['splits'][attr_value] = self.fit_discrete(sub_X, sub_y, depth + 1)

        self.tree = node
        return node

    def fit_real(self, X: pd.DataFrame, y: pd.Series, depth=0):
        """
        Function to train and construct the decision tree for real output
        """

        optimal_feature, optimal_split_value = opt_split_attribute(X, y, self.criterion, X.columns, self.isRealInput, self.isRealOutput)

        if optimal_feature is None:
            return {'leaf': y.mean()}

        if depth == self.max_depth or len(y.unique()) == 1:
            return {'leaf': y.mode().iloc[0]}

        node = {'feature': optimal_feature, 'split_value': optimal_split_value, 'left': None, 'right': None}

        # Split the data
        splitted_data = split_data(X, y, optimal_feature, optimal_split_value, self.isRealInput)
        # Recursively build the tree for left and right branches
        node['left'] = self.fit_real(splitted_data['below'][0], splitted_data['below'][1], depth + 1)
        node['right'] = self.fit_real(splitted_data['above'][0], splitted_data['above'][1], depth + 1)

        self.tree = node
        return node
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to predict the output based on the trained decision tree
        """
        if self.isRealInput:
            return self.predict_real(X)
        else:
            return self.predict_discrete(X)

    def predict_discrete(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to predict the output for discrete output
        """
        predictions = []

        for _, row in X.iterrows():
            node = self.tree
            if(node==None):
                predictions.append(0)
                continue
            while 'leaf' not in node:
                if(node['feature'] == -1 or node['feature']==None):
                    break
                attr_value = row[node['feature']]
                if attr_value in node['splits']:
                    node = node['splits'][attr_value]
                else:
                    node = {'leaf': node['splits'].mode().iloc[0]}
            if 'leaf' in node:
                predictions.append(node['leaf'])
            else: 
                predictions.append(0)

        return pd.Series(predictions)

    def predict_real(self, X: pd.DataFrame) -> pd.Series:
        predictions = []
        node = self.tree
        if(node==None):
            predictions.append(0)
        else : 
            for _, row in X.iterrows():
                node = self.tree
                while 'leaf' not in node:
                    if(node['feature'] == -1 or node['feature']==None):
                        break
                    if row[node['feature']] <= node['split_value']:
                        node = node['left']
                    else:
                        node = node['right']
                if 'leaf' in node:
                    predictions.append(node['leaf'])
                else: 
                    predictions.append(0)

        return pd.Series(predictions)
    

    def plot(self) -> None:
        """
        Function to plot the tree
        """
        if self.isRealInput:
            self.plot_real()
        else:
            self.plot_discrete()
        
    def plot_discrete(self) -> None:
        self._plot_discrete_recursive(self.tree, '')

    def _plot_discrete_recursive(self, node, indent):
        
        if 'leaf' in node:
            print(f"Y: {node['leaf']}")
        else:
            print(f"?(Feature_{node['feature']})")
            for attr_value, sub_node in node['splits'].items():
                print(f"{indent}    {attr_value}: ", end="")
                self._plot_discrete_recursive(sub_node, f"{indent }     ")
               


    def plot_real(self) -> None:
        self._plot_real_recursive(self.tree, '')

    def _plot_real_recursive(self, node, indent):
        
        if 'leaf' in node:
            print(f"Value: {node['leaf']}")
        else:
            print(f"?(Feature_{node['feature']} > {node['split_value']})")
            print(f"{indent}   Y: ", end="")
            self._plot_real_recursive(node['left'], f"{indent }     ")
            print(f"{indent}   N: ", end="")
            self._plot_real_recursive(node['right'], f"{indent }     ")


""" 
data = {
    'Feature1': [2, 4, 7, 5, 8, 1],
    'Feature2': [3, 1, 6, 8, 4, 2],
    'y': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

data = {
    'Feature1': [3, 1, 6, 8, 4, 2],
    'Feature2': [2, 4, 7, 5, 8, 1],
    'y': [40, 48, 60, 72, 80, 90]
}
df_dido = pd.DataFrame(data)
print(df_dido.columns)
features = pd.Series(["Feature1" , "Feature2"])
dt_dido = DecisionTree(criterion='information_gain', max_depth=10)
dt_dido.fit(df_dido[features], df_dido['y'])
data1 = {
    'Feature1': [3, 1, 6, 8, 4, 2],
    'Feature2': [21, 4, 7, 50, 8, 11],
}
df_data1 = pd.DataFrame(data1)
print(dt_dido.predict(df_data1))
#print(dt_dido.tree) 
#dt_dido.plot()
"""
""""  
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'High', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Minutes Played': [20, 24, 40, 50, 60, 10, 4, 10, 60, 40, 45, 40, 35, 20]
}

df_dido = pd.DataFrame(data)
features = pd.Series(["Outlook", "Temp", "Humidity", "Wind"])
dt_dido = DecisionTree(criterion='information_gain', max_depth=2)
dt_dido.fit(df_dido[features], df_dido['Minutes Played'])
#print(dt_dido.tree) 
dt_dido.plot()
print(dt_dido.predict(df_dido[features]))

print(dt_dido.predict(df_dido[features]))
"""



