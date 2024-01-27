"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np 

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    
    # 1 for the continuous and 0 for the discreate 
    y_dtype = y.dtype
    
    if y_dtype == 'category':
        return 0 
    
    else : 
        return 1 
    
    """  elif ratio< 0.2:
        return 0  # data is discrete   
    uniq = y.unique() ; 
    ratio = len(uniq)/len(y)
    """

    
def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
   
    
    m = len(Y)
    uniq, counts = np.unique(Y, return_counts=True)
    probabilities = counts / m
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value
    

 
def gini_index(Y: pd.Series, attr: pd.Series, value_of_split=0) -> float:
    """
    Function to calculate the gini index
    """
    size_of_y = len(Y)
    all_uniq, counts = np.unique(Y, return_counts=True)
    gini_value = 1

    total_elements = len(Y)
    if value_of_split != 0:
        y1 = Y[attr <= value_of_split]
        y2 = Y[attr > value_of_split]
        gini_value += -((len(y1) / total_elements) ** 2 + (len(y2) / total_elements) ** 2)
    else:
        probabilities = counts / size_of_y
        gini_value -= np.sum(probabilities ** 2)

    return gini_value



def information_gain(Y: pd.Series, attr: pd.Series,  value_of_split ,  isRealInput: bool, isRealOutput: bool) -> float:
    """
    Function to calculate the information gain
    """
    # gain = entropy(S)- sum(|Sv|/|S|)
    if isRealOutput==0 :
        info_gain = entropy(Y) 
        total_elements = len(Y)
        # splitting on the real input 
        if(isRealInput == 1):
            y1 = Y[attr <= value_of_split]
            y2 = Y[attr > value_of_split]
            info_gain += -(len(y1)/total_elements)*entropy(y1)-(len(y2)/total_elements)*entropy(y2)

        else : 
            
            # finding all unique in attribute column 
            all_uniq = attr.unique()
            for item in all_uniq: 
                items = Y[attr == item] 
                siz = len(items)
                info_gain -= ((siz)/total_elements)*entropy(items)
        return info_gain
    
    # if real output then finding the reduction in mean squared error 
    else:
         # finding the mean of whole data 
        mu = Y.mean()
        total_size = len(Y)
        info_gain = 0 
        MSE_of_y = ((Y - mu) ** 2).mean()
        info_gain = MSE_of_y 
        # finding the MSE after splitting 
        all_uniq = attr.unique()
        for att in all_uniq:
            subsetY = Y[attr==att]
            siz = len(subsetY)
            MSE_of_subset = 0 
            MSE_of_subset = ((subsetY - subsetY.mean()) ** 2).mean() 
            info_gain -= (siz/total_size)*MSE_of_subset
        return info_gain 

   

def opt_split_attribute(X: pd.DataFrame, y: pd.Series,criterion, features: pd.Series, isRealInput: bool, isRealOutput: bool):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    
    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    num_columns = X.shape[1]

    if num_columns == 0:
        return -1, None

    

    if isRealInput == 0 and isRealOutput ==0:
        opt_attribute = -1
        max_info_gain = -1 
        for feature in features:
            if criterion == 'gini_index':
                info_gain =  gini_index(y, X[feature], )
            else : 
                info_gain = information_gain( y , X[feature] ,0, isRealInput, isRealOutput )
            if info_gain>max_info_gain:
                max_info_gain = info_gain 
                opt_attribute = feature 

        return opt_attribute,0


    # Discrete input and real output 
    elif isRealOutput==1 and isRealInput ==0:
        opt_attribute = -1
        max_info_gain = -1 
        for feature in features:
            info_gain = information_gain( y , X[feature], 0, isRealInput, isRealOutput )
            if info_gain>max_info_gain:
                max_info_gain = info_gain 
                opt_attribute = feature 

        return opt_attribute,0
    
    # if input is real and output is discrete 
    # sort by the input value, if at any instance attribute values is changing picking that point 
    elif isRealOutput ==0 and isRealInput: 
        opt_attribute = -1
        opt_info_gain = -1 
        opt_value = -1 
        for feature in features:
            # sorting 
            x_feature = X[feature]
            # Get the indices that would sort the Temperature Series
            sorted_indices = x_feature.argsort()

            # Use the sorted indices to rearrange both Series
            sorted_input = x_feature.iloc[sorted_indices]
            sorted_output = y.iloc[sorted_indices]

            siz = len(sorted_input)
            for i in range(siz-1):
                if sorted_output.iloc[i] != sorted_output.iloc[i+1]:
                    # at this point we have to check the info_gain 
                    value_of_split = (sorted_input.iloc[i]+sorted_input.iloc[i+1])/2 
                    # info gain 
                    if criterion == "gini_index": 
                        info_gain = gini_index(y, X[feature], value_of_split)
                    else : 
                        info_gain = information_gain(y, X[feature], value_of_split , isRealInput, isRealOutput )

                    if info_gain>opt_info_gain:
                        opt_info_gain = info_gain 
                        opt_value = value_of_split
                        opt_attribute = feature
           

        return opt_attribute,opt_value

    # for real input and real output: 
    else : 
        optimal_feature = None
        optimal_split_value = None
        min_loss = float('inf')
        for feature in features:
            feature_values = X[feature]
            data = pd.concat([feature_values, y], axis=1, keys=[feature, 'y'])

            # Sorting data by feature values
            sorted_data = data.sort_values(by=feature)

            for i in range(1, len(sorted_data)):
                split_point = (sorted_data.iloc[i-1, 0] + sorted_data.iloc[i, 0]) / 2

                # Split the data into two regions
                region1 = sorted_data[sorted_data[feature] <= split_point]['y']
                region2 = sorted_data[sorted_data[feature] > split_point]['y']

                mean1 = region1.mean()
                mean2 = region2.mean()

                # Calculate mean squared loss
                loss = ((region1 - mean1)**2).sum() + ((region2 - mean2)**2).sum()

                if loss < min_loss:
                    min_loss = loss
                    optimal_feature = feature
                    optimal_split_value = split_point
        return optimal_feature, optimal_split_value



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value, isRealInput: bool):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    
    # no need to keep attribute attribute column, delete it
    # return the splitted data 
    # the below implementation is for the discreate 
    num_columns = X.shape[1]
    splitted_data = {}
    if num_columns == 0:
        return splitted_data


    # for DIDO & DIRO 
    if isRealInput ==0: 
        uniq = X[attribute].unique()
        # make an array of all splited data 
        for attr in uniq:
            subset_of_attr = X[X[attribute] == attr].drop(attribute, axis=1)  # Dropping the attribute column
            subset_of_attr_y = y[X[attribute] == attr]
            splitted_data[attr] = (subset_of_attr, subset_of_attr_y)
    # RIDO && RIRO 
    else :
        splitted_data = {'above': (X[X[attribute] > value], y[X[attribute] > value]),
                         'below': (X[X[attribute] <= value], y[X[attribute] <= value])}

    return splitted_data

