from itertools import count
import numpy as np
import pandas as pd
import math
from typing import Literal, Union

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
	
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.

    Inputs:
    > y_hat: predictions
    > y: ground truth

    """
    assert y_hat.size == y.size
    y_hat.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Calculate the number of correct predictions
    correct_predictions = sum(y_hat == y)

    # Calculate the accuracy
    accuracy_score = correct_predictions / len(y_hat)

    return accuracy_score

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision

    Inputs:
    > y_hat: predictions
    > y: ground truth

    """
    assert y_hat.size == y.size
    assert y_hat.size > 0
    y_hat.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    prediction_positives = y_hat == cls
    if sum(prediction_positives) > 0:
        return (y_hat[prediction_positives] == y[prediction_positives]).sum() / prediction_positives.sum()
    else:
        return None

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall

    Inputs:
    > y_hat: predictions
    > y: ground truth
    > cls: The class chosen
    
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0
    y_hat.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    ground_positives = y == cls
    if sum(ground_positives) > 0:
        return (y_hat[ground_positives] == y[ground_positives]).sum() / ground_positives.sum()
    else:
        return None

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    
    """
    y_hat.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    rmse = 0.0
    mse = 0.0
    for ind in range(len(y_hat)):
        v = (y_hat[ind] - y[ind] + 0.0) ** 2
        mse += v
    if mse == 0.0:
        return 0.0
    rmse = ((mse + 0.0) / len(y_hat)) ** 0.5

    return float(rmse)

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth

    """
    mae = 0.0
    for ind in range(len(y_hat)):
        mae += abs(y_hat[ind] - y[ind])
    return float((mae + 0.0) / len(y))
