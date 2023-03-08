import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    try:
        precision = (np.sum(y_pred) - np.sum(y_pred > y_true)) / np.sum(y_pred)
        print(f'Precision = {precision}')
    except ZeroDivisionError:
        precision = 0
        print('Precision = 0')
        
    try:
        recall = (np.sum(y_true) - np.sum(y_pred < y_true)) / np.sum(y_true)    
        print(f'Recall = {recall}')
    except ZeroDivisionError:
        recall = 0
        print('Recall = 0')
        
    try:
        f1 = 2 * (precision * recall) / (precision + recall)    
        print(f'F1 = {f1}')
    except ZeroDivisionError:
        f1 = 0
        print('F1 = 0')
        
    accuracy = np.sum(y_pred == y_true) / len(y_pred)
    print(f'Accuracy = {accuracy}')
    
    return (precision, recall, f1, accuracy)


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    accuracy = np.sum(y_pred == y_true) / len(y_pred)
    print(f'Accuracy = {accuracy}')
    
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_true - np.mean(y_true)))
    
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.mean(np.square(y_pred - y_true))
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.mean(np.abs(y_pred - y_true))
    
    return mae
    