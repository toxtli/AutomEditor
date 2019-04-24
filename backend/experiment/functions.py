import numpy as np
from keras import backend as K

def ccc_loss(y_true, y_pred):
    # using CCC as loss function
    true_mean = K.mean(y_true)
    pred_mean = K.mean(y_pred)
    
    std_predictions = K.std(y_pred)
    std_gt = K.std(y_true)
    
    covariance = K.mean((y_true - true_mean)*(y_pred - pred_mean))
    ccc = 2 * covariance / (
    std_predictions ** 2 + std_gt ** 2 +
    (pred_mean - true_mean) ** 2)

    return 1 - ccc

def ccc_metric(y_true, y_pred):
    # using CCC as metric
    true_mean = K.mean(y_true)
    pred_mean = K.mean(y_pred)
    
    std_predictions = K.std(y_pred)
    std_gt = K.std(y_true)
    
    covariance = K.mean((y_true - true_mean)*(y_pred - pred_mean))
    ccc = 2 * covariance / (
    std_predictions ** 2 + std_gt ** 2 +
    (pred_mean - true_mean) ** 2)

    return ccc

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true),axis = -1)



