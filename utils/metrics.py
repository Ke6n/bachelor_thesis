#!/usr/bin/env python3
import numpy as np
import errors

# 1. metrics of goodness of fit
#   1.1 Scale dependent errors
#       MAE
#       MdAE
#       MSE
#       RMSE
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    abs_error = errors.absolute_error(y_true, y_pred)
    return np.mean(abs_error)
mae = mean_absolute_error

def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    abs_error = errors.absolute_error(y_true, y_pred)
    return np.median(abs_error)
mdae = median_absolute_error

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    squared_error = errors.squared_error(y_true, y_pred)
    return np.mean(squared_error)
mse = mean_squared_error

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(mse(y_true, y_pred))
rmse = root_mean_squared_error



# 2. metrics of biasedness

# 3. metrics of correct sign