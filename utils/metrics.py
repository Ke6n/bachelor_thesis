#!/usr/bin/env python3
import numpy as np
import errors

# 1. metrics of goodness of fit
#   1.1 Scale dependent measures
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

#   1.2 measures based on percentage errors
#       MAPE
#       MdAPE
#       RMSPE
#       RMdSPE
def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    ape = errors.absolute_percentage_error(y_true, y_pred)
    return np.mean(ape)
mape = mean_absolute_percentage_error

def median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    ape = errors.absolute_percentage_error(y_true, y_pred)
    return np.median(ape)
mdape = median_absolute_percentage_error

def root_mean_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    spe = errors.squared_percentage_error(y_true, y_pred)
    return np.sqrt(np.mean(spe))
rmspe = root_mean_squared_percentage_error

def root_median_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    spe = errors.squared_percentage_error(y_true, y_pred)
    return np.sqrt(np.median(spe))
rmdspe = root_median_squared_percentage_error

#   1.3 measures based on symmetric errors
#       sMAPE
#       sMdAPE
#       msMAPE
def symmetric_MAPE(y_true: np.ndarray, y_pred: np.ndarray):
    sape = errors.symmetric_absolute_percentage_error(y_true, y_pred)
    return np.mean(100 * sape)

def symmetric_MdAPE(y_true: np.ndarray, y_pred: np.ndarray):
    sape = errors.symmetric_absolute_percentage_error(y_true, y_pred)
    return np.mean(100 * sape)

def modified_sMAPE(y_true: np.ndarray, y_pred: np.ndarray):
    msape = errors.modified_symmetric_absolute_percentage_error(y_true, y_pred)
    return np.mean(msape)

#   1.4 measures based on relative errors
#       MRAE
#       MdRAE
#       GMRAE
def mean_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    rae = errors.relative_absolute_error(y_true, y_pred,bm_pred)
    return np.mean(rae)
mrae = mean_relative_absolute_error

def median_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    rae = errors.relative_absolute_error(y_true, y_pred,bm_pred)
    return np.median(rae)
mdrae = median_relative_absolute_error

def geometric_mean_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    rae = errors.relative_absolute_error(y_true, y_pred,bm_pred)
    return np.exp(np.mean(np.log(rae)))
gmrae = geometric_mean_relative_absolute_error

#   1.5 relative measures
#       RMAE
#       RRMSE
#       LMR
def relative_MAE(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    return mae(y_true, y_pred)/mae(y_true, bm_pred)

def relative_RMSE(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    return rmse(y_true, y_pred)/rmse(y_true, bm_pred)

import math
def log_mean_squared_error_ratio(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray) -> np.ndarray:
    return math.log(relative_RMSE(y_true, y_pred, bm_pred))
lmr = log_mean_squared_error_ratio


#   1.6 measures based on scaled errors
#       MASE
#       RMSSE

# 2. metrics of biasedness

# 3. metrics of correct sign