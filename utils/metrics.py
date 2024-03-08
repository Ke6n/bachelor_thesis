#!/usr/bin/env python3
import numpy as np
import utils.errors as errors

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

#   1.3 Weighted arithmetic mean based metrics
#       wMAPE
def weighted_MAPE(y_true: np.ndarray, y_pred: np.ndarray):
    sum_AE = np.sum(errors.absolute_error(y_true, y_pred))
    sum_true = np.sum(np.abs(y_true))
    return 100*(sum_AE/sum_true)

#   1.4 measures based on symmetric errors
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

#   1.5 measures based on relative errors
#       MRAE
#       MdRAE
#       GMRAE
#       UMBRAE
def mean_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray):
    rae = errors.relative_absolute_error(y_true, y_pred,bm_pred)
    return np.mean(rae)
mrae = mean_relative_absolute_error

def median_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray):
    rae = errors.relative_absolute_error(y_true, y_pred,bm_pred)
    return np.median(rae)
mdrae = median_relative_absolute_error

from scipy.stats import gmean
def geometric_mean_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray):
    rae = errors.relative_absolute_error(y_true, y_pred,bm_pred)
    return gmean(rae)
gmrae = geometric_mean_relative_absolute_error

def unscaled_mean_bounded_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray):
    brae = errors.bounded_RAE(y_true, y_pred,bm_pred)
    mbrae = np.mean(brae)
    return mbrae/(1-mbrae)
umbrae = unscaled_mean_bounded_relative_absolute_error

#   1.6 relative measures
#       RMAE
#       RRMSE
#       LMR
def relative_MAE(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray):
    return mae(y_true, y_pred)/mae(y_true, bm_pred)

def relative_RMSE(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray):
    return rmse(y_true, y_pred)/rmse(y_true, bm_pred)

import math
def log_mean_squared_error_ratio(y_true: np.ndarray, y_pred: np.ndarray, bm_pred: np.ndarray):
    return math.log(relative_RMSE(y_true, y_pred, bm_pred))
lmr = log_mean_squared_error_ratio

#   1.7 measures based on scaled errors
#       MASE
#       MdASE
#       RMSSE
def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_in_sample: np.ndarray):
    return np.mean(np.abs(errors.scaled_error(y_true, y_pred, y_in_sample)))
mase = mean_absolute_scaled_error

def median_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_in_sample: np.ndarray):
    return np.median(np.abs(errors.scaled_error(y_true, y_pred, y_in_sample)))
mdase = median_absolute_scaled_error

def root_mean_squared_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_in_sample: np.ndarray):
    return np.sqrt(np.mean(np.square(errors.scaled_error(y_true, y_pred, y_in_sample))))
rmsse = root_mean_squared_scaled_error

#   1.8 metrics for evaluating the forecast interval
#       MSIS
def mean_scaled_interval_score(y_true: np.ndarray, y_in_sample: np.ndarray,
                                lower: np.ndarray, upper: np.ndarray, alpha = 0.1, seasonality = 1):
    mean_abs_sea_error = np.mean(np.abs(errors.seasonal_error(y_in_sample, seasonality)))
    interval_score = np.mean(
            upper - lower
            + 2.0/ alpha* (lower - y_true)*(y_true < lower)
            + 2.0/ alpha*(y_true - upper)*(y_true > upper)
        )
    return interval_score/mean_abs_sea_error
msis = mean_scaled_interval_score

# 2. metrics of biasedness
#    PTSU (aka. PSTSU) with standard sign test
def proportion_of_tests_supporting_unbiasedness(y_true: np.ndarray, y_pred: np.ndarray):
    errors = np.array(y_true - y_pred)
    test_arr = np.delete(errors, np.where(errors == 0))
    sign_arr = np.sign(test_arr)
    sign_tests = np.abs(np.cumsum(sign_arr))
    # hypothesis of unbiasedness: The differences between the number of positive errors and negative errors <= 1
    z = np.array(sign_tests <= 1).astype(int)
    return np.mean(z)
ptsu = proportion_of_tests_supporting_unbiasedness

# 3. metrics of correct sign
#    PCDCP
def percentage_of_correct_direction_change_prediction(y_true: np.ndarray, y_pred: np.ndarray):
    change_true = np.delete(y_true, 0) - np.delete(y_true, -1)
    change_pred = np.delete(y_pred, 0) - np.delete(y_true, -1)
    z = np.array(change_true*change_pred>0).astype(int)
    return np.mean(z)
pcdcp = percentage_of_correct_direction_change_prediction