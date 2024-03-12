#!/usr/bin/env python3
import pandas as pd
import numpy as np
import utils.metrics as metrics

# Calculate all metrics and return a ndarray of accuracy
def get_accuracy_arr(**kwargs:np.ndarray):
    __input_check(**kwargs)
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    bm_pred = kwargs.get("bm_pred")
    y_in_sample = kwargs.get("y_in_sample")
    lower = kwargs.get("lower")
    upper = kwargs.get("upper")
    
    list = []
    mae = metrics.mae(y_true, y_pred)
    list.append(mae)
    mdae = metrics.mdae(y_true, y_pred)
    list.append(mdae)
    mse = metrics.mse(y_true, y_pred)
    list.append(mse)
    rmse = metrics.rmse(y_true, y_pred)
    list.append(rmse)
    mape = metrics.mape(y_true, y_pred)
    list.append(mape)
    mdape = metrics.mdape(y_true, y_pred)
    list.append(mdape)
    rmspe = metrics.rmspe(y_true, y_pred)
    list.append(rmspe)
    rmdspe = metrics.rmdspe(y_true, y_pred)
    list.append(rmdspe)
    wMAPE = metrics.weighted_MAPE(y_true, y_pred)
    list.append(wMAPE)
    sMAPE = metrics.symmetric_MAPE(y_true, y_pred)
    list.append(sMAPE)
    sMdAPE = metrics.symmetric_MdAPE(y_true, y_pred)
    list.append(sMdAPE)
    msMAPE = metrics.modified_sMAPE(y_true, y_pred)
    list.append(msMAPE)
    mrae = metrics.mrae(y_true, y_pred, bm_pred)
    list.append(mrae)
    mdrae = metrics.mdrae(y_true, y_pred, bm_pred)
    list.append(mdrae)
    gmrae = metrics.gmrae(y_true, y_pred, bm_pred)
    list.append(gmrae)
    umbrae = metrics.umbrae(y_true, y_pred, bm_pred)
    list.append(umbrae)
    relMAE = metrics.relative_MAE(y_true, y_pred, bm_pred)
    list.append(relMAE)
    relRMAE = metrics.relative_RMSE(y_true, y_pred, bm_pred)
    list.append(relRMAE)
    lmr = metrics.lmr(y_true, y_pred, bm_pred)
    list.append(lmr)
    mase = metrics.mase(y_true, y_pred, y_in_sample)
    list.append(mase)
    mdase = metrics.mdase(y_true, y_pred, y_in_sample)
    list.append(mdase)
    rmsse = metrics.rmsse(y_true, y_pred, y_in_sample)
    list.append(rmsse)
    msis = metrics.msis(y_true, y_in_sample, lower, upper)
    list.append(msis)
    ptsu = metrics.ptsu(y_true, y_pred)
    list.append(ptsu)
    pcdcp = metrics.pcdcp(y_true, y_pred)
    list.append(pcdcp)
    return np.array(list)

import utils.rolling_window as rwin      
# Calculate the set of a metric with the help of rolling windows
def get_accuracy_set(metric: str, window_size: int, **kwargs:np.ndarray):
    __input_check(**kwargs)
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    bm_pred = kwargs.get("bm_pred")
    y_in_sample = kwargs.get("y_in_sample")
    lower = kwargs.get("lower")
    upper = kwargs.get("upper")
    
    subset_tups = rwin.rolling_window(window_size, y_true, y_pred, bm_pred, lower, upper)
    list = []
    for tup in subset_tups:
        acc = __match_metric(metric, y_true=tup[0], y_pred=tup[1], bm_pred=tup[2],
                    y_in_sample=y_in_sample, lower=tup[3], upper=tup[4])
        list.append(acc)
    return np.array(list)

def __match_metric(metric, **kwargs:np.ndarray): 
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    bm_pred = kwargs.get("bm_pred")
    y_in_sample = kwargs.get("y_in_sample")
    lower = kwargs.get("lower")
    upper = kwargs.get("upper")
    match metric:
        case "MAE":
            return metrics.mae(y_true, y_pred)
        case "MdAE": 
            return metrics.mdae(y_true, y_pred)
        case "MSE": 
            return metrics.mse(y_true, y_pred)
        case "RMSE":
            return metrics.rmse(y_true, y_pred)
        case "MAPE":
            return metrics.mape(y_true, y_pred)
        case "MdAPE":
            return metrics.mdape(y_true, y_pred)
        case "RMSPE":
            return metrics.rmspe(y_true, y_pred)
        case "RMdSPE":
            return metrics.rmdspe(y_true, y_pred)
        case "wMAPE":
            return metrics.weighted_MAPE(y_true, y_pred)
        case "sMAPE":
            return metrics.symmetric_MAPE(y_true, y_pred)
        case "sMdAPE":
            return metrics.symmetric_MdAPE(y_true, y_pred)
        case "msMAPE":
            return metrics.modified_sMAPE(y_true, y_pred)
        case "MRAE":
            return metrics.mrae(y_true, y_pred, bm_pred)
        case "MdRAE":
            return metrics.mdrae(y_true, y_pred, bm_pred)
        case "GMRAE":
            return metrics.gmrae(y_true, y_pred, bm_pred)
        case "UMBRAE":
            return metrics.umbrae(y_true, y_pred, bm_pred)
        case "RMAE":
            return metrics.relative_MAE(y_true, y_pred, bm_pred)
        case "RRMSE":
            return metrics.relative_RMSE(y_true, y_pred, bm_pred)
        case "LMR":
            return metrics.lmr(y_true, y_pred, bm_pred)
        case "MASE":
            return metrics.mase(y_true, y_pred, y_in_sample)
        case "MdASE":
            return metrics.mdase(y_true, y_pred, y_in_sample)
        case "RMSSE":
            return metrics.rmsse(y_true, y_pred, y_in_sample)
        case "MSIS":
            return metrics.msis(y_true, y_in_sample, lower, upper)
        case "PTSU":
            return metrics.ptsu(y_true, y_pred)
        case "PCDCP":
            return metrics.pcdcp(y_true, y_pred)
        case _:
            raise ValueError("Metric name not recognized")
        
def __input_check(**kwargs:np.ndarray):
    keys = ["y_true", "y_pred", "bm_pred", "y_in_sample", "lower", "upper"]
    for key in keys:
        if key not in kwargs:
            raise KeyError(f"Keyword argument {key} is required")