#!/usr/bin/env python3
import pandas as pd
import numpy as np
import utils.metrics as metrics

# Calculate all metrics and return a ndarray of accuracy
def get_accuracy_arr(y_true: np.ndarray, y_pred: np.ndarray, **kwargs:np.ndarray):
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