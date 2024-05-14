#!/usr/bin/env python3
import pandas as pd

__datasets = ["Price", "Sales", "Stock"]
# __metrics = ["MAE", "MdAE", "MSE", "RMSE", "MAPE", "MdAPE", "RMSPE", "RMdSPE", "wMAPE", "sMAPE", "sMdAPE", "msMAPE",
#             "MRAE", "MdRAE", "GMRAE", "UMBRAE", "RMAE", "RRMSE", "LMR", "MASE", "MdASE", "RMSSE", "MSIS","PTSU", "PCDCP"]
__metrics = ["MAE", "MdAE", "MSE", "RMSE", "MAPE", "MdAPE", "RMSPE", "RMdSPE", "wMAPE", "sMAPE", "sMdAPE", "msMAPE",
            "MRAE", "MdRAE", "GMRAE", "UMBRAE", "RMAE", "RRMSE", "LMR", "MASE", "MdASE", "RMSSE", "PTSU", "PCDCP"]
__models = ["ARIMA(010)*", "ARIMA(111)", "LightGBM", "Holt Winters", "Naive 1"]
__main_models = ["ARIMA", "Holt Winters", "LightGBM"]

__col_names = pd.MultiIndex.from_product([__datasets, __models])

def get_models():
    return __col_names

def get_metrics():
    return __metrics

def get_datatypes():
    return __datasets

def get_main_models():
    return __main_models