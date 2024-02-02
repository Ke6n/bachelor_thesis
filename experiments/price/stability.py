#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
sys.path.append("../..")
import utils.errors as errors

df_true = pd.read_json('experimental_data/true.json')
df_pred_arima110 = pd.read_json('experimental_data/pred_arima110.json')
df_pred_holt = pd.read_json('experimental_data/pred_holt.json')
df_pred_gbm = pd.read_json('experimental_data/pred_gbm.json')

arr_true = df_true.transpose().values[0]
arr_pred_arima110 = df_pred_arima110.transpose().values[0]
arr_pred_holt = df_pred_holt.transpose().values[0]
arr_pred_gbm = df_pred_gbm.transpose().values[0]

# errors
ae_arima110 = errors.absolute_error(arr_true, arr_pred_arima110)
ae_holt = errors.absolute_error(arr_true, arr_pred_holt)
ae_gbm = errors.absolute_error(arr_true, arr_pred_gbm)

# plotting
sys.path.append("..")
import plotting
save_path = '../../experiments_plots/violin_linien_plots/price_AE.png'
plotting.line_violin_plotting(save_path, ['arima(1,1,0)', 'holt-winters', 'lightGBM'], ae_arima110, ae_holt, ae_gbm)

# variance
var_ae_arima110 = np.var(ae_arima110)
var_ae_holt = np.var(ae_holt)
var_ae_gbm = np.var(ae_gbm)
