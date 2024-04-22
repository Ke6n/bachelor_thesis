#!/usr/bin/env python3
import pandas as pd
import numpy as np

# loading the results of predictions
# --- price ---
price_true = pd.read_csv('../exp_data/price/multi_true.csv', index_col=['region','Date'])
price_pred_arima =  pd.read_csv('../exp_data/price/multi_pred_arima.csv', index_col=['region','Date'])
price_pred_holt = pd.read_csv('../exp_data/price/multi_pred_holt.csv', index_col=['region','Date'])
price_pred_gbm = pd.read_csv('../exp_data/price/multi_pred_gbm.csv', index_col=['region','Date'])
price_in_sample = pd.read_csv('../exp_data/price/multi_in_sample.csv', index_col=['region','Date'])
price_bm_pred =  pd.read_csv('../exp_data/price/multi_pred_arima010.csv', index_col=['region','Date'])

#--- sales ---
sales_true = pd.read_csv('../exp_data/sales/multi_true.csv', index_col=['Store','Date'])
sales_pred_arima =  pd.read_csv('../exp_data/sales/multi_pred_arima.csv', index_col=['Store','Date'])
sales_pred_holt = pd.read_csv('../exp_data/sales/multi_pred_holt.csv', index_col=['Store','Date'])
sales_pred_gbm = pd.read_csv('../exp_data/sales/multi_pred_gbm.csv', index_col=['Store','Date'])
sales_in_sample = pd.read_csv('../exp_data/sales/multi_in_sample.csv', index_col=['Store','Date'])
sales_bm_pred =  pd.read_csv('../exp_data/sales/multi_pred_arima010.csv', index_col=['Store','Date'])

#--- stock ---
stock_true = pd.read_csv('../exp_data/stock/multi_true.csv', index_col=['TickerSymbol','Date'])
stock_pred_arima =  pd.read_csv('../exp_data/stock/multi_pred_arima.csv', index_col=['TickerSymbol','Date'])
stock_pred_holt = pd.read_csv('../exp_data/stock/multi_pred_holt.csv', index_col=['TickerSymbol','Date'])
stock_pred_gbm = pd.read_csv('../exp_data/stock/multi_pred_gbm.csv', index_col=['TickerSymbol','Date'])
stock_in_sample = pd.read_csv('../exp_data/stock/multi_in_sample.csv', index_col=['TickerSymbol','Date'])
stock_bm_pred =  pd.read_csv('../exp_data/stock/multi_pred_arima010.csv', index_col=['TickerSymbol','Date'])


import sys
sys.path.append("../..")
import utils.accuracy as accu
import utils.df_labels as df_labels
metrics = df_labels.get_metrics()
win_size = 12

data_types = df_labels.get_datatypes()
methods = df_labels.get_main_models()
true_vars = [price_true, sales_true, stock_true]
in_sample_vars = [price_in_sample, sales_in_sample, stock_in_sample]
bm_vars = [price_bm_pred, sales_bm_pred, stock_bm_pred]
price_methods = [price_pred_arima, price_pred_holt, price_pred_gbm]
sales_methods = [sales_pred_arima, sales_pred_holt, sales_pred_gbm]
stock_methods = [stock_pred_arima, stock_pred_holt, stock_pred_gbm]
methods_price_dict = {k: v for k, v in zip(methods, price_methods)}
methods_sales_dict = {k: v for k, v in zip(methods, sales_methods)}
methods_stock_dict = {k: v for k, v in zip(methods, stock_methods)}
methods_dicts = [methods_price_dict, methods_sales_dict, methods_stock_dict]

types_true_dict = {k: v for k, v in zip(data_types, true_vars)}
types_methods_dict = {k: v for k, v in zip(data_types, methods_dicts)}
types_in_sample_dict = {k: v for k, v in zip(data_types, in_sample_vars)}
types_bm_dict = {k: v for k, v in zip(data_types, bm_vars)}

# Compute sets of metrics for each dataset and each model.
sys.path.append("..")
import plotting
for typ in data_types:
    for mod in methods:
        variance = []
        for metric in metrics:
            acc_arr = accu.get_multi_series_acc_set(metric, types_true_dict[typ], types_methods_dict[typ][mod], types_in_sample_dict[typ],
                                      types_bm_dict[typ], win_size)
            # plotting
            save_path = f'../../experiments_plots/violin_plots/{typ}_{mod}_{metric}.png'
            plotting.violin_plotting(save_path, metric, acc_arr)

            #variance
            var_acc = np.var(acc_arr)
            variance.append(var_acc)
        df = pd.DataFrame([variance], columns=metrics)
        df = df.sort_values(by=df.index[0], axis=1)
        df.to_latex(f'../exp_data/{typ}_{mod}_variance.tex', float_format="{:0.2e}".format, index=False)
