#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
sys.path.append("../..")
import utils.accuracy as accu

price_in_sample = np.loadtxt('../exp_data/price/in_sample.txt')
price_bm_pred =  np.loadtxt('../exp_data/price/pred_arima010.txt')
price_pred_arima010 =  np.loadtxt('../exp_data/price/pred_arima010.txt')
price_pred_arima111 =  np.loadtxt('../exp_data/price/pred_arima111.txt')
price_pred_gbm =  np.loadtxt('../exp_data/price/pred_gbm.txt')
price_pred_holt =  np.loadtxt('../exp_data/price/pred_holt.txt')
price_pred_naive =  np.loadtxt('../exp_data/price/pred_naive.txt')
price_true =  np.loadtxt('../exp_data/price/true.txt')

sales_in_sample = np.loadtxt('../exp_data/sales/in_sample.txt')
sales_bm_pred =  np.loadtxt('../exp_data/sales/pred_arima010.txt')
sales_pred_arima010 =  np.loadtxt('../exp_data/sales/pred_arima010.txt')
sales_pred_arima111 =  np.loadtxt('../exp_data/sales/pred_arima111.txt')
sales_pred_gbm =  np.loadtxt('../exp_data/sales/pred_gbm.txt')
sales_pred_holt =  np.loadtxt('../exp_data/sales/pred_holt.txt')
sales_pred_naive =  np.loadtxt('../exp_data/sales/pred_naive.txt')
sales_true =  np.loadtxt('../exp_data/sales/true.txt')

stock_in_sample = np.loadtxt('../exp_data/stock/in_sample.txt')
stock_bm_pred =  np.loadtxt('../exp_data/stock/pred_arima010.txt')
stock_pred_arima010 =  np.loadtxt('../exp_data/stock/pred_arima010.txt')
stock_pred_arima111 =  np.loadtxt('../exp_data/stock/pred_arima111.txt')
stock_pred_gbm =  np.loadtxt('../exp_data/stock/pred_gbm.txt')
stock_pred_holt =  np.loadtxt('../exp_data/stock/pred_holt.txt')
stock_pred_naive =  np.loadtxt('../exp_data/stock/pred_naive.txt')
stock_true =  np.loadtxt('../exp_data/stock/true.txt')

accu_price_arima010 = accu.get_accuracy_arr(price_true, price_pred_arima010, price_bm_pred, price_in_sample)
accu_price_arima111 = accu.get_accuracy_arr(price_true, price_pred_arima111, price_bm_pred, price_in_sample)
accu_price_gbm = accu.get_accuracy_arr(price_true, price_pred_gbm, price_bm_pred, price_in_sample)
accu_price_holt = accu.get_accuracy_arr(price_true, price_pred_holt, price_bm_pred, price_in_sample)
accu_price_naive = accu.get_accuracy_arr(price_true, price_pred_naive, price_bm_pred, price_in_sample)

accu_sales_arima010 = accu.get_accuracy_arr(sales_true, sales_pred_arima010, sales_bm_pred, sales_in_sample)
accu_sales_arima111 = accu.get_accuracy_arr(sales_true, sales_pred_arima111, sales_bm_pred, sales_in_sample)
accu_sales_gbm = accu.get_accuracy_arr(sales_true, sales_pred_gbm, sales_bm_pred, sales_in_sample)
accu_sales_holt = accu.get_accuracy_arr(sales_true, sales_pred_holt, sales_bm_pred, sales_in_sample)
accu_sales_naive = accu.get_accuracy_arr(sales_true, sales_pred_naive, sales_bm_pred, sales_in_sample)

accu_stock_arima010 = accu.get_accuracy_arr(stock_true, stock_pred_arima010, stock_bm_pred, stock_in_sample)
accu_stock_arima111 = accu.get_accuracy_arr(stock_true, stock_pred_arima111, stock_bm_pred, stock_in_sample)
accu_stock_gbm = accu.get_accuracy_arr(stock_true, stock_pred_gbm, stock_bm_pred, stock_in_sample)
accu_stock_holt = accu.get_accuracy_arr(stock_true, stock_pred_holt, stock_bm_pred, stock_in_sample)
accu_stock_naive = accu.get_accuracy_arr(stock_true, stock_pred_naive, stock_bm_pred, stock_in_sample)

overview = np.hstack((accu_price_arima010.reshape(-1, 1), accu_price_arima111.reshape(-1, 1),
                       accu_price_gbm.reshape(-1, 1), accu_price_holt.reshape(-1, 1), accu_price_naive.reshape(-1, 1),
                       accu_sales_arima010.reshape(-1, 1), accu_sales_arima111.reshape(-1, 1),
                       accu_sales_gbm.reshape(-1, 1), accu_sales_holt.reshape(-1, 1), accu_sales_naive.reshape(-1, 1),
                       accu_stock_arima010.reshape(-1, 1), accu_stock_arima111.reshape(-1, 1),
                       accu_stock_gbm.reshape(-1, 1), accu_stock_holt.reshape(-1, 1), accu_stock_naive.reshape(-1, 1)
                       ))

df = pd.DataFrame(overview)
import df_labels
df = df.set_index(pd.Index(df_labels.get_metrics()))
col_name_lv2 = df_labels.get_models()
df.columns = col_name_lv2
df = df.round(2)
df.to_latex('../exp_data/accuracy_overview.tex',multicolumn_format="c",column_format="l|rrrrr|rrrrr|rrrrr", float_format="{:0.2f}".format, index=True)
