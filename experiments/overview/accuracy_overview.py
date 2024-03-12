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
price_lower_arima010 =  np.loadtxt('../exp_data/price/lower_arima010.txt')
price_lower_arima111 =  np.loadtxt('../exp_data/price/lower_arima111.txt')
price_lower_gbm =  np.loadtxt('../exp_data/price/lower_gbm.txt')
price_lower_holt =  np.loadtxt('../exp_data/price/lower_holt.txt')
price_lower_naive =  np.loadtxt('../exp_data/price/lower_naive.txt')
price_upper_arima010 =  np.loadtxt('../exp_data/price/upper_arima010.txt')
price_upper_arima111 =  np.loadtxt('../exp_data/price/upper_arima111.txt')
price_upper_gbm =  np.loadtxt('../exp_data/price/upper_gbm.txt')
price_upper_holt =  np.loadtxt('../exp_data/price/upper_holt.txt')
price_upper_naive =  np.loadtxt('../exp_data/price/upper_naive.txt')

sales_in_sample = np.loadtxt('../exp_data/sales/in_sample.txt')
sales_bm_pred =  np.loadtxt('../exp_data/sales/pred_arima010.txt')
sales_pred_arima010 =  np.loadtxt('../exp_data/sales/pred_arima010.txt')
sales_pred_arima111 =  np.loadtxt('../exp_data/sales/pred_arima111.txt')
sales_pred_gbm =  np.loadtxt('../exp_data/sales/pred_gbm.txt')
sales_pred_holt =  np.loadtxt('../exp_data/sales/pred_holt.txt')
sales_pred_naive =  np.loadtxt('../exp_data/sales/pred_naive.txt')
sales_true =  np.loadtxt('../exp_data/sales/true.txt')
sales_lower_arima010 =  np.loadtxt('../exp_data/sales/lower_arima010.txt')
sales_lower_arima111 =  np.loadtxt('../exp_data/sales/lower_arima111.txt')
sales_lower_gbm =  np.loadtxt('../exp_data/sales/lower_gbm.txt')
sales_lower_holt =  np.loadtxt('../exp_data/sales/lower_holt.txt')
sales_lower_naive =  np.loadtxt('../exp_data/sales/lower_naive.txt')
sales_upper_arima010 =  np.loadtxt('../exp_data/sales/upper_arima010.txt')
sales_upper_arima111 =  np.loadtxt('../exp_data/sales/upper_arima111.txt')
sales_upper_gbm =  np.loadtxt('../exp_data/sales/upper_gbm.txt')
sales_upper_holt =  np.loadtxt('../exp_data/sales/upper_holt.txt')
sales_upper_naive =  np.loadtxt('../exp_data/sales/upper_naive.txt')

stock_in_sample = np.loadtxt('../exp_data/stock/in_sample.txt')
stock_bm_pred =  np.loadtxt('../exp_data/stock/pred_arima010.txt')
stock_pred_arima010 =  np.loadtxt('../exp_data/stock/pred_arima010.txt')
stock_pred_arima111 =  np.loadtxt('../exp_data/stock/pred_arima111.txt')
stock_pred_gbm =  np.loadtxt('../exp_data/stock/pred_gbm.txt')
stock_pred_holt =  np.loadtxt('../exp_data/stock/pred_holt.txt')
stock_pred_naive =  np.loadtxt('../exp_data/stock/pred_naive.txt')
stock_true =  np.loadtxt('../exp_data/stock/true.txt')
stock_lower_arima010 =  np.loadtxt('../exp_data/stock/lower_arima010.txt')
stock_lower_arima111 =  np.loadtxt('../exp_data/stock/lower_arima111.txt')
stock_lower_gbm =  np.loadtxt('../exp_data/stock/lower_gbm.txt')
stock_lower_holt =  np.loadtxt('../exp_data/stock/lower_holt.txt')
stock_lower_naive =  np.loadtxt('../exp_data/stock/lower_naive.txt')
stock_upper_arima010 =  np.loadtxt('../exp_data/stock/upper_arima010.txt')
stock_upper_arima111 =  np.loadtxt('../exp_data/stock/upper_arima111.txt')
stock_upper_gbm =  np.loadtxt('../exp_data/stock/upper_gbm.txt')
stock_upper_holt =  np.loadtxt('../exp_data/stock/upper_holt.txt')
stock_upper_naive =  np.loadtxt('../exp_data/stock/upper_naive.txt')

accu_price_arima010 = accu.get_accuracy_arr(y_true=price_true, y_pred=price_pred_arima010, 
                                            bm_pred=price_bm_pred, y_in_sample=price_in_sample, 
                                            lower=price_lower_arima010, upper=price_upper_arima010)
accu_price_arima111 = accu.get_accuracy_arr(y_true=price_true, y_pred=price_pred_arima111,
                                            bm_pred=price_bm_pred, y_in_sample=price_in_sample,
                                            lower=price_lower_arima111, upper=price_upper_arima111)
accu_price_gbm = accu.get_accuracy_arr(y_true=price_true, y_pred=price_pred_gbm, 
                                       bm_pred=price_bm_pred, y_in_sample=price_in_sample, 
                                       lower=price_lower_gbm, upper=price_upper_gbm)
accu_price_holt = accu.get_accuracy_arr(y_true=price_true, y_pred=price_pred_holt,
                                        bm_pred=price_bm_pred, y_in_sample=price_in_sample,
                                        lower=price_lower_holt, upper=price_upper_holt)
accu_price_naive = accu.get_accuracy_arr(y_true=price_true, y_pred=price_pred_naive, 
                                         bm_pred=price_bm_pred, y_in_sample=price_in_sample, 
                                         lower=price_lower_naive, upper=price_upper_naive)

accu_sales_arima010 = accu.get_accuracy_arr(y_true=sales_true, y_pred=sales_pred_arima010, 
                                            bm_pred=sales_bm_pred, y_in_sample=sales_in_sample, 
                                            lower=sales_lower_arima010, upper=sales_upper_arima010)
accu_sales_arima111 = accu.get_accuracy_arr(y_true=sales_true, y_pred=sales_pred_arima111, 
                                            bm_pred=sales_bm_pred, y_in_sample=sales_in_sample, 
                                            lower=sales_lower_arima111, upper=sales_upper_arima111)
accu_sales_gbm = accu.get_accuracy_arr(y_true=sales_true, y_pred=sales_pred_gbm, 
                                       bm_pred=sales_bm_pred, y_in_sample=sales_in_sample, 
                                       lower=sales_lower_gbm, upper=sales_upper_gbm)
accu_sales_holt = accu.get_accuracy_arr(y_true=sales_true, y_pred=sales_pred_holt, 
                                       bm_pred=sales_bm_pred, y_in_sample=sales_in_sample, 
                                       lower=sales_lower_holt, upper=sales_upper_holt)
accu_sales_naive = accu.get_accuracy_arr(y_true=sales_true, y_pred=sales_pred_naive, 
                                       bm_pred=sales_bm_pred, y_in_sample=sales_in_sample, 
                                       lower=sales_lower_naive, upper=sales_upper_naive)

accu_stock_arima010 = accu.get_accuracy_arr(y_true=stock_true, y_pred=stock_pred_arima010, 
                                            bm_pred=stock_bm_pred, y_in_sample=stock_in_sample, 
                                            lower=stock_lower_arima010, upper=stock_upper_arima010)
accu_stock_arima111 = accu.get_accuracy_arr(y_true=stock_true, y_pred=stock_pred_arima111, 
                                            bm_pred=stock_bm_pred, y_in_sample=stock_in_sample, 
                                            lower=stock_lower_arima111, upper=stock_upper_arima111)
accu_stock_gbm = accu.get_accuracy_arr(y_true=stock_true, y_pred=stock_pred_gbm, 
                                       bm_pred=stock_bm_pred, y_in_sample=stock_in_sample, 
                                       lower=stock_lower_gbm, upper=stock_upper_gbm)
accu_stock_holt = accu.get_accuracy_arr(y_true=stock_true, y_pred=stock_pred_holt, 
                                        bm_pred=stock_bm_pred, y_in_sample=stock_in_sample, 
                                        lower=stock_lower_holt, upper=stock_upper_holt)
accu_stock_naive = accu.get_accuracy_arr(y_true=stock_true, y_pred=stock_pred_naive, 
                                        bm_pred=stock_bm_pred, y_in_sample=stock_in_sample, 
                                        lower=stock_lower_naive, upper=stock_upper_naive)

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
df.to_latex('../exp_data/accuracy_overview.tex',multicolumn_format="c",column_format="l|rrrrr|rrrrr|rrrrr", 
            float_format="{:0.2f}".format, index=True)
