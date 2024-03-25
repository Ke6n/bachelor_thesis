#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
sys.path.append("../..")
import utils.accuracy as accu
from pymcdm.methods import PROMETHEE_II
import plotting

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

accu_price = np.hstack((accu_price_arima010.reshape(-1, 1), accu_price_arima111.reshape(-1, 1),
                       accu_price_gbm.reshape(-1, 1), accu_price_holt.reshape(-1, 1), accu_price_naive.reshape(-1, 1)
                       ))

accu_sales = np.hstack((accu_sales_arima010.reshape(-1, 1), accu_sales_arima111.reshape(-1, 1),
                       accu_sales_gbm.reshape(-1, 1), accu_sales_holt.reshape(-1, 1), accu_sales_naive.reshape(-1, 1)
                       ))

models = ["ARIMA(010)*", "ARIMA(111)", "LightGBM", "Hot Winters", "Naive 1"]
metrics = ["MAE", "MdAE", "MSE", "RMSE", "MAPE", "MdAPE", "RMSPE", "RMdSPE", "wMAPE", "sMAPE", "sMdAPE", "msMAPE",
            "MRAE", "MdRAE", "GMRAE", "UMBRAE", "RMAE", "RRMSE", "LMR", "MASE", "MdASE", "RMSSE", "MSIS","PTSU", "PCDCP"]

df_accu_price = pd.DataFrame(accu_price).transpose()
df_accu_price.index = models
df_accu_price.columns = metrics
df_accu_sales = pd.DataFrame(accu_sales).transpose()
df_accu_sales.index = models
df_accu_sales.columns = metrics

# Sorting using single metric
df_asc = df_accu_price.drop(columns=["PTSU", "PCDCP"]).rank(method='dense')
df_desc = df_accu_price[["PTSU", "PCDCP"]].rank(ascending=False, method='dense')
sig_rank_price = pd.concat([df_asc, df_desc], axis=1).transpose()

save_path = '../../experiments_plots/model_ranking/price.png'
plotting.line_plotting(save_path, sig_rank_price)

df_asc = df_accu_sales.drop(columns=["PTSU", "PCDCP"]).rank(method='dense')
df_desc = df_accu_sales[["PTSU", "PCDCP"]].rank(ascending=False, method='dense')
sig_rank_sales = pd.concat([df_asc, df_desc], axis=1).transpose()

save_path = '../../experiments_plots/model_ranking/sales.png'
plotting.line_plotting(save_path, sig_rank_sales)

# Sorting using mcda method
body = PROMETHEE_II('vshape')
weights = [0.5, 0.3, 0.2]
types = np.array([-1, -1, 1])

#   Compute thresholds in preference functions 
def threshold(df:pd.DataFrame, criterion:str, coef):
    return coef*(df[criterion].max()- df[criterion].min())

# The coefficients of thresholds see:
#   Xu B, Ouenniche J. Performance evaluation of competing forecasting models: A multidimensional framework based on MCDA[J]. 
#   Expert Systems with Applications, 2012, 39(9): 8312-8324.
q2_price = threshold(df_accu_price, 'PCDCP', 0.01)
q3_price = threshold(df_accu_price, 'PTSU', 0.2)
p2_price = threshold(df_accu_price, 'PCDCP', 0.05)
p3_price = threshold(df_accu_price, 'PTSU', 0.33)
q2_sales = threshold(df_accu_sales, 'PCDCP', 0.01)
q3_sales = threshold(df_accu_sales, 'PTSU', 0.2)
p2_sales = threshold(df_accu_sales, 'PCDCP', 0.05)
p3_sales = threshold(df_accu_sales, 'PTSU', 0.33)

mcda_rank_price = pd.DataFrame()
mcda_rank_sales = pd.DataFrame()

for goodness_of_fit in metrics[0:-2]:
    criteria = [goodness_of_fit, 'PCDCP', 'PTSU']
    matrix_price = df_accu_price[criteria].values
    matrix_sales = df_accu_sales[criteria].values
    q1_price = threshold(df_accu_price, goodness_of_fit, 0.01)
    q_price = np.array([q1_price, q2_price, q3_price])
    p1_price = threshold(df_accu_price, goodness_of_fit, 0.05)
    p_price = np.array([p1_price, p2_price, p3_price])
    q1_sales = threshold(df_accu_sales, goodness_of_fit, 0.01)
    q_sales = np.array([q1_price, q2_price, q3_price])
    p1_sales = threshold(df_accu_sales, goodness_of_fit, 0.05)
    p_sales = np.array([p1_price, p2_price, p3_price])
    col_price = body(matrix_price, weights, types, p=p_price, q=q_price)
    col_sales = body(matrix_sales, weights, types, p=p_sales, q=q_sales)
    col_name = ','.join(criteria)
    mcda_rank_price[col_name] = col_price
    mcda_rank_sales[col_name] = col_sales

mcda_rank_price.index = models
mcda_rank_sales.index = models

mcda_rank_price = mcda_rank_price.rank(ascending=False, method='dense').transpose()
mcda_rank_sales = mcda_rank_sales.rank(ascending=False, method='dense').transpose()

save_path = '../../experiments_plots/model_ranking/mcda_price.png'
plotting.line_plotting(save_path, mcda_rank_price)

save_path = '../../experiments_plots/model_ranking/mcda_sales.png'
plotting.line_plotting(save_path, mcda_rank_sales)