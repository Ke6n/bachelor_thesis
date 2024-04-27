#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
sys.path.append("../..")
import utils.accuracy as accu
from pymcdm.methods import PROMETHEE_II
import plotting

df_acc_price = pd.read_csv('../exp_data/mean_price.csv', index_col=['Unnamed: 0'])
df_acc_sales = pd.read_csv('../exp_data/mean_sales.csv', index_col=['Unnamed: 0'])
df_acc_stock = pd.read_csv('../exp_data/mean_stock.csv', index_col=['Unnamed: 0'])

# Sorting using single metric
def single_criterion_ranking(df: pd.DataFrame):
    df_asc = df.drop(columns=["PTSU", "PCDCP"]).rank(method='dense')
    df_desc = df[["PTSU", "PCDCP"]].rank(ascending=False, method='dense')
    sig_rank = pd.concat([df_asc, df_desc], axis=1).transpose()
    return sig_rank

sig_rank_price = single_criterion_ranking(df_acc_price)
save_path = '../../experiments_plots/model_ranking/price.png'
plotting.line_plotting(save_path, sig_rank_price)

sig_rank_sales = single_criterion_ranking(df_acc_sales)
save_path = '../../experiments_plots/model_ranking/sales.png'
plotting.line_plotting(save_path, sig_rank_sales)

sig_rank_stock = single_criterion_ranking(df_acc_stock)
save_path = '../../experiments_plots/model_ranking/stock.png'
plotting.line_plotting(save_path, sig_rank_stock)

# Sorting using mcda method
# The preference function see:
#   Xu B, Ouenniche J. Performance evaluation of competing forecasting models: A multidimensional framework based on MCDA[J]. 
#   Expert Systems with Applications, 2012, 39(9): 8312-8324.
body = PROMETHEE_II('vshape')
weights = [0.5, 0.3, 0.2]
types = np.array([-1, -1, 1])

#   Compute thresholds in preference functions 
def threshold(df:pd.DataFrame, criterion:str, coef):
    return coef*(df[criterion].max()- df[criterion].min())

# The coefficients of thresholds see also:
#   Xu B, Ouenniche J. Performance evaluation of competing forecasting models: A multidimensional framework based on MCDA[J]. 
#   Expert Systems with Applications, 2012, 39(9): 8312-8324.
q2_price = threshold(df_acc_price, 'PCDCP', 0.01)
q3_price = threshold(df_acc_price, 'PTSU', 0.2)
p2_price = threshold(df_acc_price, 'PCDCP', 0.05)
p3_price = threshold(df_acc_price, 'PTSU', 0.33)

q2_sales = threshold(df_acc_sales, 'PCDCP', 0.01)
q3_sales = threshold(df_acc_sales, 'PTSU', 0.2)
p2_sales = threshold(df_acc_sales, 'PCDCP', 0.05)
p3_sales = threshold(df_acc_sales, 'PTSU', 0.33)

q2_stock = threshold(df_acc_stock, 'PCDCP', 0.01)
q3_stock = threshold(df_acc_stock, 'PTSU', 0.2)
p2_stock = threshold(df_acc_stock, 'PCDCP', 0.05)
p3_stock = threshold(df_acc_stock, 'PTSU', 0.33)

mcda_rank_price = pd.DataFrame()
mcda_rank_sales = pd.DataFrame()
mcda_rank_stock = pd.DataFrame()

import utils.df_labels as labels
metrics = labels.get_metrics()
for goodness_of_fit in metrics[0:-2]:
    criteria = [goodness_of_fit, 'PCDCP', 'PTSU']
    matrix_price = df_acc_price[criteria].values
    matrix_sales = df_acc_sales[criteria].values
    matrix_stock = df_acc_stock[criteria].values
  
    q1_price = threshold(df_acc_price, goodness_of_fit, 0.01)
    q_price = np.array([q1_price, q2_price, q3_price])
    p1_price = threshold(df_acc_price, goodness_of_fit, 0.05)
    p_price = np.array([p1_price, p2_price, p3_price])

    q1_sales = threshold(df_acc_sales, goodness_of_fit, 0.01)
    q_sales = np.array([q1_sales, q2_sales, q3_sales])
    p1_sales = threshold(df_acc_sales, goodness_of_fit, 0.05)
    p_sales = np.array([p1_sales, p2_sales, p3_sales])
  
    q1_stock = threshold(df_acc_stock, goodness_of_fit, 0.01)
    q_stock = np.array([q1_stock, q2_stock, q3_stock])
    p1_stock = threshold(df_acc_stock, goodness_of_fit, 0.05)
    p_stock = np.array([p1_stock, p2_stock, p3_stock])

    col_price = body(matrix_price, weights, types, p=p_price, q=q_price)
    col_sales = body(matrix_sales, weights, types, p=p_sales, q=q_sales)
    col_stock = body(matrix_stock, weights, types, p=p_stock, q=q_stock)
    col_name = ','.join(criteria)
    mcda_rank_price[col_name] = col_price
    mcda_rank_sales[col_name] = col_sales
    mcda_rank_stock[col_name] = col_stock

mcda_rank_price.index = df_acc_price.index
mcda_rank_sales.index = df_acc_sales.index
mcda_rank_stock.index = df_acc_stock.index

mcda_rank_price = mcda_rank_price.rank(ascending=False, method='dense').transpose()
mcda_rank_sales = mcda_rank_sales.rank(ascending=False, method='dense').transpose()
mcda_rank_stock = mcda_rank_stock.rank(ascending=False, method='dense').transpose()

save_path = '../../experiments_plots/model_ranking/mcda_price.png'
plotting.line_plotting(save_path, mcda_rank_price, fontsize=6.5)

save_path = '../../experiments_plots/model_ranking/mcda_sales.png'
plotting.line_plotting(save_path, mcda_rank_sales, fontsize=6.5)

save_path = '../../experiments_plots/model_ranking/mcda_stock.png'
plotting.line_plotting(save_path, mcda_rank_stock, fontsize=6.5)