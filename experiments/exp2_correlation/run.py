#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
sys.path.append("../..")
import utils.accuracy as accuracy

y_in_sample = np.loadtxt('../exp_data/stock/in_sample.txt')
bm_pred =  np.loadtxt('../exp_data/stock/pred_arima010.txt')
y_pred =  np.loadtxt('../exp_data/stock/pred_naive.txt')
true =  np.loadtxt('../exp_data/stock/true.txt')
lower =  np.loadtxt('../exp_data/stock/lower_naive.txt')
upper =  np.loadtxt('../exp_data/stock/upper_naive.txt')

metrics = ["MAE", "MdAE", "MSE", "RMSE", "MAPE", "MdAPE", "RMSPE", "RMdSPE", "wMAPE", "sMAPE", "sMdAPE", "msMAPE",
            "MRAE", "MdRAE", "GMRAE", "UMBRAE", "RMAE", "RRMSE", "LMR", "MASE", "MdASE", "RMSSE", "MSIS","PTSU", "PCDCP"]

window_size = 30

input_array = np.array([accuracy.get_accuracy_set(metric, window_size, 
                              y_true=true, y_pred=y_pred, 
                              bm_pred=bm_pred, y_in_sample=y_in_sample, 
                              lower=lower, upper=upper) 
                              for metric in metrics])
input_df = pd.DataFrame(input_array)
input_df.index = metrics
input_df = input_df.transpose()

# Heatmap
corr_matrix = input_df.corr()
corr_matrix.columns = metrics
corr_matrix.index = metrics
save_path = '../../experiments_plots/correlation/heatmap.png'
import plotting
plotting.heatmap_plotting(save_path, corr_matrix)

# Scatter Matrix
save_path = '../../experiments_plots/correlation/scatter_matrix.png'
plotting.scatter_matrix_plotting(save_path, input_df)