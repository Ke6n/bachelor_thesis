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
variance = []
window_size = 30

sys.path.append("..")
import plotting
for metric in metrics:
    # Calculate the set of a metrics
    acc_arr = accuracy.get_accuracy_set(metric, window_size, 
                              y_true=true, y_pred=y_pred, 
                              bm_pred=bm_pred, y_in_sample=y_in_sample, 
                              lower=lower, upper=upper)
    # plotting
    save_path = f'../../experiments_plots/violin_plots/{metric}.png'
    plotting.violin_plotting(save_path, metric, acc_arr)

    #variance
    var_acc = np.var(acc_arr)
    variance.append(var_acc)

df = pd.DataFrame({'Metrics': metrics, 'Variance': variance})
#df = df.round(2)
df.to_latex('../exp_data/variance.tex', float_format="{:0.2e}".format, index=False)
