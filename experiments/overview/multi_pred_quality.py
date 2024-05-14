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

# A function that calculates the accuracy of a model for a data set
import sys
sys.path.append("../..")
import utils.accuracy as accu
import utils.df_labels as df_labels
metrics = df_labels.get_metrics()
def accuracy(df_true, df_pred, df_in_sample, df_bm_pred, n=30):
    acc_np = np.empty((0, len(metrics)))
    for i in range(n):
        index_label = df_true.index.get_level_values(0).unique()[i]
        true = df_true.loc[index_label].values.ravel()
        pred = df_pred.loc[index_label].values.ravel()
        in_sample = df_in_sample.loc[index_label].values.ravel()
        bm_pred = df_bm_pred.loc[index_label].values.ravel()
        acc_arr = accu.get_accuracy_arr(y_true=true, y_pred=pred, 
                                    bm_pred=bm_pred, y_in_sample=in_sample)
        acc_np = np.vstack([acc_np, acc_arr])
    return acc_np

# Evaluate accuracy for each data set
models = ['ARIMA','Holt-Winters','LightGBM']
#--- price ---
acc_price_arima = accuracy(price_true, price_pred_arima, price_in_sample, price_bm_pred)
acc_price_holt = accuracy(price_true, price_pred_holt, price_in_sample, price_bm_pred)
acc_price_gbm = accuracy(price_true, price_pred_gbm, price_in_sample, price_bm_pred)

mean_acc_price = np.mean(acc_price_arima, axis=0)
mean_acc_price = np.vstack([mean_acc_price, np.mean(acc_price_holt, axis=0)])
mean_acc_price = np.vstack([mean_acc_price, np.mean(acc_price_gbm, axis=0)])
mean_acc_price = pd.DataFrame(mean_acc_price, index=models, columns=metrics).round(3)

var_acc_price = np.var(acc_price_arima, axis=0)
var_acc_price = np.vstack([var_acc_price, np.var(acc_price_holt, axis=0)])
var_acc_price = np.vstack([var_acc_price, np.var(acc_price_gbm, axis=0)])
var_acc_price = pd.DataFrame(var_acc_price, index=models, columns=metrics).round(3)

#--- sales ---
acc_sales_arima = accuracy(sales_true, sales_pred_arima, sales_in_sample, sales_bm_pred)
acc_sales_holt = accuracy(sales_true, sales_pred_holt, sales_in_sample, sales_bm_pred)
acc_sales_gbm = accuracy(sales_true, sales_pred_gbm, sales_in_sample, sales_bm_pred)

mean_acc_sales = np.mean(acc_sales_arima, axis=0)
mean_acc_sales = np.vstack([mean_acc_sales, np.mean(acc_sales_holt, axis=0)])
mean_acc_sales = np.vstack([mean_acc_sales, np.mean(acc_sales_gbm, axis=0)])
mean_acc_sales = pd.DataFrame(mean_acc_sales, index=models, columns=metrics).round(3)

var_acc_sales = np.var(acc_sales_arima, axis=0)
var_acc_sales = np.vstack([var_acc_sales, np.var(acc_sales_holt, axis=0)])
var_acc_sales = np.vstack([var_acc_sales, np.var(acc_sales_gbm, axis=0)])
var_acc_sales = pd.DataFrame(var_acc_sales, index=models, columns=metrics).round(3)

#--- stock ---
acc_stock_arima = accuracy(stock_true, stock_pred_arima, stock_in_sample, stock_bm_pred)
acc_stock_holt = accuracy(stock_true, stock_pred_holt, stock_in_sample, stock_bm_pred)
acc_stock_gbm = accuracy(stock_true, stock_pred_gbm, stock_in_sample, stock_bm_pred)

mean_acc_stock = np.mean(acc_stock_arima, axis=0)
mean_acc_stock = np.vstack([mean_acc_stock, np.mean(acc_stock_holt, axis=0)])
mean_acc_stock = np.vstack([mean_acc_stock, np.mean(acc_stock_gbm, axis=0)])
mean_acc_stock = pd.DataFrame(mean_acc_stock, index=models, columns=metrics).round(3)

var_acc_stock = np.var(acc_stock_arima, axis=0)
var_acc_stock = np.vstack([var_acc_stock, np.var(acc_stock_holt, axis=0)])
var_acc_stock = np.vstack([var_acc_stock, np.var(acc_stock_gbm, axis=0)])
var_acc_stock = pd.DataFrame(var_acc_stock, index=models, columns=metrics).round(3)

# Generate latex table
col_format = 'c' * (len(metrics) + 1)
mean_acc_price.T.to_latex('../exp_data/mean_price.tex', column_format=col_format, index=True, float_format="{:0.3f}".format)
mean_acc_sales.T.to_latex('../exp_data/mean_sales.tex', column_format=col_format, index=True, float_format="{:0.3e}".format)
mean_acc_stock.T.to_latex('../exp_data/mean_stock.tex', column_format=col_format, index=True, float_format="{:0.3f}".format)
# var_acc_price.to_latex('../exp_data/var_price.tex', column_format=col_format, index=True, float_format="{:0.3e}".format)
# var_acc_sales.to_latex('../exp_data/var_sales.tex', column_format=col_format, index=True, float_format="{:0.3e}".format)
# var_acc_stock.to_latex('../exp_data/var_stock.tex', column_format=col_format, index=True, float_format="{:0.3e}".format)

# save to csv
mean_acc_price.to_csv('../exp_data/mean_price.csv')
mean_acc_sales.to_csv('../exp_data/mean_sales.csv')
mean_acc_stock.to_csv('../exp_data/mean_stock.csv')

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

def barplot(file_name, df, colors=["green", "blue", "yellow"]):
    metrics = df_labels.get_metrics()
    models = df_labels.get_main_models()
    fontsize = 18
    groups = [df.iloc[:, i:i+4] for i in range(0, 24, 4)]

    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(18, 26), constrained_layout=True)
    for i in range(6):
        for j in range(4):
            metric = metrics[i * 4 + j]
            ax = axs[i, j]
            group_data = groups[i][metric]
            ax.barh(models, group_data.values.flatten(), color = colors)
            ax.set_title(metric, fontsize = fontsize-2)
            ax.tick_params(axis='both', labelsize=fontsize-3)
            ax.xaxis.offsetText.set_fontsize(fontsize-3)
            
    for i in range(6):
        axs[i, 0].set_ylabel('Model', fontsize=fontsize-3)

    plt.tight_layout()
    plt.savefig(f'../../experiments_plots/overview/{file_name}.png')
    plt.show()


# price
barplot(file_name='multi_price_acc', df = mean_acc_price)

# sales
barplot(file_name='multi_sales_acc', df = mean_acc_sales)

# stock
barplot(file_name='multi_stock_acc', df = mean_acc_stock)