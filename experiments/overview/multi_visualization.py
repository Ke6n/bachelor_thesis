#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the results of predictions
# --- price ---
price_true = pd.read_csv('../exp_data/price/multi_true.csv')
price_arima =  pd.read_csv('../exp_data/price/multi_pred_arima.csv')
price_pred_holt = pd.read_csv('../exp_data/price/multi_pred_holt.csv')
price_pred_gbm = pd.read_csv('../exp_data/price/multi_pred_gbm.csv') 

price_true['Model']='Actual'
price_arima['Model']='ARIMA'
price_pred_holt['Model']='Holt-Winter'
price_pred_gbm['Model']='LightGBM'
df_price = pd.concat([price_true, price_arima, price_pred_holt, price_pred_gbm])

#--- sales ---
sales_true = pd.read_csv('../exp_data/sales/multi_true.csv')
sales_arima =  pd.read_csv('../exp_data/sales/multi_pred_arima.csv')
sales_pred_holt = pd.read_csv('../exp_data/sales/multi_pred_holt.csv')
sales_pred_gbm = pd.read_csv('../exp_data/sales/multi_pred_gbm.csv') 

sales_true['Model']='Actual'
sales_arima['Model']='ARIMA'
sales_pred_holt['Model']='Holt-Winter'
sales_pred_gbm['Model']='LightGBM'
df_sales = pd.concat([sales_true, sales_arima, sales_pred_holt, sales_pred_gbm])

#--- stock ---
stock_true = pd.read_csv('../exp_data/stock/multi_true.csv')
stock_arima =  pd.read_csv('../exp_data/stock/multi_pred_arima.csv')
stock_pred_holt = pd.read_csv('../exp_data/stock/multi_pred_holt.csv')
stock_pred_gbm = pd.read_csv('../exp_data/stock/multi_pred_gbm.csv') 

stock_true['Model']='Actual'
stock_arima['Model']='ARIMA'
stock_pred_holt['Model']='Holt-Winter'
stock_pred_gbm['Model']='LightGBM'
df_stock = pd.concat([stock_true, stock_arima, stock_pred_holt, stock_pred_gbm])


# Box-Plot of Predictions
fig = plt.figure(figsize=(15,6))
#--- price ---
sns.boxplot(x = 'Model', y= 'AveragePrice', data=df_price)
plt.savefig('../../experiments_plots/overview/multi_price.png')
plt.show()
#--- sales ---
sns.boxplot(x = 'Model', y= 'Weekly_Sales', data=df_sales)
plt.savefig('../../experiments_plots/overview/multi_sales.png')
plt.show()
#--- stock ---
sns.boxplot(x = 'Model', y= 'Price', data=df_stock)
plt.savefig('../../experiments_plots/overview/multi_stock.png')
plt.show()