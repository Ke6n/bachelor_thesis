#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
sys.path.append("../..")
import utils.split as split
import pickle
from sktime.forecasting.base import ForecastingHorizon

df_sales = pd.read_csv('../../processed_data/sales.csv')
y_train, y_test, X_train, X_test = split.split(df_sales, 'Weekly_Sales')
fh = ForecastingHorizon(y_test.index, is_relative=False)

# reading models
file1= open('../../models/naive1_sales', 'rb')
mod_naive = pickle.load(file1)
file1.close()

file2= open('../../models/arima010_sales', 'rb')
mod_arima010 = pickle.load(file2)
file2.close()

file3= open('../../models/arima111_sales', 'rb')
mod_arima111 = pickle.load(file3)
file3.close()

file4= open('../../models/holt_sales', 'rb')
mod_holt = pickle.load(file4)
file4.close()

import lightgbm as lgb
mod_gbm = lgb.Booster(model_file='../../models/gbm_sales.txt')

# forecasting
pred_naive = mod_naive.predict(fh, X_test)
pred_arima010 = mod_arima010.predict(fh, X_test)
pred_arima111 = mod_arima111.predict(fh, X_test)
pred_holt = mod_holt.predict(fh, X_test)
pred_gbm = mod_gbm.predict(X_test)

# persistenz
in_sample = pd.DataFrame(y_train)
np.savetxt('../exp_data/sales/in_sample.txt',in_sample.transpose().values[0])

df_true = pd.DataFrame(y_test)
df_true.rename(columns={'Weekly_Sales':'True'}, inplace=True)
np.savetxt('../exp_data/sales/true.txt',df_true.transpose().values[0])

df_pred_naive = pd.DataFrame(pred_naive)
df_pred_naive.rename(columns={'Weekly_Sales':'Naive_1'}, inplace=True)
np.savetxt('../exp_data/sales/pred_naive.txt',df_pred_naive.transpose().values[0])

df_pred_arima010 = pd.DataFrame(pred_arima010)
df_pred_arima010.rename(columns={'Weekly_Sales':'ARIMA(0,1,0)'}, inplace=True)
np.savetxt('../exp_data/sales/pred_arima010.txt',df_pred_arima010.transpose().values[0])

df_pred_arima111 = pd.DataFrame(pred_arima111)
df_pred_arima111.rename(columns={'Weekly_Sales':'ARIMA(1,1,1)'}, inplace=True)
np.savetxt('../exp_data/sales/pred_arima111.txt',df_pred_arima111.transpose().values[0])

df_pred_holt = pd.DataFrame(pred_holt)
df_pred_holt.rename(columns={'Weekly_Sales':'Holt-Winters'}, inplace=True)
np.savetxt('../exp_data/sales/pred_holt.txt',df_pred_holt.transpose().values[0])

df_pred_gbm = pd.DataFrame(pred_gbm)
df_pred_gbm.index = y_test.index
df_pred_gbm.rename(columns={0:'lightGBM'}, inplace=True)
np.savetxt('../exp_data/sales/pred_gbm.txt',df_pred_gbm.transpose().values[0])

# visualization
df_pred_overview = pd.concat([df_true, df_pred_naive,df_pred_arima010, df_pred_arima111, df_pred_holt, df_pred_gbm], axis=1)
graph = df_pred_overview.plot(figsize=(15,6), title="Walmart sales prediction overview")

fig = graph.get_figure()
fig.savefig('../../experiments_plots/overview/sales.png')


