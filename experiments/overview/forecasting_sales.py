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
mod_gbm_lower = lgb.Booster(model_file='../../models/gbm_lower_sales.txt')
mod_gbm_upper = lgb.Booster(model_file='../../models/gbm_upper_sales.txt')

# forecasting
pred_naive = mod_naive.predict(fh, X_test)
pred_arima010 = mod_arima010.predict(fh, X_test)
pred_arima111 = mod_arima111.predict(fh, X_test)
pred_holt = mod_holt.predict(fh, X_test)
pred_gbm = mod_gbm.predict(X_test)
# forecast interval
ints_naive = mod_naive.predict_interval(fh, X_test, coverage=0.9)
lower_naive = ints_naive['Weekly_Sales'][0.9]['lower']
upper_naive = ints_naive['Weekly_Sales'][0.9]['upper']
ints_arima010 = mod_arima010.predict_interval(fh, X_test, coverage=0.9)
lower_arima010 = ints_arima010['Weekly_Sales'][0.9]['lower']
upper_arima010 = ints_arima010['Weekly_Sales'][0.9]['upper']
ints_arima111 = mod_arima111.predict_interval(fh, X_test, coverage=0.9)
lower_arima111 = ints_arima111['Weekly_Sales'][0.9]['lower']
upper_arima111 = ints_arima111['Weekly_Sales'][0.9]['upper']
ints_holt = mod_holt.predict_interval(fh, X_test, coverage=0.9)
lower_holt = ints_holt['Weekly_Sales'][0.9]['lower']
upper_holt = ints_holt['Weekly_Sales'][0.9]['upper']
lower_gbm = mod_gbm_lower.predict(X_test)
upper_gbm = mod_gbm_upper.predict(X_test)

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

df_lower_naive = pd.DataFrame(lower_naive)
df_lower_naive.rename(columns={'Weekly_Sales':'lower_Naive_1'}, inplace=True)
np.savetxt('../exp_data/sales/lower_naive.txt',df_lower_naive.values.ravel())
df_upper_naive = pd.DataFrame(upper_naive)
df_upper_naive.rename(columns={'Weekly_Sales':'upper_Naive_1'}, inplace=True)
np.savetxt('../exp_data/sales/upper_naive.txt',df_upper_naive.values.ravel())
df_lower_arima010 = pd.DataFrame(lower_arima010)
df_lower_arima010.rename(columns={'Weekly_Sales':'lower_ARIMA(0,1,0)'}, inplace=True)
np.savetxt('../exp_data/sales/lower_arima010.txt',df_lower_arima010.values.ravel())
df_upper_arima010 = pd.DataFrame(upper_arima010)
df_upper_arima010.rename(columns={'Weekly_Sales':'upper_ARIMA(0,1,0)'}, inplace=True)
np.savetxt('../exp_data/sales/upper_arima010.txt',df_upper_arima010.values.ravel())
df_lower_arima111 = pd.DataFrame(lower_arima010)
df_lower_arima111.rename(columns={'Weekly_Sales':'lower_ARIMA(1,1,1)'}, inplace=True)
np.savetxt('../exp_data/sales/lower_arima111.txt',df_lower_arima111.values.ravel())
df_upper_arima111 = pd.DataFrame(upper_arima010)
df_upper_arima111.rename(columns={'Weekly_Sales':'upper_ARIMA(1,1,1)'}, inplace=True)
np.savetxt('../exp_data/sales/upper_arima111.txt',df_upper_arima111.values.ravel())
df_lower_holt = pd.DataFrame(lower_holt)
df_lower_holt.rename(columns={'Weekly_Sales':'lower_Holt-Winters'}, inplace=True)
np.savetxt('../exp_data/sales/lower_holt.txt',df_lower_holt.values.ravel())
df_upper_holt = pd.DataFrame(upper_holt)
df_upper_holt.rename(columns={'Weekly_Sales':'upper_Holt-Winters'}, inplace=True)
np.savetxt('../exp_data/sales/upper_holt.txt',df_upper_holt.values.ravel())
df_lower_gbm = pd.DataFrame(lower_gbm)
df_lower_gbm.index = y_test.index
df_lower_gbm.rename(columns={0:'lower_lightGBM'}, inplace=True)
np.savetxt('../exp_data/sales/lower_gbm.txt',df_lower_gbm.values.ravel())
df_upper_gbm = pd.DataFrame(upper_gbm)
df_upper_gbm.index = y_test.index
df_upper_gbm.rename(columns={0:'upper_lightGBM'}, inplace=True)
np.savetxt('../exp_data/sales/upper_gbm.txt',df_upper_gbm.values.ravel())

# visualization
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,6))
plt.plot(df_true, label = "True")
plt.plot(df_pred_naive, label = "Naive_1")
plt.plot(df_pred_arima010, label = "ARIMA(0,1,0)")
#plt.plot(df_pred_arima111, label = "ARIMA(1,1,1)")
plt.plot(df_pred_holt, label = "Holt-Winters")
plt.plot(df_pred_gbm, label = "LightGBM")
x = df_true.index.ravel()
plt.fill_between(x ,lower_naive, upper_naive, alpha=0.1, label = "Naive_1 90% forecast interval")
plt.fill_between(x ,lower_arima010, upper_arima010, alpha=0.1, label = "ARIMA(0,1,0) 90% forecast interval")
#plt.fill_between(x ,lower_arima111, upper_arima111, alpha=0.1, label = "ARIMA(1,1,1) 90% forecast interval")
plt.fill_between(x ,lower_holt, upper_holt, alpha=0.1, label = "Holt-Winters 90% forecast interval")
plt.fill_between(x ,lower_gbm, upper_gbm, alpha=0.1, label = "LightGBM 90% forecast interval")
plt.legend(loc="upper left", ncol = 2)
plt.show()
fig.savefig('../../experiments_plots/overview/sales.png')


