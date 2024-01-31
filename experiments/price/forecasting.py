#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
import pickle
from sktime.forecasting.base import ForecastingHorizon

df_price = pd.read_csv('../../processed_data/price.csv')
y_train, y_test, X_train, X_test = split.split(df_price, 'AveragePrice')
fh = ForecastingHorizon(y_test.index, is_relative=False)

# reading models
file1= open('../../models/naive1_price', 'rb')
mod_naive = pickle.load(file1)
file1.close()

file2= open('../../models/arima110_price', 'rb')
mod_arima110 = pickle.load(file2)
file2.close()

file3= open('../../models/arima121_price', 'rb')
mod_arima121 = pickle.load(file3)
file3.close()

file4= open('../../models/holt_price', 'rb')
mod_holt = pickle.load(file4)
file4.close()

import lightgbm as lgb
mod_gbm = lgb.Booster(model_file='../../models/gbm_price.txt')

# forecasting
pred_naive = mod_naive.predict(fh, X_test)
pred_arima110 = mod_arima110.predict(fh, X_test)
pred_arima121 = mod_arima121.predict(fh, X_test)
pred_holt = mod_holt.predict(fh, X_test)
pred_gbm = mod_gbm.predict(X_test)

# persistenz
true = pd.DataFrame(y_test)
df_true = pd.DataFrame(true)
df_true.rename(columns={'AveragePrice':'True'}, inplace=True)
df_true.to_json('experimental_data/true.json')

df_pred_naive = pd.DataFrame(pred_naive)
df_pred_naive.rename(columns={'AveragePrice':'Naive_1'}, inplace=True)
df_pred_naive.to_json('experimental_data/pred_naive.json')

df_pred_arima110 = pd.DataFrame(pred_arima110)
df_pred_arima110.rename(columns={'AveragePrice':'ARIMA(1,1,0)'}, inplace=True)
df_pred_arima110.to_json('experimental_data/pred_arima110.json')

df_pred_arima121 = pd.DataFrame(pred_arima121)
df_pred_arima121.rename(columns={'AveragePrice':'ARIMA(1,2,1)'}, inplace=True)
df_pred_arima121.to_json('experimental_data/pred_arima121.json')

df_pred_holt = pd.DataFrame(pred_holt)
df_pred_holt.rename(columns={'AveragePrice':'Holt-Winters'}, inplace=True)
df_pred_holt.to_json('experimental_data/pred_holt.json')

df_pred_gbm = pd.DataFrame(pred_gbm)
df_pred_gbm.index = y_test.index
df_pred_gbm.rename(columns={0:'lightGBM'}, inplace=True)
df_pred_gbm.to_json('experimental_data/pred_gbm.json')

# visualization
df_pred_overview = pd.concat([df_true, df_pred_arima110, df_pred_arima121, df_pred_holt, df_pred_gbm], axis=1)
graph = df_pred_overview.plot(figsize=(15,6), title="Avocado price prediction overview")

fig = graph.get_figure()
fig.savefig('../../experiments_plots/overview/price.png')


