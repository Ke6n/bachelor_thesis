#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
import pickle
from sktime.forecasting.base import ForecastingHorizon

y = pd.read_csv('../../processed_data/multi_price_y.csv')
X = pd.read_csv('../../processed_data/multi_price_X.csv')
y['Date'] = pd.to_datetime(y['Date'])
y = y.set_index(['region','Date'])
X['Date'] = pd.to_datetime(X['Date'])
X = X.set_index(['region','Date'])

y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
idx_label = y_test.index.get_level_values(0)[0]
fh = ForecastingHorizon(y_test.loc[idx_label].index, is_relative=False)

# loading models
file = open('../../models/multi_naive1_price', 'rb')
mod_naive = pickle.load(file)
file.close()

file = open('../../models/multi_arima010_price', 'rb')
mod_arima010 = pickle.load(file)
file.close()

file = open('../../models/multi_arima111_price', 'rb')
mod_arima111 = pickle.load(file)
file.close()

file = open('../../models/multi_arima111_price', 'rb')
mod_arima = pickle.load(file)
file.close()

file = open('../../models/multi_holt_price', 'rb')
mod_holt = pickle.load(file)
file.close()

import lightgbm as lgb
mod_gbm = lgb.Booster(model_file='../../models/multi_gbm_price.txt')

# forecasting
pred_naive = mod_naive.predict(fh, X_test)
pred_arima010 = mod_arima010.predict(fh, X_test)
pred_arima111 = mod_arima111.predict(fh, X_test)
pred_arima = mod_arima.predict(fh, X_test)
pred_holt = mod_holt.predict(fh, X_test)
pred_gbm = mod_gbm.predict(X_test)
pred_gbm = pd.DataFrame(pred_gbm, index=y_test.index, columns=['AveragePrice'])

# persistenz
y_train.to_csv('../exp_data/price/multi_in_sample.csv')
y_test.to_csv('../exp_data/price/multi_true.csv')
pred_naive.to_csv('../exp_data/price/multi_pred_naive.csv')
pred_arima010.to_csv('../exp_data/price/multi_pred_arima010.csv')
pred_arima111.to_csv('../exp_data/price/multi_pred_arima111.csv')
pred_arima.to_csv('../exp_data/price/multi_pred_arima.csv')
pred_holt.to_csv('../exp_data/price/multi_pred_holt.csv')
pred_gbm.to_csv('../exp_data/price/multi_pred_gbm.csv')