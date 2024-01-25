#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import AutoARIMA


df = pd.read_csv('../../processed_data/price.csv')
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df = df.set_index('date')
X = df.drop(['e5','e10','diesel'], axis=1)
y = df['e5']

X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
forecaster = AutoARIMA(start_P=1, start_q=0)

forecaster.fit(y_train, X_train)

#from sktime.forecasting.base import ForecastingHorizon
# fh = ForecastingHorizon(y_test.index, is_relative=False)
# pred = forecaster.predict(fh, X_test)

import pickle
file = open('../../models/arima_price', 'wb')
pickle.dump(forecaster, file)
file.close()

# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pred)**0.5)

# load_f= open('../../models/arima_price', 'rb')
# mod = pickle.load(load_f)
# pr = mod.predict(fh, X_test)
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pr)**0.5)
