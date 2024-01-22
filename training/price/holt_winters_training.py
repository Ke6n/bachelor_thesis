#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.base import ForecastingHorizon

df = pd.read_csv('../../processed_data/price.csv')
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df = df.set_index('date')
X = df.drop(['e5','e10','diesel'], axis=1)
y = df['e5']

X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
forecaster = ExponentialSmoothing()

forecaster.fit(y_train, X_train)
# fh = ForecastingHorizon(y_test.index, is_relative=False)
# pred = forecaster.predict(fh, X_test)

import pickle
file = open('../../models/holt_price', 'wb')
pickle.dump(forecaster, file)
file.close()

# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pred)**0.5)

# load_f= open('../../models/holt_price', 'rb')
# mod = pickle.load(load_f)
# pr = mod.predict(fh)
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pr)**0.5)
