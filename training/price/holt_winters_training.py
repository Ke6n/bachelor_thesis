#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import libs.split as split
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


df = pd.read_csv('../../processed_data/price.csv')
X_train, X_test, y_train, y_test = split.split_price(df)
forecaster = ExponentialSmoothing()

forecaster.fit(y_train, X_train)

#from sktime.forecasting.base import ForecastingHorizon
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
