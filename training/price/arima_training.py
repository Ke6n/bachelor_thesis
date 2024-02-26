#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.arima import ARIMA


df = pd.read_csv('../../processed_data/price.csv')
y_train, y_test, X_train, X_test = split.split(df, 'AveragePrice')

forecaster010 = ARIMA(order=(0,1,0))
forecaster111 = ARIMA(order=(1,1,1))
forecaster010.fit(y_train, X_train)
forecaster111.fit(y_train, X_train)

#forecaster.summary()

#from sktime.forecasting.base import ForecastingHorizon
# fh = ForecastingHorizon(y_test.index, is_relative=False)
# pred = forecaster.predict(fh, X_test)

import pickle
file = open('../../models/arima010_price', 'wb')
pickle.dump(forecaster010, file)
file.close()
file = open('../../models/arima111_price', 'wb')
pickle.dump(forecaster111, file)
file.close()

# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pred)**0.5)

# load_f= open('../../models/arima_price', 'rb')
# mod = pickle.load(load_f)
# pr = mod.predict(fh, X_test)
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pr)**0.5)
