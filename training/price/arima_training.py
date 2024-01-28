#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.arima import ARIMA


df = pd.read_csv('../../processed_data/price.csv')
X_train, X_test, y_train, y_test = split.split_price(df)
forecaster110 = ARIMA(order=(1,1,0))
forecaster121 = ARIMA(order=(1,2,1))
forecaster110.fit(y_train, X_train)
forecaster121.fit(y_train, X_train)

#forecaster.summary()

#from sktime.forecasting.base import ForecastingHorizon
# fh = ForecastingHorizon(y_test.index, is_relative=False)
# pred = forecaster.predict(fh, X_test)

import pickle
file = open('../../models/arima110_price', 'wb')
pickle.dump(forecaster110, file)
file.close()
file = open('../../models/arima121_price', 'wb')
pickle.dump(forecaster121, file)
file.close()

# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pred)**0.5)

# load_f= open('../../models/arima_price', 'rb')
# mod = pickle.load(load_f)
# pr = mod.predict(fh, X_test)
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, pr)**0.5)
