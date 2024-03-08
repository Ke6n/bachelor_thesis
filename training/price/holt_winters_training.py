#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.ets import AutoETS


df = pd.read_csv('../../processed_data/price.csv')
y_train, y_test, X_train, X_test = split.split(df, 'AveragePrice')
forecaster = AutoETS(trend='add', seasonal='mul', sp=7)

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
