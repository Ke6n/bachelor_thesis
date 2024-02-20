#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.arima import ARIMA


df = pd.read_csv('../../processed_data/stock.csv')
y_train, y_test, X_train, X_test = split.split(df, 'Close')

forecaster110 = ARIMA(order=(1,1,0))
forecaster121 = ARIMA(order=(1,2,1))
forecaster110.fit(y_train, X_train)
forecaster121.fit(y_train, X_train)

import pickle
file = open('../../models/arima110_stock', 'wb')
pickle.dump(forecaster110, file)
file.close()
file = open('../../models/arima121_stock', 'wb')
pickle.dump(forecaster121, file)
file.close()

