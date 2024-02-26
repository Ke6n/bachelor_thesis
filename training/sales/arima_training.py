#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.arima import ARIMA


df = pd.read_csv('../../processed_data/sales.csv')
y_train, y_test, X_train, X_test = split.split(df, 'Weekly_Sales')

forecaster010 = ARIMA(order=(0,1,0))
forecaster111 = ARIMA(order=(1,1,1))
forecaster010.fit(y_train, X_train)
forecaster111.fit(y_train, X_train)

import pickle
file = open('../../models/arima010_sales', 'wb')
pickle.dump(forecaster010, file)
file.close()
file = open('../../models/arima111_sales', 'wb')
pickle.dump(forecaster111, file)
file.close()

