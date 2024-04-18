#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA

df = pd.read_csv('../../datasets/walmart_sales.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df[df['Store'] <= 30]
df = df.set_index(['Store','Date'])
df = df.sort_index()

y = df.loc[:,['Weekly_Sales']]
X = df.drop('Weekly_Sales', axis=1)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

forecaster010 = ARIMA(order=(0,1,0))
forecaster111 = ARIMA(order=(1,1,1))
forecaster010.fit(y_train)
forecaster111.fit(y_train)

import pickle
file = open('../../models/multi_arima010_sales', 'wb')
pickle.dump(forecaster010, file)
file.close()
file = open('../../models/multi_arima111_sales', 'wb')
pickle.dump(forecaster111, file)
file.close()