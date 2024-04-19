#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA, AutoARIMA

df = pd.read_csv('../../datasets/stock_cleaned.csv')
df = df.drop('Date', axis=1)
df = df.rename(columns={'Week': 'Date'})
df['Date'] = pd.to_datetime(df['Date'])
df = df.iloc[:, :31]
df = pd.melt(df, id_vars=['Date'], var_name='TickerSymbol', value_name='Price')
df = df.set_index(['TickerSymbol','Date'])
df['Price_1T'] = df['Price'].shift(1)
df['Price_1T'].iloc[0] = df['Price_1T'].iloc[1]
df['Price_2T'] = df['Price_1T'].shift(1)
df['Price_2T'].iloc[0] = df['Price_2T'].iloc[1]

y = df.loc[:,['Price']]
X = df.drop('Price', axis=1)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

forecaster010 = ARIMA(order=(0,1,0))
forecaster111 = ARIMA(order=(1,1,1))
forecaster = AutoARIMA(start_p=1, start_q=1, suppress_warnings=True, seasonal=False)
forecaster010.fit(y_train, X_train)
forecaster111.fit(y_train, X_train)
forecaster.fit(y_train, X_train)


import pickle
file = open('../../models/multi_arima010_stock', 'wb')
pickle.dump(forecaster010, file)
file.close()
file = open('../../models/multi_arima111_stock', 'wb')
pickle.dump(forecaster111, file)
file.close()
file = open('../../models/multi_arima_stock', 'wb')
pickle.dump(forecaster, file)
file.close()