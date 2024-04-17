#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA

df = pd.read_csv('../../datasets/stock_cleaned.csv')
df = df.drop('Date', axis=1)
df = df.rename(columns={'Week': 'Date'})
df['Date'] = pd.to_datetime(df['Date'])
df = pd.melt(df, id_vars=['Date'], var_name='TickerSymbol', value_name='Price')
df = df.set_index(['TickerSymbol','Date'])
df = df.iloc[:, :30]
df['Price_1T'] = df['Price'].shift(1)
df['Price_1T'].iloc[0] = df['Price_1T'].iloc[1]
df['Price_2T'] = df['Price_1T'].shift(1)
df['Price_2T'].iloc[0] = df['Price_2T'].iloc[1]

y = df.loc[:,['Price']]
X = df.drop('Price', axis=1)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

forecaster010 = ARIMA(order=(0,1,0))
forecaster111 = ARIMA(order=(1,1,1))
forecaster010.fit(y_train)
forecaster111.fit(y_train)

import pickle
file = open('../../models/multi_arima010_stock', 'wb')
pickle.dump(forecaster010, file)
file.close()
file = open('../../models/multi_arima111_stock', 'wb')
pickle.dump(forecaster111, file)
file.close()