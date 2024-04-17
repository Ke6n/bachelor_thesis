#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA

df = pd.read_csv('../../datasets/avocado_price.csv')
df['Date'] = pd.to_datetime(df['Date'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['region'] = le.fit_transform(df['region'])
df = df[df['region'] < 30]
price_organic_df = df[(df["type"]=='organic')]

price_organic_df.drop(['Unnamed: 0', 'type'], axis=1, inplace=True)
price_organic_df['week'] = price_organic_df['Date'].dt.isocalendar().week

price_organic_df = price_organic_df.set_index(['region','Date'])
price_organic_df = price_organic_df.sort_index()

y = price_organic_df.loc[:,['AveragePrice']]
X = price_organic_df.drop('AveragePrice', axis=1)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

forecaster010 = ARIMA(order=(0,1,0))
forecaster111 = ARIMA(order=(1,1,1))
forecaster010.fit(y_train)
forecaster111.fit(y_train)

import pickle
file = open('../../models/multi_arima010_price', 'wb')
pickle.dump(forecaster010, file)
file.close()
file = open('../../models/multi_arima111_price', 'wb')
pickle.dump(forecaster111, file)
file.close()