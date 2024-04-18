#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster

df = pd.read_csv('../../datasets/walmart_sales.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df[df['Store'] <= 30]
df = df.set_index(['Store','Date'])
df = df.sort_index()

y = df.loc[:,['Weekly_Sales']]
X = df.drop('Weekly_Sales', axis=1)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train, X_train)

import pickle
file = open('../../models/multi_naive1_sales', 'wb')
pickle.dump(forecaster, file)
file.close()