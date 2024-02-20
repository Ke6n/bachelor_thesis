#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.naive import NaiveForecaster


df = pd.read_csv('../../processed_data/sales.csv')
y_train, y_test, X_train, X_test = split.split(df,'Weekly_Sales')

forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train, X_train)

import pickle
file = open('../../models/naive1_sales', 'wb')
pickle.dump(forecaster, file)
file.close()