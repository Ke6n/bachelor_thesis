#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import libs.split as split
from sktime.forecasting.naive import NaiveForecaster


df = pd.read_csv('../../processed_data/price.csv')
X_train, X_test, y_train, y_test = split.split_price(df)

forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train, X_train)

import pickle
file = open('../../models/naive1_price', 'wb')
pickle.dump(forecaster, file)
file.close()