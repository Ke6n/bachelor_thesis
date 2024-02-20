#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


df = pd.read_csv('../../processed_data/stock.csv')
y_train, y_test, X_train, X_test = split.split(df, 'Close')
forecaster = ExponentialSmoothing()

forecaster.fit(y_train, X_train)

import pickle
file = open('../../models/holt_stock', 'wb')
pickle.dump(forecaster, file)
file.close()
