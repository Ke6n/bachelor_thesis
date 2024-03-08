#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
from sktime.forecasting.ets import AutoETS


df = pd.read_csv('../../processed_data/sales.csv')
y_train, y_test, X_train, X_test = split.split(df, 'Weekly_Sales')
forecaster = AutoETS(trend='add', seasonal='mul', sp=7)

forecaster.fit(y_train, X_train)

import pickle
file = open('../../models/holt_sales', 'wb')
pickle.dump(forecaster, file)
file.close()
