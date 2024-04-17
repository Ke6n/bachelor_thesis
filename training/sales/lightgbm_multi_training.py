#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
import lightgbm as lgb

df = pd.read_csv('../../datasets/walmart_sales.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df.set_index(['Store','Date'])
df = df.sort_index()

y = df.loc[:,['Weekly_Sales']]
X = df.drop('Weekly_Sales', axis=1)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

params = {
    'num_leaves':40,
    'learning_rate':0.1,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'max_depth':5
}
gbm = lgb.LGBMRegressor(**params)
gbm.fit(X_train,y_train)
gbm.booster_.save_model('../../models/multi_gbm_sales.txt')