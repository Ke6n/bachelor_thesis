#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
import lightgbm as lgb

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

params = {
    'num_leaves':40,
    'learning_rate':0.1,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'max_depth':5
}
gbm = lgb.LGBMRegressor(**params)
gbm.fit(X_train,y_train)
gbm.booster_.save_model('../../models/multi_gbm_stock.txt')