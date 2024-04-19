#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
import lightgbm as lgb

df = pd.read_csv('../../datasets/avocado_price.csv')
df['Date'] = pd.to_datetime(df['Date'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['region'] = le.fit_transform(df['region'])
df = df[df['region'] < 30]
price_organic_df = df[(df["type"]=='organic')]

price_organic_df.drop(['Unnamed: 0', 'type'], axis=1, inplace=True)

price_organic_df = price_organic_df.set_index(['region','Date'])
price_organic_df = price_organic_df.sort_index()

y = price_organic_df.loc[:,['AveragePrice']]
X = price_organic_df.drop('AveragePrice', axis=1)
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
gbm.booster_.save_model('../../models/multi_gbm_price.txt')