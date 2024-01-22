#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
import lightgbm as lgb

df = pd.read_csv('../../processed_data/price.csv')
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df = df.set_index('date')
X = df.drop(['e5','e10','diesel'], axis=1)
y = df['e5']

X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
regressor = lgb.LGBMRegressor()

params = {
    'num_leaves':40,
    'learning_rate':0.1,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'max_depth':5
}

gbm = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=lgb_test)

gbm.save_model('../../models/gbm_price.txt')

# from sklearn.metrics import mean_squared_error
# pred = gbm.predict(X_test)
# print(mean_squared_error(y_test, pred)**0.5)
