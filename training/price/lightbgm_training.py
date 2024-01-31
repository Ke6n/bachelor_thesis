#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
import lightgbm as lgb

df = pd.read_csv('../../processed_data/price.csv')
y_train, y_test, X_train, X_test = split.split(df,'AveragePrice')
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
