#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.append("../..")
import utils.split as split
import lightgbm as lgb

df = pd.read_csv('../../processed_data/stock.csv')
y_train, y_test, X_train, X_test = split.split(df,'Close')

params = {
    'num_leaves':40,
    'learning_rate':0.1,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'max_depth':5
}

gbm_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.05,**params)
gbm_upper.fit(X_train,y_train)
gbm_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.95,**params)
gbm_lower.fit(X_train,y_train)
gbm = lgb.LGBMRegressor(**params)
gbm.fit(X_train,y_train)
gbm_upper.booster_.save_model('../../models/gbm_upper_stock.txt')
gbm_lower.booster_.save_model('../../models/gbm_lower_stock.txt')
gbm.booster_.save_model('../../models/gbm_stock.txt')
