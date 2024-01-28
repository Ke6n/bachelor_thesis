#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.split.base._common import SPLIT_TYPE

def split_price(df: pd.DataFrame) -> SPLIT_TYPE:  
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df = df.set_index('date')
    X = df.drop(['e5','e10','diesel'], axis=1)
    y = df['e5']
    return temporal_train_test_split(X, y)