#!/usr/bin/env python3
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.split.base._common import SPLIT_TYPE

def split(df: pd.DataFrame, target_name: str) -> SPLIT_TYPE:  
    df['Date']=pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    y = df[target_name]
    X = df.drop(target_name, axis=1)
    return temporal_train_test_split(y, X)