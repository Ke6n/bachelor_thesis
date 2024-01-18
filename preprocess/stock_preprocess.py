#!/usr/bin/env python3
import pandas as pd

# describe
df = pd.read_csv('../datasets/TESLA_Stock.csv')
#print(df.head())
#print(df.describe(include='all'))
#print(df.isna().sum())

# preprocess
df = df.set_index('Date')
stock_df = df.truncate(before='2021')
close_df = stock_df.loc[:,['Close']]
graph = close_df.plot(figsize=(15,6), title="TESLA stock daily closing price, 04-01-2021 to 24-03-2022", use_index=True)
graph.set_ylabel("USD")

fig = graph.get_figure()
fig.savefig('../images/stock.png')

#def get_dataframe() -> pd.DataFrame:
#    return stock_df

stock_df.to_csv('../experimental_data/stock.csv')