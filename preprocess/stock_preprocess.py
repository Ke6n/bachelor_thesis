#!/usr/bin/env python3
import pandas as pd

# describe
df = pd.read_csv('../datasets/TESLA_Stock.csv')
#print(df.head())
#print(df.describe(include='all'))
#print(df.isna().sum())

# preprocess
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
#stock_df = df.truncate(before='2021')
#close_df = stock_df.loc[:,['Close']]
close_df = df.loc[:,['Close']]
graph = close_df.plot(figsize=(15,6), title="Overview of closing prices for selected stock series, 29-06-2021 to 24-03-2022", use_index=True)
graph.set_ylabel("USD")

fig = graph.get_figure()
fig.savefig('../data_images/stock.png')

#def get_dataframe() -> pd.DataFrame:
#    return stock_df

df.to_csv('../processed_data/stock.csv')