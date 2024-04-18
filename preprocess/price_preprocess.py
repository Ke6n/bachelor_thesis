#!/usr/bin/env python3
import pandas as pd
import datetime

# describe
df = pd.read_csv('../datasets/avocado_price.csv')
#print(df.head())
#df.dtypes
#print(df.describe(include='all'))
#print(df.isna().sum())

# preprocess
df['Date'] = pd.to_datetime(df['Date'])
price_Albany_df = df[(df["region"]=='Albany') & (df["type"]=='organic')]
price_Albany_df.drop(['Unnamed: 0', 'type', 'region'], axis=1, inplace=True)
price_Albany_df['week'] = price_Albany_df['Date'].dt.isocalendar().week
price_Albany_df = price_Albany_df.set_index('Date')
price_Albany_df = price_Albany_df.sort_index()

price_df = price_Albany_df.loc[:,['AveragePrice']]
graph = price_df.plot(figsize=(15,6), title="Overview of the selected price series, Jan. 2015 to Mar. 2018")
graph.set_ylabel("USD")

fig = graph.get_figure()
fig.savefig('../data_images/price.png')

price_Albany_df.to_csv('../processed_data/price.csv')

# for multi series
df = pd.read_csv('../datasets/avocado_price.csv')
df['Date'] = pd.to_datetime(df['Date'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['region'] = le.fit_transform(df['region'])
df = df[df['region'] < 30]
price_organic_df = df[(df["type"]=='organic')]

price_organic_df.drop(['Unnamed: 0', 'type'], axis=1, inplace=True)
price_organic_df['week'] = price_organic_df['Date'].dt.isocalendar().week

price_organic_df = price_organic_df.set_index(['region','Date'])
price_organic_df = price_organic_df.sort_index()

y = price_organic_df.loc[:,['AveragePrice']]
X = price_organic_df.drop('AveragePrice', axis=1)
y.to_csv('../processed_data/multi_price_y.csv')
X.to_csv('../processed_data/multi_price_X.csv')