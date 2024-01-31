#!/usr/bin/env python3
import pandas as pd

# describe
df = pd.read_csv('../datasets/walmart_sales.csv')
#print(df.head())
#print(df[df["Weekly_Sales"]>0].count())
#print(df[df["Store"]==1].count())
#print(df.describe(include='all'))
#print(df.isna().sum())

# preprocess
df = df.set_index('Date')
store1_df = df[df["Store"]==1]
store1_df.drop('Store', axis=1, inplace=True)

sales_df = store1_df.loc[:,['Weekly_Sales']]
graph = sales_df.plot(figsize=(15,6), title="Walmart store_1 weekly sales, February 2010 to October 2012")
graph.set_ylabel("Million USD")

fig = graph.get_figure()
fig.savefig('../data_images/sales.png')

#def get_dataframe() -> pd.DataFrame:
#    return store1_df

store1_df.to_csv('../processed_data/sales.csv')