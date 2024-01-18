#!/usr/bin/env python3
import pandas as pd
import datetime

UUID = 'c8fbf014-a55f-4f69-ab76-4b96a6612fd6'
date_begin = datetime.date(2023,1,1)
date_end = datetime.date(2023,12,31)
path_head = '../datasets/petrol_station_price/'


def getPathList() -> list:
    path_list = []
    for i in range((date_end-date_begin).days+1):
        date = date_begin + datetime.timedelta(days=i)
        path = path_head +  '{:02d}'.format(date.month) + '/' + str(date.strftime('%Y-%m-%d')) + '-prices.csv'        
        path_list.append(path)
    return path_list

def getDataFrame() -> pd.DataFrame:
    path_list = getPathList()
    df = pd.DataFrame()
    for i in path_list:
        df_temp = pd.read_csv(i)
        df_i = df_temp.loc[df_temp["station_uuid"]=='c8fbf014-a55f-4f69-ab76-4b96a6612fd6']
        if not df.empty:
            df = pd.concat([df, df_i])
        else:
            df = df_i
    return df

#preprocess
df = getDataFrame()
df.drop(['station_uuid', 'dieselchange', 'e5change', 'e10change'], axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.strftime('%Y-%m-%d'))
df = df.groupby('date').mean()
for i in range(len(df.columns)):
    df[df.columns[i]] = round(df[df.columns[i]], 3)

# describe
#print(df.head())
#print(df.describe(include='all'))
#print(df.isna().sum())

e5_df = df.loc[:,['e5']]
graph = e5_df.plot(figsize=(15,6), title="The daily price of Super E5 at a German petrol station, 01-01-2023 to 31-12-2023")
graph.set_ylabel("EUR")

fig = graph.get_figure()
fig.savefig('../images/price.png')

df.to_csv('../experimental_data/price.csv')