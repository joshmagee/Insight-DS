#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process data into sqlite db.

@author: joshmagee
Thu Sep 27 08:40:47 2018
"""

import wine
import sqlite3
import pandas as db
import dill


df = wine.read_wine()
df = wine.clean_data(df)
df = wine.select_top10(df)
df['region'] = df['region_1'].apply(wine.convert_region1)
df.drop(columns=['taster_name', 'region_1', 'region_2', \
                 'designation', 'winery', 'title'], inplace=True)

conn = sqlite3.connect('wine_data_clean.db')

df.to_sql("wine", conn, if_exists="replace")

conn.close

with open('wine_dataframe.dill', 'wb') as file:
    dill.dump(df, file)

    
