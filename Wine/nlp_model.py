#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start to build final NLP prediction model

@author: joshmagee
Tue Sep 18 15:00:42 2018
"""

import wine
import pandas as pd

df = wine.read_wine()
df = wine.clean_data(df)
df = wine.select_top10(df)


#map points to categorical grade
df['points'] = df['points'].apply(wine.convert_score)
df['region_1'] = df['region_1'].apply(wine.convert_region1)
df['color'] = df['variety'].apply(wine.coloring)
'''
    It turns out that 'general california wine' covers major players:
        Kendall-Jackson, Robert Mondavi, Josh Cellars, Cupcake, and others
    Scanning by eye, region_1  larrgely matches region_2, so region_2 can
        be dropped. Also dropping incomplete taster information.
'''
df.drop(columns=['taster_name', 'region_2'], inplace=True)
#drop designation temporarily to train
df.drop(columns=['designation'], inplace=True)

dg = df.copy()

#cut sets everything above the maximum bin as NaN
#must fix that by setting them to '100'
df['price'] = \
    pd.cut(df['price'], \
       bins = [x for x in range(0,111,10)], \
       labels=[str(x) for x in range(0,101,10)])
df['price'].fillna('100', inplace=True)



