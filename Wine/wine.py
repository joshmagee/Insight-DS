#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Wine functions for analysis

@author: joshmagee
Tue Sep 18 14:40:55 2018
"""

#setup
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_wine():
    path = '/Users/joshuamagee/Projects/Python/Jobs/Insight/'
    df = pd.read_csv(path + 'winemag-data-130k-v2.csv')
    
    #now we need to clean everything
    df.drop(columns=['Unnamed: 0', 'taster_twitter_handle'], inplace=True)
    
    return df

def plot_coefs(X, coef, label):
    coefs = pd.Series(coef, index = X.columns)
    coefs.plot(kind = "barh")
    plt.title(label)
    plt.show()
    return

def clean_data(df):
   df = df[df['country'] == 'US'].copy()
   df.drop(columns=['country'], inplace=True)

   #drop entries with blank information, useless columns, blends, expensive
   #and non-California wines
   df.dropna(subset=['variety', 'province', 'price', 'region_1', 'region_2'], \
             inplace=True)
   df.drop(columns=['taster_name'])
   df['variety'] = df['variety'].str.lower()

   df = df[  (df['variety'] != 'bordeaux-style red blend') \
       & (df['variety'] != 'red blend' ) \
       ]
   df = df[df['province'] == 'California'].copy()
   df.drop(columns=['province'], inplace=True)
   df = df[df['price'] < 101] #lets stick to regular wines       
   
   return df

def select_top10(df):
    #masks for various varietals
    mask_cab = df['variety'] == 'cabernet sauvignon'
    mask_pin = df['variety'] == 'pinot noir'
    mask_syr = df['variety'] == 'syrah'
    mask_zin = df['variety'] == 'zinfandel'
    mask_mer = df['variety'] == 'merlot'
    
    mask_chd = df['variety'] == 'chardonnay'
    mask_sav = df['variety'] == 'sauvignon blanc'
    mask_rsl = df['variety'] == 'riesling'
    
    df = df[(mask_cab | mask_pin | mask_syr | mask_zin | mask_mer | \
             mask_chd | mask_sav | mask_rsl)
       ]

    return df


def coloring(grape):
    if grape in['sauvignon blanc', 'riesling', 'chardonnay']:
        return 'white'
    else:
        return 'red'

def extract_year(line):
    for word in line.split():
        if '½' in word:
            word = word.replace('½','.5')
        if word.isnumeric():
            if (float(word) > 1980 and float(word) < 2020):
                return word #year is categorical, not continuous
    return 'generic'  #return generic

def convert_score(val):
    if val >= 97.0:
        return 'A+'
    elif val >= 93.0:
        return 'A'
    elif val >= 90.0:
        return 'A-'
    elif val >= 87.0:
        return 'B+'
    elif val >= 84.0:
        return 'B'
    elif val >= 80.0:
        return 'B-'
    else:
        return 'F'


def convert_region1(reg):
    d = {'north coast':
            ['lake', 'carneros', 'mendocino', 'napa', 'solano', 'sonoma', \
             'north', 'creek', 'alexander', 'rutherford', 'oakville', \
             'helena', 'green', 'howell', 'veeder', 'knoll', 'spring', \
             'diamond mountain', 'calistoga', 'knights', 'yountville', \
             'bennett', 'ross-seaview', 'chalk', 'atlas', 'rockpile', \
             'yorkville', 'marin', 'coombsville', 'contra cost', 'clarksburg', \
             'chiles valley', 'cloverdale', 'suisun'],
         'central coast':
             ['livermore', 'monterey', 'paso', 'benito', 'francisco', \
              'obispo', 'barbara', 'clara', 'cruz', 'central', 'rita',
              'lucia', 'maria', 'ynez', 'edna', 'arroyo', 'stags', 'carmel', \
              'harlan', 'chalone', 'ballard', 'adelaida', 'lomond', \
              'paicines', 'cienega', 'york', 'templeton', 'san lucas', 'pomar', \
              'bernabe', 'lime kiln', 'antonio'],
         'sierra foothills':
             ['amador', 'calaveras', 'dorado', 'nevada', 'placer', 'sierra', \
              'shenandoah', 'fiddletown', 'fair play'],
         'inland valleys':
             ['lodi', 'delta', 'madera', 'sacramento', 'joaquin', 'yolo', \
              'dunnigan', 'mokelumne', 'river junction', 'clements hills', \
              'capay'],
         'south cali':
             ['cucamonga', 'angeles', 'diego', 'temecula', 'south coast', \
              'malibu-newton', 'malibu', 'ventura'],
         'far north':
             ['russia', 'far north', 'humboldt', 'tehama', 'lime kiln'],
         'generic':
             ['california']
         }
    
    for key, value in d.items():
        for val in value:
            if re.search(val, reg.lower()):
                return key
    return reg


if __name__ == '__main__':
    df = read_wine()
    df = clean_data(df)
    df = select_top10(df)
    
    
    




