#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Wine functions for analysis

@author: joshmagee
Tue Sep 18 14:40:55 2018
"""

#setup
import re
import pandas as pd

def read_wine():
    path = '/Users/joshuamagee/Projects/Python/Jobs/Insight/'
    df = pd.read_csv(path + 'winemag-data-130k-v2.csv')
    
    #now we need to clean everything
    df.drop(columns=['Unnamed: 0', 'taster_twitter_handle'], inplace=True)
    
    return df

def clean_data(df):
   df = df[df['country'] == 'US'].copy()
   df.drop(columns=['country'], inplace=True)

   #drop entries with blank information, useless columns, blends, expensive
   #and non-California wines
   df.dropna(subset=['variety', 'province', 'price', 'region_1', 'region_2'], \
             inplace=True)
   df.drop(columns=['taster_name'])
   df = df[  (df['variety'] != 'Bordeaux-style Red Blend') \
       & (df['variety'] != 'Red Blend' ) \
       ]
   df = df[df['province'] == 'California'].copy()
   df.drop(columns=['province'], inplace=True)
   df = df[df['price'] < 150] #lets stick to regular wines       
   
   return df

def select_top10(df):
    #masks for various varietals
    mask_cab = df['variety'] == 'Cabernet Sauvignon'
    mask_pin = df['variety'] == 'Pinot Noir'
    mask_syr = df['variety'] == 'Syrah'
    mask_zin = df['variety'] == 'Zinfandel'
    mask_mer = df['variety'] == 'Merlot'
    
    mask_chd = df['variety'] == 'Chardonnay'
    mask_sav = df['variety'] == 'Sauvignon Blanc'
    mask_rsl = df['variety'] == 'Riesling'
    
    df = df[(mask_cab | mask_pin | mask_syr | mask_zin | mask_mer | \
             mask_chd | mask_sav | mask_rsl)
       ]

    return df


def coloring(grape):
    if grape in['Sauvignon Blanc', 'Riesling', 'Chardonnay']:
        return 'white'
    else:
        return 'red'

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
    
    
    




