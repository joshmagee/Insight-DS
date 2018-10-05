#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Background wine plots

@author: joshmagee
Mon Sep 17 09:34:15 2018
"""

import wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("dark")

df = pd.read_csv('/Users/joshuamagee/Projects/Python/Jobs/Insight/wine_institute_retail_list.txt', \
                 delim_whitespace=True)

#total California wine production has increased 60% over the last 15 years
fig, ax = plt.subplots()
#ax.plot(df['Year'], df['Table'], label='Table')
#ax.plot(df['Year'], df['Dessert'], label='Dessert')
#ax.plot(df['Year'], df['Sparkling'],  label='Sparkling')
ax.plot(df['Year'], df['Total'],  label='Total', color='red')
ax.set_title('California Wine Production')
ax.set_ylabel('Cases (Millions)')
ax.set_xlabel('Year')
ax.legend(loc='best')
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig('cases_by_year.png')
plt.show()

#retail value of produced wine has doubled in that same time
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Retail'])
ax.set_title('California Wine Value')
ax.set_ylabel('Retail Sale Price ($, billions)')
ax.set_xlabel('Year')
ax.legend(loc='best')
fig.tight_layout()
ax.grid(alpha=0.3)
plt.savefig('retail_by_year.png')
plt.show()

#bar chart of napa production
napa = [4, 35]
other = [96, 65]
width = 0.35
ind = np.arange(2)
fig, ax = plt.subplots()
p1 = plt.bar(ind, napa, width, color='orange')
p2 = plt.bar(ind, other, width, bottom=napa, color='blue')
ax.set_ylabel('Percent')
ax.set_xlabel('')
plt.xticks(ind, ('Produced', 'Revenue'), rotation=45)
plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0]), ('Napa', 'Everything else'), loc='upper center')
plt.tight_layout()
plt.savefig('total_area_revenue.png')
plt.show()


df = wine.read_wine()
df = wine.clean_data(df)
df = wine.select_top10(df)

df['region'] = df['region_1'].apply(wine.convert_region1)
df.drop(columns=['taster_name', 'region_1', 'region_2', \
                 'designation', 'winery', 'title'], inplace=True)

df['price_int'] = df['price'].apply(lambda x: int(5 * round(float(x)/5)))
df['model'] = df['points'].apply(lambda x: x*x)

#masks
mask_cab = df['variety'] == 'cabernet sauvignon'
mask_pin = df['variety'] == 'pinot noir'
mask_syr = df['variety'] == 'syrah'
mask_zin = df['variety'] == 'zinfandel'
mask_mer = df['variety'] == 'merlot'

mask_chd = df['variety'] == 'chardonnay'
mask_sav = df['variety'] == 'sauvignon blanc'
mask_rsl = df['variety'] == 'riesling'
mask_red = mask_cab | mask_pin | mask_syr | mask_zin | mask_mer
mask_wht = mask_chd | mask_sav | mask_rsl

#now lets looks at some simple linear regression models
sns.lmplot(data=df[mask_wht], x='price_int', y='points', \
           hue = 'variety', \
           x_estimator=np.mean,
           legend = False,
           fit_reg = False,
           logx = False
           )
ax = plt.gca()
ax.axhline(y=92.5, linestyle='dashed', alpha=0.5)
ax.legend(loc='upper left')
ax.set_xlabel('Price ($)')
ax.set_ylabel('Score')
ax.set_ylim(ymax=100)
plt.tight_layout()
plt.savefig('linear_regression_score_v_price_whites.png')
plt.show()

#square score
#now lets looks at some simple linear regression models
sns.lmplot(data=df[mask_wht], x='price_int', y='model', \
           hue = 'variety', \
           x_estimator=np.mean,
           legend = False,
           fit_reg = True,
           logx = False
           )
ax = plt.gca()
ax.legend(loc='upper left')
ax.set_xlabel('Price ($)')
ax.set_ylabel('Score squared')
plt.tight_layout()
plt.savefig('linear_regression_price_v_score_squared_whites.png')
plt.show()


sns.residplot(data=df[mask_chd], x='price_int', y='model', label='chardonnay')
ax = plt.gca()
ax.legend(loc='upper left')
ax.set_xlabel('Price ($)')
ax.set_ylabel('Score squared')
plt.tight_layout()
plt.savefig('residual_price_v_score_squared_whites.png')
plt.show()

#now lets looks at some simple linear regression models
sns.lmplot(data=df[mask_red], x='price_int', y='points', \
           hue = 'variety', \
           x_estimator=np.mean,
           legend = False,
           fit_reg = False,
           logx = False
           )
ax = plt.gca()
ax.axhline(y=91.5, linestyle='dashed', alpha=0.5)
ax.legend(loc='upper left')
ax.set_xlabel('Price ($)')
ax.set_ylabel('Score')
ax.set_ylim(ymax=100)
plt.tight_layout()
plt.savefig('linear_regression_score_v_price_reds.png')
plt.show()


#fig, ax = plt.subplots()
#sns.kdeplot(df[mask_mer]['points'], label = 'Merlot', shade=True)
#ax.legend(loc='best')
#ax.set_xlim(xmin=80, xmax=100)
#ax.set_ylabel('Frequency')
#ax.set_xlabel('Score')
#plt.tight_layout()
#plt.savefig('price_kde_merlot.png')
#plt.show()
#
#fig, ax = plt.subplots()
#sns.distplot(df[mask_mer]['points'], label = 'Merlot', bins=19)
#ax.legend(loc='best')
#ax.set_xlim(xmin=80, xmax=100)
#ax.set_ylabel('Frequency')
#ax.set_xlabel('Score')
#plt.tight_layout()
#plt.savefig('score_hist_merlot.png')
#plt.show()
#
#fig, ax = plt.subplots()
#sns.distplot(df[df['region'] == 'generic']['points'], label = 'Generic', bins=20)
#ax.legend(loc='best')
#ax.set_xlim(xmin=80, xmax=100)
#ax.set_ylabel('Frequency')
#ax.set_xlabel('Score')
#plt.tight_layout()
#plt.savefig('score_hist_generic.png')
#plt.show()



