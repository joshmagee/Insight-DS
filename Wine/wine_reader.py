#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial data exploration of wine

@author: joshmagee
Wed Sep 12 10:33:50 2018
"""

#setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("dark")

#now some specific plotting functions
def plot_empty_fields(df, title, ylabel, scale, file):
    index = [i for i,_ in enumerate(df.columns)]
    
    fig, ax = plt.subplots()
    ax.bar(index, df.isna().sum()/len(df.index)*100)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([i for i,_ in enumerate(df.columns)])
    ax.set_xticklabels([x for x in df.columns],rotation = 45, ha="right")
    ax.set_yscale(scale)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(file)
    plt.show()
    
    return


#read input stream
df = pd.read_csv('/Users/joshuamagee/Downloads/winemag-data-130k-v2.csv')

#start cleaning data set

'''
We can immediately drop columns: unnamed, taster_twitter_handle
Cleaning questions:
    1) How many NaN's are there? Can we just drop them?
    2) Is there a way to visualize where the NaNs live in the data?
'''
#drop shit we don't care about
df.drop(columns=['Unnamed: 0', 'taster_twitter_handle'], inplace=True)

#answer to question #1, above
print('Total NaNs: ', df.isna().sum())
print('Better estimate: ', df.isnull().values.ravel().sum())

title = 'Number of Empty Fields'
ylabel = 'Percent of total entries (%)'

#plot for full world data
#plot_empty_fields(df, title, ylabel, 'log', 'full_nan.png')
#plot_empty_fields(df, title, ylabel, 'linear', 'full_nan_zoomed.png')

#now plot for US only, notice we have a much cleaner data set
df = df[df['country'] == 'US']
df.drop(columns=['country'], inplace=True)
title = 'Number of Empty Fields (US only)'
plot_empty_fields(df, title, ylabel, 'log', 'usa_nan.png')
plot_empty_fields(df, title, ylabel, 'linear', 'usa_nan_zoomed.png')

#Anything without a varietal we MUST drop, and we can lose anything without
#a country. Let's also drop anything without a price.
df.dropna(subset=['variety', 'province', 'price'], inplace=True)

#we don't care so much about the taster name, so we can probably drop that
#column in total at the moment
df.drop(columns=['taster_name'])

#30% of designations that are blank
#designations are things like:
#   * Reserve, small batch status, proprietary
#   * Specific vineyard
#   * OldVine status
#   * Anything special, 'Barrel Fermented'?
#Let's ignore designation for now, knowing later we should further categorize
#later

'''
Initial notes on the data

Roughly 130k winerys, 7% are missing price information.
    1) Should confirm that we can discard this 7%. Need to determine if its
        independent of US, or how they are distributed?
'''

#Initial data exploration
df.count().sort_values(ascending=False)
df.describe()
df.describe(percentiles=[x/10. for x in range(0,10)])


'''
Let's start visualizing some data:
    0) Histogram of reviewed wines by type? By country?
    1) What's the distribution of price/points by varietal?
    2) Plot the same for various countries
    3) How many regions are in each country?
'''

#let's look at varietals. Which varietals do we have the most data for?
#Answer to question 0
grape = df.groupby(['variety'])['variety'].count().sort_values(ascending=True)

#plot CDF
fig, ax = plt.subplots()
ax.plot(np.arange(len(grape)), np.cumsum(grape)/np.sum(grape), 'r')
ax.set_title('Cumulative Distribution Function of Grape Varietal')
ax.set_ylabel('Cumulative sum (%)')
ax.set_xlabel('Varietal ordered by frequency')
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig('varietal_cdf.png')
plt.show()

#the 96th-percentile covers roughly the last 10 single-varietal wines
#(a generic 'Bordeaux-style red blend' is the only blend this frequently reviewed
# and is discarded)
df = df[  (df['variety'] != 'Bordeaux-style Red Blend') \
        & (df['variety'] != 'Red Blend' ) \
        ]
grape = df.groupby(['variety'])['variety'].count().sort_values(ascending=True)

#top varietals are:
#varietal               number of reviews
#Rose                   900
#Cabernet Franc         999
#Riesling              1745
#Sauvignon Blanc       2154
#Merlot                2300
#Zinfandel             2705
#Syrah                 3232
#Chardonnay            6773
#Cabernet Sauvignon    7280
#Pinot Noir            9857

#instead of region, province gives the state
#looking just a cabernet sauvignon, we see that there are 135 designated regions
#many of which are outside of california. We will have to go through

state  = df[df['variety'] == 'Cabernet Sauvignon'].groupby(['province'])['variety'].count()
state/state.sum()

index = [i for i,_ in enumerate(state)]

fig, ax = plt.subplots()
ax.barh(index, state/state.sum())
#ax.set_title(title)
#ax.set_ylabel(ylabel)
ax.set_yticks([i for i,_ in enumerate(state)])
ax.set_yticklabels([x for x in state.index])
#ax.set_yscale(scale)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig('hist_province.png')
plt.show()


#let's just look at California wines first
mask_cal = df['province'] == 'California'
#masks for various varietals
mask_cab = df['variety'] == 'Cabernet Sauvignon'
mask_pin = df['variety'] == 'Pinot Noir'
mask_syr = df['variety'] == 'Syrah'
mask_zin = df['variety'] == 'Zinfandel'
mask_mer = df['variety'] == 'Merlot'

mask_chd = df['variety'] == 'Chardonnay'
mask_sav = df['variety'] == 'Sauvignon Blanc'
mask_rsl = df['variety'] == 'Riesling'

reds   = ['Cabernet Sauvignon', 'Pinot Noir', 'Syrah', 'Zinfandel', 'Merlot']
whites = ['Chardonnay', 'Sauvignon Blanc', 'Riesling']
#start of our visualization. Let's start by looking at the overall distributions
#box plots
fig, ax = plt.subplots()
sns.boxplot(data=df[mask_cal], x='variety', y='price', \
               order=reds)
plt.savefig('price_box_reds.png')
plt.show()

fig, ax = plt.subplots()
sns.boxplot(data=df[mask_cal], x='variety', y='price', \
               order=whites)
plt.savefig('price_box_whites.png')
plt.show()

fig, ax = plt.subplots()
sns.boxplot(data=df[mask_cal], x='variety', y='points', \
               order=reds)
plt.savefig('score_box_reds.png')
plt.show()

fig, ax = plt.subplots()
sns.boxplot(data=df[mask_cal], x='variety', y='points', \
               order=whites)
plt.savefig('score_box_whites.png')
plt.show()

df = df[df['price'] < 150]
mask_cal = df['province'] == 'California'
mask_cab = df['variety'] == 'Cabernet Sauvignon'
mask_pin = df['variety'] == 'Pinot Noir'
mask_syr = df['variety'] == 'Syrah'
mask_zin = df['variety'] == 'Zinfandel'
mask_mer = df['variety'] == 'Merlot'

mask_chd = df['variety'] == 'Chardonnay'
mask_sav = df['variety'] == 'Sauvignon Blanc'
mask_rsl = df['variety'] == 'Riesling'

#now violins for scores
fig, ax = plt.subplots()
sns.violinplot(data=df[mask_cal], x='variety', y='points', \
               order=reds, fliersize=20)
plt.savefig('points_violin_reds.png')
plt.show()

fig, ax = plt.subplots()
sns.violinplot(data=df[mask_cal], x='variety', y='points', \
               order=whites, fliersize=20)
plt.savefig('points_violin_whites.png')
plt.show()

fig, ax = plt.subplots()
sns.violinplot(data=df[mask_cal], x='variety', y='price', \
               order=reds, fliersize=20)
plt.savefig('price_violin_reds.png')
plt.show()

fig, ax = plt.subplots()
sns.violinplot(data=df[mask_cal], x='variety', y='price', \
               order=whites, fliersize=20)
plt.savefig('price_violin_whites.png')
plt.show()

fig, ax = plt.subplots()
sns.kdeplot(df[mask_cal & mask_cab]['price'], label = 'Cabernet Sauvignon')
sns.kdeplot(df[mask_cal & mask_pin]['price'], label = 'Pinot Noir')
sns.kdeplot(df[mask_cal & mask_syr]['price'], label = 'Syrah')
sns.kdeplot(df[mask_cal & mask_zin]['price'], label = 'Zinfandel')
sns.kdeplot(df[mask_cal & mask_mer]['price'], label = 'Merlot')
ax.legend(loc='best')
ax.set_xlim(xmin=0, xmax=100)
ax.set_ylabel('Frequency')
ax.set_xlabel('US Price ($)')
plt.savefig('price_kde_reds.png')
plt.show()

fig, ax = plt.subplots()
sns.kdeplot(df[mask_cal & mask_chd]['price'], label = 'Chardonnay')
sns.kdeplot(df[mask_cal & mask_sav]['price'], label = 'Sauvignon Blanc')
sns.kdeplot(df[mask_cal & mask_rsl]['price'], label = 'Riesling')
ax.legend(loc='best')
ax.set_xlim(xmin=0, xmax=100)
ax.set_ylabel('Frequency')
ax.set_xlabel('US Price ($)')
plt.savefig('price_kde_whites.png')
plt.show()



fig, ax = plt.subplots()
sns.kdeplot(df[mask_cal & mask_cab]['points'], label = 'Cabernet \nSauvignon')
sns.kdeplot(df[mask_cal & mask_pin]['points'], label = 'Pinot Noir')
sns.kdeplot(df[mask_cal & mask_syr]['points'], label = 'Syrah')
sns.kdeplot(df[mask_cal & mask_zin]['points'], label = 'Zinfandel')
sns.kdeplot(df[mask_cal & mask_mer]['points'], label = 'Merlot')
ax.legend(loc='best')
ax.set_xlim(xmin=80, xmax=100)
ax.set_ylabel('Frequency')
ax.set_xlabel('Score')
plt.savefig('score_kde_reds.png')
plt.show()

fig, ax = plt.subplots()
sns.kdeplot(df[mask_cal & mask_chd]['points'], label = 'Chardonnay')
sns.kdeplot(df[mask_cal & mask_sav]['points'], label = 'Sauvignon \nBlanc')
sns.kdeplot(df[mask_cal & mask_rsl]['points'], label = 'Riesling')
ax.legend(loc='best')
ax.set_xlim(xmin=80, xmax=100)
ax.set_ylabel('Frequency')
ax.set_xlabel('Score')
plt.savefig('score_kde_whites.png')
plt.show()

mask_red = mask_cab | mask_pin | mask_syr | mask_zin | mask_mer
mask_wht = mask_chd | mask_sav | mask_rsl

#now lets looks at some simple linear regression models
sns.lmplot(data=df[mask_cal & mask_red], x='points', y='price', \
           hue = 'variety', \
           x_estimator=np.mean,
           legend = False,
           fit_reg = True,
           logx = False
           )
ax = plt.gca()
ax.legend(loc='upper left')
ax.set_ylabel('Price ($)')
ax.set_xlabel('Score')
plt.show()

sns.residplot(data=df[mask_cal & mask_cab], x='points', y='price')
sns.residplot(data=df[mask_cal & mask_mer], x='points', y='price')
sns.residplot(data=df[mask_cal & mask_pin], x='points', y='price')
sns.residplot(data=df[mask_cal & mask_zin], x='points', y='price')
sns.residplot(data=df[mask_cal & mask_syr], x='points', y='price')
ax = plt.gca()
ax.legend(loc='upper left')
ax.set_ylabel('Price Residual')
ax.set_xlabel('Score')
plt.show()


sns.lmplot(data=df[mask_cal & mask_wht], x='points', y='price', \
           hue = 'variety', \
           x_estimator=np.mean,
           legend = False,
           fit_reg = True,
           logx=False
           )
ax = plt.gca()
ax.legend(loc='upper left')
ax.set_ylabel('Price ($)')
ax.set_xlabel('Score')
plt.show()

sns.residplot(data=df[mask_cal & mask_chd], x='points', y='price')
sns.residplot(data=df[mask_cal & mask_sav], x='points', y='price')
sns.residplot(data=df[mask_cal & mask_rsl], x='points', y='price')
ax = plt.gca()
ax.legend(loc='upper left')
ax.set_ylabel('Price Residual')
ax.set_xlabel('Score')
plt.show()




# =============================================================================
# #lets look only at Pinot Noir varietal
# dpinot = df[df['variety'] == 'Pinot Noir']
# dpinot.count().sort_values(ascending=False)
# dpinot.describe()
# dpinot.describe(percentiles=[x/10. for x in range(0,10)])
# '''
# Only two numerical features: rating points and price
# There are 13272 unique wines, but only 9036 (68%) with all attributes.
# 
# Average points: 89.4 +- 3.12 (min=80, max=99)
# Median points = 90.0
# 
# Average price: 47.5 +- 47.609 (min = 5, max = 2500)
# Median price = 42, 95% percentile price = 90.0
# Average discounting >95%: 41.78 +- 18.0
# 
# '''
# 
# dpinot.describe(include='O') #ordinal features
# '''
# There are 256 unique main regions, and 17 sub-regions. Region 2 is only
# available for 68% of all wines.
# 
# To do:
#     1) Is region 1 too large/small to be useful? Can we condense into one
#         region label?
#     2) Designation label seems mostly useless. Does ID some reserve bottles,
#         but most are labelling the vineyard.
#     3) Designation may be sueful to identify specific vineyard
#     4) Province includes lots of other places. Would be nice to plot frequency
#         of places.
#     5) Do longer (ie, more pretenious?) titles correlate with higher prices or
#         lower quality wines?
#     6) taster_name might correlate with various scores or descriptors. Possibly
#         need to regress that out or fix that.
# '''
# 
# 
# 
# x = dpinot['price']
# x = x.dropna()
# fig = plt.figure
# ax = plt.gca()
# n, bins, patches = ax.hist(x=x,bins=np.logspace(0,3,50))
# plt.show()
# 
# =============================================================================


