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
ax.set_title('California Wine Production', fontsize=15)
ax.set_ylabel('Cases (Millions)', fontsize=15)
ax.set_xlabel('Year', fontsize=15)
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.tick_params(labelsize='large')
fig.tight_layout()
plt.savefig('cases_by_year.png')
plt.show()

#retail value of produced wine has doubled in that same time
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Retail'])
ax.set_title('California Wine Value', fontsize=15)
ax.set_ylabel('Retail Price ($, billions)', fontsize=15)
ax.set_xlabel('Year', fontsize=15)
ax.legend(loc='best')
fig.tight_layout()
ax.grid(alpha=0.3)
ax.tick_params(labelsize='large')
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
ax.set_ylabel('Percent', fontsize=15)
ax.set_xlabel('')
plt.xticks(ind, ('Produced', 'Revenue'), rotation=45, fontsize=20)
plt.yticks(np.arange(0, 101, 10))
ax.tick_params(labelsize='large')
plt.legend((p1[0], p2[0]), ('Napa', 'Everything else'), loc='upper center',
           fontsize=10)
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

def plot_score_v_price(data, x, y, hue, color, line, ylabel):
    sns.lmplot(data=data, x=x, y=y,
           hue = hue,
           x_estimator=np.mean,
           legend = False,
           fit_reg = False,
           logx = False
           )
    ax = plt.gca()
    if y == 'model':
        ymax = 10000
    else:
        ymax = 100
    if line != None:
        ax.axhline(y=line, linestyle='dashed', alpha=0.5)
    ax.legend(loc='upper left', fontsize=15)
    ax.set_xlabel('Price ($)', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_ylim(ymax=ymax)
    ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt.savefig('linear_regression_score_v_price_' + color + '.png')
    plt.show()


def plot_score_v_price_fit(data, x, y, hue, color, line, ylabel):
    sns.lmplot(data=data, x=x, y=y,
           hue = hue,
           x_estimator=np.mean,
           legend = False,
           fit_reg = True,
           logx = False
           )
    ax = plt.gca()
    if y == 'model':
        ymax = 10000
    else:
        ymax = 100
    if line != None:
        ax.axhline(y=line, linestyle='dashed', alpha=0.5)
    ax.legend(loc='lower right', fontsize=15)
    ax.set_xlabel('Price ($)', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_ylim(ymax=ymax)
    ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt.savefig('linear_regression_fit_score_v_price_' + color + '.png')
    plt.show()

def plot_avg_residuals(df, xname, yname, label):
    dg = df.copy()
    x = [x for x in range(5,101,5)]
    
    yvals = []
    yerrs = []
    for i, val in enumerate(x):
        mask = dg[xname] == val  
        yvals.append(dg[mask][yname].mean())
        yerrs.append(dg[mask][yname].std())
        
    #seaborn only takes arrays, not lists
    x = np.array(x)
    yvals = np.array(yvals)
    yerrs = np.array(yerrs)
    
    sns.residplot(x=x, y=yvals, label=label)
    ax = plt.gca()
#    line = ax.get_lines()[0]
#    yd = line.get_ydata()
#    yd = np.array(yd)
#    ax.errorbar(x, yvals, yerr=yerrs*yd/yvals)
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel('Price ($)', fontsize=20)
    ax.set_ylabel('Score Residual', fontsize=20)
    ax.tick_params(labelsize='large')
    ax.set_xlim(xmin=0, xmax=100)
    ax.set_ylim(ymin=-3, ymax=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('residual_price_v_score_squared_' + label + '.png')
    plt.show()

plot_score_v_price(df[mask_wht], line=92.5, ylabel='Score',
                    x='price_int', y='points', 
                    hue='variety', color='white')

plot_score_v_price(df[mask_red], line=91.5, ylabel='Score',
                    x='price_int', y='points', 
                    hue='variety', color='red')

plot_score_v_price(df[mask_wht], line=None, ylabel='Score Squared',
                    x='price_int', y='model', 
                    hue='variety', color='squared_white')

plot_score_v_price_fit(df[mask_wht], line=None, ylabel='Score Squared',
                    x='price_int', y='model', 
                    hue='variety', color='squared_white')

##now plot residuals
wine_masks  = [mask_cab , mask_pin , mask_syr , mask_zin , mask_mer,
              mask_chd, mask_sav, mask_rsl]
wine_labels = ['cabernet sauvignon', 'pinot noir', 'syrah', 'zinfandel', 
              'merlot', 'chardonnay', 'sauvignon blanc', 'riesling']

#for masks, labels in zip(wine_masks, wine_labels):
#    plot_avg_residuals(df[masks], xname='price_int',
#                       yname='points', label=labels)
#
#    


fig, ax = plt.subplots()
sns.kdeplot(df[mask_mer]['points'], label = 'merlot', shade=True)
ax.legend(loc='best', fontsize=20)
ax.set_xlim(xmin=80, xmax=100)
ax.set_ylabel('Frequency', fontsize=20)
ax.set_xlabel('Score', fontsize=20)
ax.tick_params(labelsize='large')
plt.tight_layout()
plt.savefig('price_kde_merlot.png')
plt.show()

fig, ax = plt.subplots()
sns.distplot(df[mask_mer]['points'], label = 'merlot', bins=19)
ax.legend(loc='best', fontsize=20)
ax.set_xlim(xmin=80, xmax=100)
ax.set_ylabel('Frequency', fontsize=20)
ax.set_xlabel('Score', fontsize=20)
ax.tick_params(labelsize='large')
plt.tight_layout()
plt.savefig('score_hist_merlot.png')
plt.show()

fig, ax = plt.subplots()
sns.distplot(df[df['region'] == 'generic']['points'], label = 'generic', bins=18)
ax.legend(loc='best', fontsize=20)
ax.set_xlim(xmin=80, xmax=100)
ax.set_ylabel('Frequency', fontsize=20)
ax.set_xlabel('Score', fontsize=20)
ax.tick_params(labelsize='large')
plt.tight_layout()
plt.savefig('score_hist_generic.png')
plt.show()



def plot_avg_residuals_multiple(dlist, xname, label, file, ncol=1): 
    for i, df in enumerate(dlist):
        dg = df.copy()
        x = [x for x in range(5,101,5)]
        yvals = []
        for _, val in enumerate(x):
            mask = dg[xname] == val  
            yvals.append(dg[mask]['points'].mean())

        x = np.array(x)
        yvals = np.array(yvals)
        sns.residplot(x=x, y=yvals, label=label[i])
    ax = plt.gca()
    ax.legend(loc='best', fontsize=15, ncol=ncol)
    ax.set_xlabel('Price ($)', fontsize=20)
    ax.set_ylabel('Score Residual', fontsize=20)
    ax.tick_params(labelsize='large')
    ax.set_xlim(xmin=0, xmax=100)
    ax.set_ylim(ymin=-5, ymax=5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('residual_price_v_score_squared_' + file + '.png')
    plt.show()


white_masks  = [mask_chd, mask_sav, mask_rsl]
white_labels = ['chardonnay', 'sauvignon blanc', 'riesling']

dlist = [df[mask_chd], df[mask_sav], df[mask_rsl]]
plot_avg_residuals_multiple(dlist, xname='price_int', label=white_labels,
                            file='whites')


red_masks  = [mask_cab , mask_pin , mask_syr , mask_zin , mask_mer]
red_labels = ['cabernet sauvignon', 'pinot noir', 'syrah', 'zinfandel', 
              'merlot']

dlist = [df[mask_cab], df[mask_pin], df[mask_syr], df[mask_zin], df[mask_mer]]
plot_avg_residuals_multiple(dlist, xname='price_int', label=red_labels,
                            file='reds', ncol=2)



