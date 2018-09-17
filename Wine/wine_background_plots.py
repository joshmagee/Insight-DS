#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Background wine plots

@author: joshmagee
Mon Sep 17 09:34:15 2018
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("dark")

df = pd.read_csv('/Users/joshuamagee/Projects/Python/Jobs/Insight/wine_institute_retail_list.txt', \
                 delim_whitespace=True)

#total California wine production has increased 60% over the last 15 years
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Table'], label='Table')
ax.plot(df['Year'], df['Dessert'], label='Dessert')
ax.plot(df['Year'], df['Sparkling'],  label='Sparkling')
ax.plot(df['Year'], df['Total'],  label='Total')
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



