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


df = wine.read_wine()
df = wine.clean_data(df)
df = wine.select_top10(df)

conn = sqlite3.connect('wine_data_clean.db')

df.to_sql("Wine", conn, if_exists="replace")


