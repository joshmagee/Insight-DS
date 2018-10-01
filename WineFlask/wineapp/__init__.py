#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask module initialization

@author: joshmagee
Thu Sep 27 08:16:42 2018
"""

from flask import Flask
app = Flask(__name__,static_url_path='/static')
from wineapp import views
