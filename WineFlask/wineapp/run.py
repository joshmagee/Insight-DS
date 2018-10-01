#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run file for WineFlask

@author: joshmagee
Thu Sep 27 23:47:48 2018
"""

from WinefFlask import app

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import dill

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = SnowballStemmer('english')
def tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    stems = [stemmer.stem(x) for x in tokens]
    return stems


try:
    base_dir = '/Users/joshuamagee/Projects/Python/Insight-DS/WineFlask/'
    tfidf_vectorizer = dill.load(open(base_dir+'tfidf_vectorizer.m', 'rb'))
    c = dill.load(open(base_dir+"trained_pipeline.m", 'rb'))
except:
    print('first import failed, better luck next time!')


app.run(debug=True)
