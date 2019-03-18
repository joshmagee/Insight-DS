#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module of common/useful functions for WineFlask.
Broken into two parts:
    1) regular dataframe manipulation and plotting
    2) NLP parts

@author: joshmagee
Tue Sep 18 14:40:55 2018
"""

##############################################################################
############# Dataframe plotting/manipulation functions ######################    
##############################################################################
import re
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

def convert_score_to_category(val):
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

def convert_score_to_int_category(val):
    if val >= 97.0:
        return '1'
    elif val >= 93.0:
        return '2'
    elif val >= 90.0:
        return '3'
    elif val >= 87.0:
        return '4'
    elif val >= 84.0:
        return '5'
    elif val >= 80.0:
        return '6'
    else:
        return '7'

def convert_int_to_score(val):
    if val == 1:
        return 97.0
    elif val == 2:
        return 90.0
    elif val == 3:
        return 90.0
    elif val == 4:
        return 87.0
    elif val == 5:
        return 84.0
    elif val == 6:
        return 80.0
    else:
        return 75


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
    '''Testing methods'''
    df = read_wine()
    df = clean_data(df)
    df = select_top10(df)
    
    
##############################################################################
############# Natural Language Processing functions ##########################    
##############################################################################
import itertools
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, \
                            precision_score, recall_score
    
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"\d{1,4}", " ")
    df[text_field] = df[text_field].str.replace(r",", " ")
    df[text_field] = df[text_field].str.replace(r"  ", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    
    stop = stopwords.words('english')
    domain = ['carneros', 'sonoma', 'annapolis', 'paso', 'livermore', 'napa',
              'robles', 'stags', 'soon', 'lot', 'santa', 'central', 'coast',
              'calera', 'ross', 'county', 'rita', 'fort', 'marin', 'got',
              'town', '60', '17', 'wine', 'flavors', 'cabernet', 'pinot',
              'noir', 'zin', 'syrah', 'chardonnay', 'riesling', 'blanc', 
              'sauvignon', 'bottling', 'well', 'palate', 'asian', 'black',
              'like']
    stop.extend(domain)
    
    df[text_field] = \
        df[text_field].apply(lambda x: \
          ' '.join([item for item in x.split() if item not in stop]))
    return df

def compare_regions(x, region1, region2):
    if x == region1:
        return 0
    elif x == region2:
        return 1
    else:
        return np.nan


def prep_df_for_tfidf(df, variety, region):
    #only keep entries from the correct variety
    dg = df[(df['variety'] == variety)].copy()
    dg['region'] = dg['region'].apply(lambda x: 1 if x == region else 0)
    dg.drop(columns=['price', 'variety', 'points'], inplace=True)
    return dg


def prep_df_for_tfidf2(df, variety, region1, region2):
    #only keep entries from the correct variety
    dg = df[(df['variety'] == variety)].copy()
    dg['region'] = dg['region'].apply(compare_regions, args=(region1, region2,))
    dg.drop(columns=['price', 'variety', 'points'], inplace=True)
    return dg



def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Napa', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=20)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Other', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=20)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show()


def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt
    
    
    
    



