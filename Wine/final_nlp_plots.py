#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final NLP plots

@author: joshmagee
Tue Oct  2 09:33:17 2018
"""

import wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
                            recall_score, classification_report

sns.set_palette("dark")

###############################################################################
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
              'sauvignon']
    stop.extend(domain)
    
    df[text_field] = \
        df[text_field].apply(lambda x: \
          ' '.join([item for item in x.split() if item not in stop]))
    return df

def prep_df_for_tfidf(df, variety, region):
    #only keep entries from the correct variety
    dg = df[(df['variety'] == variety)].copy()
    dg['region'] = dg['region'].apply(lambda x: 1 if x == region else 0)
    dg.drop(columns=['price', 'variety', 'points'], inplace=True)
    return dg

def compare_regions(x, region1, region2):
    if x == region1:
        return 0
    elif x == region2:
        return 1
    else:
        return np.nan


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
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Other', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
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
###############################################################################
    
df = wine.read_wine()
df = wine.clean_data(df)
df = wine.select_top10(df)

dtemp = df.copy()

#unnecessary for final submission
df['region'] = df['region_1'].apply(wine.convert_region1)
df.drop(columns=['taster_name', 'region_1', 'region_2', \
                 'designation', 'winery', 'title'], inplace=True)

df = standardize_text(df, 'description')

variety = df['variety'].unique()
regions = df['region'].unique()

'''
We want to determine the top ten descriptive words for each varietal and region.
That is a total of 8*7=56 entries.
'''
from sklearn.feature_extraction.text import CountVectorizer

def find_top10_words(df, variety, region):
    '''Function to fine the top 10 descriptions given a region and varietal'''
    #create mask to only look at relevant data
    mask = (df['variety'] == variety) & (df['region'] == region)
    
    #create and fit CountVectorizer
    cv = CountVectorizer()
    if df[mask]['description'].empty:
#        print('CHEESY PUFFS: ' + variety + ' ' + region)
        return ['None' for x in range(10)], sum(mask)
#    print(variety + ' ' + region + ' ' + str(sum(mask)))
    cv_fit = cv.fit_transform(df[mask]['description'])
    
    #collapse array of word counts and zip+sort to extract desired info
    cv_fit = cv_fit.toarray().sum(axis=0)
    features = cv.get_feature_names()
    cv_sorted = sorted(list(zip(features, cv_fit)), \
                   key=lambda x: x[1], reverse=True)   
    
    cv_sorted = [x[0] for x in cv_sorted]
    #return only top10 solutions
    return cv_sorted[:10], sum(mask)

dg = pd.DataFrame(columns=['variety', 'region', 'total', 'array'])
for grape in variety:
    for loc in regions:
        top10, total = find_top10_words(df, grape, loc)
        d = {'variety': grape,
             'region' : loc,
             'total'  : total,
             'array'  : [top10]}
        dt = pd.DataFrame(d)
        dg = pd.concat([dg, dt])

import dill
with open('wine_nlp_dataframe.dill', 'wb') as file:
    dill.dump(dg, file)




#input_variety = 'cabernet sauvignon'
# =============================================================================
# input_variety = 'pinot noir'
# region1  = 'north coast'
# region2  = 'central coast'
# region2  = 'generic'
#     
# df = standardize_text(df, 'description')
# #df = prep_df_for_tfidf(df, input_variety, region1)
# #df = prep_df_for_tfidf2(df, input_variety, region1, region2)
# df = df.dropna()
# 
# list_corpus = df["description"].tolist()
# list_labels = df["region"].tolist()
# 
# #investigate relative weights of features in word doc
# from sklearn.feature_extraction.text import CountVectorizer
# 
# dg = df[df['region'] == 1]
# dg = dg.drop(columns=['region'])
# cv = CountVectorizer()
# cv_fit = cv.fit_transform(dg['description'])
# 
# cv_fit_flat = cv_fit.toarray().sum(axis=0)
# cv_fit_flat_norm = cv_fit_flat/np.sum(cv_fit_flat)
# features = cv.get_feature_names()
# 
# #now need to zip features with feature frequency
# cv_sorted = sorted(list(zip(features, cv_fit_flat_norm)), \
#                    key=lambda x: x[1], reverse=True)
# 
# cv_cdf = [x[1] for x in cv_sorted]
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(cv_cdf)), np.cumsum(cv_cdf)/np.sum(cv_cdf), 'r')
# ax.set_title('Cumulative Distribution Function of Word Frequency')
# ax.set_ylabel('Cumulative sum (%)')
# ax.set_xlabel('Word features ordered by frequency')
# ax.set_xlim(xmin=0, xmax=100)
# ax.grid(alpha=0.3)
# fig.tight_layout()
# plt.savefig('word_cdf.png')
# plt.show()
# 
# =============================================================================


# =============================================================================
# 
# def tfidf(data):
#     tfidf_vectorizer = TfidfVectorizer()
#     train = tfidf_vectorizer.fit_transform(data)
#     return train, tfidf_vectorizer
# 
# 
# X_train, X_test, y_train, y_test = \
#     train_test_split(list_corpus, list_labels, \
#                      test_size=0.2, random_state=42,stratify=list_labels)
# 
# X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
# X_test_tfidf = tfidf_vectorizer.transform(X_test)
# 
# clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
#                          multi_class='multinomial', n_jobs=-1, random_state=40)
# clf_tfidf.fit(X_train_tfidf, y_train)
# 
# y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
# 
# accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
# print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf, 
#                                                                        recall_tfidf, f1_tfidf))
# 
# 
# 
# importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
# top_scores = [a[0] for a in importance_tfidf[0]['tops']]
# top_words = [a[1] for a in importance_tfidf[0]['tops']]
# bottom_scores = [a[0] for a in importance_tfidf[0]['bottom']]
# bottom_words = [a[1] for a in importance_tfidf[0]['bottom']]
# 
# plot_important_words(top_scores, top_words, bottom_scores, bottom_words, \
#                      "Most important words for relevance")
# plt.savefig('napa_v_notnapa_pint_top10_words.png')
# 
# cm2 = confusion_matrix(y_test, y_predicted_tfidf)
# fig = plt.figure(figsize=(10, 10))
# plot = plot_confusion_matrix(cm2, classes=['Napa','Other'], \
#                              normalize=False, title='Confusion matrix')
# plt.show()
# plt.savefig('napa_v_notnapa_pinot_confusion_matrix.png')
# print("TFIDF confusion matrix")
# print(cm2)
# print("BoW confusion matrix")
# 
# =============================================================================





