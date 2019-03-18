#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final NLP plots

@author: joshmagee
Tue Oct  2 09:33:17 2018
"""

import wine
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sns.set_palette("dark")

##############################################################################
############# Function definitions ###########################################    
##############################################################################
def find_top10_words(df, variety, region):
    '''Function to find the top 10 descriptions given a region and varietal'''
    #create mask to only look at relevant data
    mask = (df['variety'] == variety) & (df['region'] == region)
    
    #create and fit CountVectorizer
    cv = CountVectorizer()
    if df[mask]['description'].empty:
        return ['None' for x in range(10)], sum(mask)
    cv_fit = cv.fit_transform(df[mask]['description'])
    
    #collapse array of word counts and zip+sort to extract desired info
    cv_fit = cv_fit.toarray().sum(axis=0)
    features = cv.get_feature_names()
    cv_sorted = sorted(list(zip(features, cv_fit)), \
                   key=lambda x: x[1], reverse=True)   
    
    cv_sorted = [x[0] for x in cv_sorted]
    #return only top10 solutions
    return cv_sorted[:10], sum(mask)

def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer
    
##############################################################################
##############################################################################
df = wine.read_wine()
df = wine.clean_data(df)
df = wine.select_top10(df)

dtemp = df.copy()

#continue to process dataframe for NLP summary use
df['region'] = df['region_1'].apply(wine.convert_region1)
df.drop(columns=['taster_name', 'region_1', 'region_2', \
                 'designation', 'winery', 'title'], inplace=True)

#process text description feature
df = wine.standardize_text(df, 'description')

variety = df['variety'].unique()
regions = df['region'].unique()

'''
We want to determine the top ten descriptive words for each varietal and region.
That is a total of 8*7=56 entries.
'''
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


#serialize data for website use
with open('wine_nlp_dataframe.dill', 'wb') as file:
    dill.dump(dg, file)



#explore data here
#choose varietal and region
input_variety = 'cabernet sauvignon'
#input_variety = 'pinot noir'
region1  = 'north coast'
region2  = 'central coast'
#region2  = 'generic'
    
#df = wine.standardize_text(df, 'description')
df = wine.prep_df_for_tfidf(df, input_variety, region1)
df = df.dropna()

list_corpus = df["description"].tolist()
list_labels = df["region"].tolist()

#investigate relative weights of features in word doc
dg = df[df['region'] == 1]
dg = dg.drop(columns=['region'])
cv = CountVectorizer()
cv_fit = cv.fit_transform(dg['description'])

cv_fit_flat = cv_fit.toarray().sum(axis=0)
cv_fit_flat_norm = cv_fit_flat/np.sum(cv_fit_flat)
features = cv.get_feature_names()

#zip features with feature frequency
cv_sorted = sorted(list(zip(features, cv_fit_flat_norm)), \
                   key=lambda x: x[1], reverse=True)

cv_cdf = [x[1] for x in cv_sorted]
fig, ax = plt.subplots()
ax.plot(np.arange(len(cv_cdf)), np.cumsum(cv_cdf)/np.sum(cv_cdf), 'r')
ax.set_title('Cumulative Distribution Function of Word Frequency')
ax.set_ylabel('Cumulative sum (%)')
ax.set_xlabel('Word features ordered by frequency')
ax.set_xlim(xmin=0, xmax=100)
ax.grid(alpha=0.3)
fig.tight_layout()
#plt.savefig('word_cdf.png')
plt.show()



#stratified test/train split
X_train, X_test, y_train, y_test = \
    train_test_split(list_corpus, list_labels, \
                     test_size=0.2, random_state=42,stratify=list_labels)

#note TFIDF vectorizer is not really what we need
#due to similar descriptors, we end up really fitting noise
#simple count vectorizer works better
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf.fit(X_train_tfidf, y_train)

y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)

accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = \
    wine.get_metrics(y_test, y_predicted_tfidf)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % \
      (accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))

importance_tfidf = \
    wine.get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
top_scores = [a[0] for a in importance_tfidf[0]['tops']]
top_words = [a[1] for a in importance_tfidf[0]['tops']]
bottom_scores = [a[0] for a in importance_tfidf[0]['bottom']]
bottom_words = [a[1] for a in importance_tfidf[0]['bottom']]

wine.plot_important_words(top_scores, top_words, bottom_scores, bottom_words, \
                     "Most important words for relevance")
#plt.savefig('napa_v_notnapa_cab_top10_words.png')

cm2 = confusion_matrix(y_test, y_predicted_tfidf)
fig = plt.figure(figsize=(10, 10))
plot = wine.plot_confusion_matrix(cm2, classes=['Napa','Other'], \
                             normalize=False, title='Confusion matrix')
plt.show()
#plt.savefig('napa_v_notnapa_cab_confusion_matrix.png')
print("TFIDF confusion matrix")
print(cm2)
print("BoW confusion matrix")






