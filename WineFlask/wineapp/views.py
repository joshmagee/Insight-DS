#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask views

@author: joshmagee
Thu Sep 27 12:56:52 2018
"""

import dill
import numpy as np
from wineapp import app
from flask import request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


###############################################################################
#to prep for if-idf, want to drop everything but description, variety = input,
#and we want to one hot encode region to input=1, everything else 0
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r",", " ")
    df[text_field] = df[text_field].str.replace(r"  ", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def prep_df_for_tfidf(df, variety, region):
    #only keep entries from the correct variety
    dg = df[(df[variety] == True)].copy()
    dg['region'] = dg[region].apply(lambda x: True if x == 1 else False)
    dg.drop(columns=['price', 'points', 'south cali', \
                     'central coast', 'far north', 'generic', \
                     'inland valleys', 'north coast', 'sierra foothills', \
                     'cabernet sauvignon', 'chardonnay', 'merlot', \
                     'pinot noir', 'riesling', 'sauvignon blanc', 'syrah', \
                     'zinfandel'], inplace=True)
    print(dg.head())
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
###############################################################################



@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/output')
def text_output():
    # pull input text and city from input field and store i
    if request.method == 'GET':
        price = request.args.get('input_price')
        input_variety = request.args.get('input_varietal')
        input_region  = request.args.get('input_region')
    
    print('THE TYPE OF PRICE IS ')
    print(type(price))
    #get dataframe
    with open('wine_dataframe.dill', 'rb') as file:
        df = dill.load(file)
    
    with open('wine_price_transform.dill', 'rb') as xform:
        scaler_xform = dill.load(xform)  
        
    with open('wine_lr_model.dill','rb') as model:
        lr= dill.load(model)

    price_scaled =  scaler_xform.transform([[float(price)]])
    df_pred = df.head(1)
    df_pred = df_pred.drop(columns=['description', 'points'])
    #set everything to 0
    for col in df_pred.columns:
        if col == 'price':
            df_pred[col].values[:] = price_scaled
        elif col == input_variety:
            df_pred[col].values[:] = 1
        elif col == input_region:
            df_pred[col].values[:] = 1
        else:
            df_pred[col].values[:] = 0

    #get linear regression model
    score = lr.predict(df_pred)
    score = int(round(score[0]))
    
    df_mask = \
        (df[input_variety] == True ) & (df[input_region] == True)
    tenth = np.percentile(df[df_mask]['price'],10)
    halve = np.percentile(df[df_mask]['price'],50)
    ninth = np.percentile(df[df_mask]['price'],90)
    
    tenth = scaler_xform.inverse_transform(tenth.reshape(-1,1))
    halve = scaler_xform.inverse_transform(halve.reshape(-1,1))
    ninth = scaler_xform.inverse_transform(ninth.reshape(-1,1))
    
    tenth = int(round(tenth[0][0]))
    halve = int(round(halve[0][0]))
    ninth = int(round(ninth[0][0]))
    
    #now for the NLP tasting characteristics portion
    df_desc = prep_df_for_tfidf(df, input_variety, input_region)
    df_desc = standardize_text(df_desc, 'description')
    
    X = df_desc['description'].tolist()
    y = df_desc['region'].tolist()
    
    tfidf_vectorizer = TfidfVectorizer()
    X_idf = tfidf_vectorizer.fit_transform(X)
    
    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', \
                            solver='newton-cg',multi_class='multinomial', \
                            n_jobs=-1, random_state=40)
    clf_tfidf.fit(X_idf, y)
    
    importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
    top_words = [a[1] for a in importance_tfidf[0]['tops']]
    top10 = ', '.join(top_words[::-1])
        
    try:
        return render_template("output.html", \
                               varietal = input_variety, \
                               region = input_region, \
                               score = score,
                               price=halve, pricetenth=tenth, \
                               priceninetieth=ninth, \
                               tastingnotes=top10)
    except:
        return render_template('error.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/error')
def error():
    return render_template('error.html')


