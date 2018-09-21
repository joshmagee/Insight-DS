#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick n' dirty model to predict wine scores

@author: joshmagee
Tue Sep 18 14:40:00 2018
"""


import wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("dark")

#from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
#from sklearn.pipeline import Pipeline

df = wine.read_wine()
df = wine.clean_data(df)
df = wine.select_top10(df)

#df['year'] = df['description'].apply(wine.extract_year)
df['region'] = df['region_1'].apply(wine.convert_region1)

'''
    It turns out that 'general california wine' covers major players:
        Kendall-Jackson, Robert Mondavi, Josh Cellars, Cupcake, and others
    Scanning by eye, region_1  larrgely matches region_2, so region_2 can
        be dropped. Also dropping incomplete taster information.
'''
#drop taster_name, region_2, winery
#drop designation, but can probably add that back later
df.drop(columns=['taster_name', 'region_1', 'region_2', 'description', \
                 'designation', 'winery', 'title'], inplace=True)
#add in color feature
df['color'] = df['variety'].apply(wine.coloring)


#let's just look at red wine
df = df[df['color'] == 'red']

#start one hot encoding shit
#dfcol = pd.get_dummies(df['color'])
dfvar = pd.get_dummies(df['variety'])
dfreg = pd.get_dummies(df['region'])
#df = pd.concat([df,dfreg,dfcol,dfvar], axis=1)
df = pd.concat([df,dfreg,dfvar], axis=1)

df.drop(columns=['variety', 'color', 'region'], inplace=True)

#normalize pricing data
scaler = StandardScaler()
df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))

#test train split
y = df['points']

X_train, X_test, y_train, y_test = train_test_split( \
    df.drop(columns=['points']), y, train_size=0.80, random_state=42)

X_train = df.drop(columns='points')
lr = LinearRegression()
lr.fit(X_train, y) # reshape to column vector
wine.plot_coefs(X_train, lr.coef_, "Coefficients in Simple Model")

y_pred = lr.predict(X_test)

# The mean squared error
print("Mean squared error: ", \
      mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('R2 score: ', \
      r2_score(y_test, y_pred))

#print("RMSE on Training set :", rmse_cv_train(lr).mean())






# =============================================================================
# #now let's just look at one example
# #north coast pinot noir
# X = df[(df['region']=='north coast') & (df['variety']=='pinot noir')]['price']
# y = df[(df['region']=='north coast') & (df['variety']=='pinot noir')]['points']
# lr = LinearRegression()
# lr.fit(X.values.reshape(-1,1), y) # reshape to column vector
# print('Intercept: ', lr.intercept_)
# print('Slope: ', lr.coef_[0])
# 
# y_pred = lr.predict(X.values.reshape(-1,1))
# 
# plt.plot(X, y, 'o', label='training data')
# plt.plot(X, y_pred, label='model prediction', \
#          color='r', linewidth=1.5, linestyle='--')
# ax = plt.gca()
# plt.title('California North Coast Pinot Noirs')
# ax.set_xlabel('Price ($)')
# ax.set_ylabel('Score')
# ax.set_xlim(xmax=101)
# ax.legend()
# plt.show()
# 
# dg = df[(df['region']=='north coast') & (df['variety']=='pinot noir')]
# g = sns.jointplot('price', 'points', data=dg, kind="hex")
# ax = plt.gca()
# ax.set_xlabel('Price ($)')
# ax.set_ylabel('Score')
# plt.legend()
# plt.show()
# =============================================================================


#significance tests for these coefficients
# =============================================================================
# alphas=[x for x in np.logspace(-6, 6, 100)]
# 
# ridge = RidgeCV(alphas=alphas)
# ridge.fit(X,y)
# print('Best alpha: ', ridge.alpha_)
# wine.plot_coefs(X, ridge.coef_, "Coefficients in Ridge Model")
# #print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
# 
# lasso = LassoCV(alphas=alphas, max_iter = 50000, cv = 10)
# lasso.fit(X,y)
# print('Best alpha: ', lasso.alpha_)
# wine.plot_coefs(X, lasso.coef_, "Coefficients in Lasso Model")
# 
# 
# elastic = ElasticNetCV()
# elastic = ElasticNetCV(alphas=[x for x in np.logspace(-3, 6, 100)], \
#                        max_iter = 50000, cv = 10, \
#                 l1_ratio = [float(x/100) for x in range(0,100,5)] \
#                 )
# elastic.fit(X,y)
# print('Best alpha: ', elastic.alpha_)
# print('L1 ratio: ', elastic.l1_ratio_)
# wine.plot_coefs(X, elastic.coef_, "Coefficients in ElasticNet Model")
# 
# =============================================================================
# =============================================================================
# bins = pd.cut(df['price'], bins=range(0,101,5))
# df.groupby(bins)['points'].agg([np.mean,np.std])
# s = df.groupby(bins)['points'].agg([np.mean,np.std]).reset_index()
# s['price'] = pd.Series(np.arange(0,101,5), name='price')
# s['price'][0] = 1
# 
# X = s['price']
# y = s['mean']
# w = s['std']
# 
# lr = LinearRegression()
# lr.fit(X.values.reshape(-1,1), y, sample_weight=w)
# print('Intercept: ', lr.intercept_)
# print('Slope: ', lr.coef_[0])
# 
# y_pred = lr.predict(X.values.reshape(-1,1))
# 
# fig, ax = plt.subplots()
# ax.errorbar(X, y, yerr=w, marker='o', label='training data')
# ax.errorbar(X, y_pred, label='model prediction', \
#          color='r', linewidth=1.5, linestyle='--')
# ax = plt.gca()
# plt.title('California North Coast Pinot Noirs')
# ax.set_xlabel('Price ($)')
# ax.set_ylabel('Score')
# ax.set_xlim(xmax=101)
# ax.legend()
# plt.show()
# =============================================================================
#from sklearn.svm import SVR
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#y_rbf = svr_rbf.fit(X.values.reshape(-1,1), y).predict(X.values.reshape(-1,1))
#y_lin = svr_lin.fit(X.values.reshape(-1,1), y).predict(X.values.reshape(-1,1))
#y_poly = svr_poly.fit(X.values.reshape(-1,1), y).predict(X.values.reshape(-1,1))
#lw = 2
#plt.scatter(X, y, color='darkorange', label='data')
#plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()


#add in year category

