#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick n' dirty model to predict wine scores

@author: joshmagee
Tue Sep 18 14:40:00 2018
"""


import wine
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("dark")

#from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
dg = df.copy()
df.drop(columns=['taster_name', 'region_1', 'region_2', \
                 'designation', 'winery', 'title'], inplace=True)
#add in color feature
df['color'] = df['variety'].apply(wine.coloring)

#let's just look at red wine
#df = df[df['color'] == 'red']
#df = df[df['color'] == 'white']

normalize_price = False
linear_regression = True
north_coast_pinot_only = False
tree_regression = False
tree_classifier = False

if north_coast_pinot_only:
    normalize_price = True 

if linear_regression:
    normalize_price = True

if normalize_price:
    scaler = StandardScaler()
    df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
    
#start one hot encoding shit
if normalize_price or tree_regression:
    dfvar = pd.get_dummies(df['variety'])
    dfreg = pd.get_dummies(df['region'])
    df = pd.concat([df,dfreg,dfvar], axis=1)
    df.drop(columns=['variety', 'color', 'region'], inplace=True)

#if we want the score to be categorical
if tree_classifier:
    df['points'] = df['points'].apply(wine.convert_score_to_int_category)
    
#test train split
if linear_regression:
    from scipy.stats import norm, chi2, chisquare
    from sklearn.linear_model import LinearRegression #, RidgeCV, LassoCV, ElasticNetCV
    with open('wine_dataframe.dill', 'wb') as file:
        dill.dump(df, file)
   
    df.drop(columns=['description'], inplace=True)  

    y = df['points']
    X_train, X_test, y_train, y_test = train_test_split( \
        df.drop(columns=['points']), y, train_size=0.80, random_state=42)
    X_train = df.drop(columns='points')
    lr = LinearRegression()
    lr.fit(X_train, y) # reshape to column vector
    wine.plot_coefs(X_train, np.fabs(lr.coef_), \
                    'Coefficients in Simple Linear Regression Model')
    plt.savefig('simple_regression_coefs.png')
    
    y_pred = lr.predict(X_test)
    
    with open('wine_lr_model.dill', 'wb') as model:
        dill.dump(lr, model)

    with open('wine_price_transform.dill', 'wb') as xform:
        dill.dump(scaler, xform)
    
    # The mean squared error
    print("Mean squared error: ", \
          mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: ', \
          r2_score(y_test, y_pred))
    
    y = y_test - y_pred
    n, bins, _ = plt.hist(y, bins=50, range=[-10,10], normed=True)
    mu, sigma = norm.fit(y)
    fit = norm.pdf(bins[:-1], mu, sigma)
    plt.plot(bins[:-1], fit, 'r--', linewidth=2)
    plt.xlabel('Difference of Real Score from Predicted')
    plt.ylabel('')
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_accuracy_reds.png')
    plt.tight_layout()
    plt.show()
    
    index_low = np.where(bins == -3)
    index_up  = np.where(bins == 3)
    
    bin_width = bins[1]-bins[0]
    integral = bin_width * sum(n[20:35])
    tot_integral = bin_width * sum(n[:])
    
    #calculate chi2 and p-value
    chisq, p = chisquare(n, f_exp=fit)
    #now calculate usual test statistic
    pval = 1 - chi2.cdf(chisq, len(bins[:-1]-2))
    #alternate
#    pval = chisqprob(chisq, len(bins[:-1]-2))
    
    print('Mean +- sigma: ' + str(mu) + ' ' + str(sigma))
    print('Total area: ' + str(integral/tot_integral))




#now let's just look at one example
#north coast pinot noir linear regression
if north_coast_pinot_only:
    X = df[(df['region']=='north coast') & (df['variety']=='pinot noir')]['price']
    y = df[(df['region']=='north coast') & (df['variety']=='pinot noir')]['points']
    lr = LinearRegression()
    lr.fit(X.values.reshape(-1,1), y) # reshape to column vector
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_[0])
    
    y_pred = lr.predict(X.values.reshape(-1,1))
    
    plt.plot(X, y, 'o', label='training data')
    plt.plot(X, y_pred, label='model prediction', \
             color='r', linewidth=1.5, linestyle='--')
    ax = plt.gca()
    plt.title('California North Coast Pinot Noirs')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Score')
    ax.set_xlim(xmax=101)
    ax.legend()
    
    dg = df[(df['region']=='north coast') & (df['variety']=='pinot noir')]
    g = sns.jointplot('price', 'points', data=dg, kind="hex")
    ax = plt.gca()
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Score')
    plt.legend()
    plt.savefig('pinot_northcoast_score_v_price.png')
    plt.show()
 

#random forest and ensemble methods
if tree_regression:
    from sklearn import model_selection, ensemble
    from sklearn import tree
    y = df['points']
    X = df.drop(columns=['points'])
    
    cv = model_selection.ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
    def compute_error(clf, X, y):
        return - model_selection.cross_val_score( \
                                                 clf, X, y, cv=cv, \
                                                 scoring='neg_mean_squared_error'\
                                                 ).mean()
    
    tree_reg = tree.DecisionTreeRegressor()
    extra_reg = ensemble.ExtraTreesRegressor()
    forest_reg = ensemble.RandomForestRegressor()
    
    model_performance = pd.DataFrame([
        ("Mean Model", y.var()),
        ("Decision Tree", compute_error(tree_reg, X, y)),
        ("Random Forest", compute_error(forest_reg, X, y)),
        ("Extra Random Forest", compute_error(extra_reg, X, y)),
    ], columns=["Model", "MSE"])
    plt.figure()
    ax = model_performance.plot(x="Model", y="MSE", kind="Bar", legend=False)
    plt.xticks(rotation=45)
    ax.set_ylabel('Mean Squared Error (score)')
    ax.set_xlabel('')
    ax.grid(alpha=0.3)
    plt.savefig('random_forrest_regressor.png')
    plt.show()

    #compute residuals?

#now random forest classifier
if tree_classifier:
    from sklearn.ensemble import RandomForestClassifier
    
    y = df['points']
    X_train, X_test, y_train, y_test = train_test_split( \
        df.drop(columns=['points']), y, train_size=0.80, random_state=42)
    X_train = df.drop(columns='points')
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y)
    print(clf.score(X_test, y_test))
    print(clf.feature_importances_)
    
    wine.plot_coefs(X_train, clf.feature_importances_, \
                    'Feature Importances (Random Forest Classifier)')
    plt.savefig('random_forest_feature_importance.png')
    
#could plot price and score for generic wines. Expect them to be somewhat correlated
    








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

