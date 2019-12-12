#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:56:11 2019

@author: barbara
"""

import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime
from dateutil.parser import parse
import sklearn

plt.style.use('fivethirtyeight')

#%% NOTE

# This script contains all the code necessary for the offline analysis, except
# for the code to create the map. 
# It includes multiple trials of different types of regressions to see which
# works best (Linear, Lasso, and Ridge), before showing the correlation
# matrices and scores indicating which variables might be most correlated
# with chowkidar_proportions.

#%% loading and exploring background data

background_info = pd.read_csv('../Data/background_info.csv')
mps = pd.read_csv('../Data/MP_Twitter_Handles.csv')

#%% Creating indv_d = state_id collates to pc_id

merged = pd.merge(left=mps, right=background_info,
                  left_on='totvotpoll', right_on='Votes')
clean_merged = merged[['st_name', 'year', 'indv_id', 'pc_no', 'pc_name', 'pc_type', 'cand_name',
                 'cand_sex', 'partyname', 'partyabbre', 'totvotpoll', 'electors', 'max_votes', 'handle',
                 'Valid_Votes', 'Constituency_Name', 'Sub_Region', 'N_Cand', 'Turnout_Percentage',
                 'Vote_Share_Percentage', 'Margin', 'Margin_Percentage', 'ENOP', 'pid', 'Party_type_TCPD',
                 'Party_ID', 'last_poll', 'Contested', 'No_Terms', 'Turncoat', 'Incumbent',
                 'Recontest']]
clean_merged['bjp_or_not'] = [1 if p == 'BJP' else 0 for p in clean_merged['partyabbre']]

# Need to further strip the handles and remove the '-'
clean_merged = clean_merged.loc[clean_merged['handle'] != '-']
clean_merged = clean_merged.loc[clean_merged['handle'] != ',-']
clean_merged = clean_merged.loc[clean_merged['cand_name'] != 'Gawali Bhavana Pundlikrao']
clean_merged['handle'] = clean_merged['handle'].str.strip('@')
clean_merged['handle'] = clean_merged['handle'].str.rstrip()

#%% Merging clean_merged to chowkidar_proportions

chowkidar_proportions = pd.read_csv('../Results/chowkidar_proportions.csv').drop(['Unnamed: 0'], axis=1)
len(chowkidar_proportions.loc[chowkidar_proportions['chowkidar yes/no mean'] == 0].index)

total_info = pd.merge(left=clean_merged, right=chowkidar_proportions,
                  left_on='handle', right_on='source')

# Quick correlation
pearsonr(total_info['chowkidar yes/no mean'], total_info['bjp_or_not'])
total_info.corr(method='pearson')

# Creating a table of just BJP MPs for use in the future.
just_bjp = total_info.loc[total_info['partyname'] == 'BJP']
just_bjp.corr(method='pearson')

#%% Imports, train/test/validation split.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

# X: features used to predict chowkidar proportion
X = total_info.drop(['st_name', 'year', 'pc_name', 'pc_type', 'cand_name', 'cand_sex', 'partyname',
                     'partyabbre', 'handle', 'Constituency_Name', 'Sub_Region', 'pid', 'Party_type_TCPD',
                     'last_poll', 'Turncoat', 'Incumbent', 'Recontest', 'source', 'chowkidar yes/no mean'], axis=1)
# y: feature we are predicting
y = total_info['chowkidar yes/no mean']

np.random.seed(10)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80, test_size=0.20)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                    train_size=0.75, test_size=0.25)

#%% Linear Regression

# Creating lin_reg method and fit model
lin_reg = LinearRegression(normalize=True)
lin_model = lin_reg.fit(X_train, y_train)

lin_predicted = cross_val_predict(lin_reg, X, y, cv = 3)
print(r2_score(y, lin_predicted))

# plot the residuals on a scatter plot
plt.scatter(y, lin_predicted)
plt.title('Linear Model (OLS)')
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.show()

#%% Ridge

# Creating ridge_reg method and fit model
ridge_reg = Ridge() 
ridge_model = ridge_reg.fit(X_train, y_train)

ridge_predicted = cross_val_predict(ridge_reg, X, y, cv = 3)
print(r2_score(y, ridge_predicted))

# plot the residuals on a scatter plot
plt.scatter(y, ridge_predicted)
plt.title('Ridge')
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.show()

#%% Lasso

# Creating lasso_reg and fit
lasso_reg = Lasso(max_iter=10000)  
lasso_model = lasso_reg.fit(X_train, y_train)

lasso_predicted = cross_val_predict(lasso_reg, X, y, cv = 3)
print(r2_score(y, lasso_predicted))

# plot the residuals on a scatter plot
plt.scatter(y, lin_predicted)
plt.title('Lasso')
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.show()

#%% Re-doing for just BJP MPs

# X: features used to predict chowkidar proportion
X = just_bjp.drop(['st_name', 'year', 'pc_name', 'pc_type', 'cand_name', 'cand_sex', 'partyname',
                     'partyabbre', 'handle', 'Constituency_Name', 'Sub_Region', 'pid', 'Party_type_TCPD',
                     'last_poll', 'Turncoat', 'Incumbent', 'Recontest', 'source', 'chowkidar yes/no mean',
                     'bjp_or_not'], axis=1)
# y: feature we are predicting
y = just_bjp['chowkidar yes/no mean']

np.random.seed(10)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80, test_size=0.20)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                    train_size=0.75, test_size=0.25)


#%% checking coefficients for just_bjp

# X.columns contains all the non-categorical varials, so I could run a regression.
# Essentially what I think this does is rule out that any of the variables in X.columns
# have a significant impact on the chowkidar proportion, i.e they are not predictors of the 
# chowkidar proportion.

print(X.columns)

for i in X.columns:
    coeff = pearsonr(y, X[str(i)])
    print("Column: ", i, ", Coefficient: ", coeff)
    
# ^^ Really rough but essentially there is no strong correlation anywhere here

#%% Creating a heatmap with the original total_info to find out what is most correlated

import seaborn as sns 

new_total_info = total_info.copy().dropna(axis=1).drop(['year'], axis=1)
corr = new_total_info.corr()

# Code for the heatmap:
fig, ax = plt.subplots(figsize=(10,10)) 
fig.suptitle('Correlation Matrix for all MPs')

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
# And then setting the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
        
#%% And also doing it with just_bjp

new_just_bjp = just_bjp.copy().dropna(axis=1).drop(['year', 'Party_ID'], axis=1)
bjp_corr = new_just_bjp.corr()

# These two lines make the heatmap larger so it's easier to read, because a lot of 
# variables are represented. Can and should be changed according to display size.
fig, ax = plt.subplots(figsize=(10,10)) 
fig.suptitle('Correlation Matrix for BJP MPs')

ax = sns.heatmap(
    bjp_corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    xticklabels=True,
    yticklabels=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


#%% Calculating the correlation between chowkidar_proportions and other columns

# Correlation for all MPs
corr_all = new_total_info[new_total_info.columns[0:]].corr(
        )['chowkidar yes/no mean'][:-1].sort_values(ascending=False).to_frame()

# Correlation for just BJP MPs
corr_bjp = new_just_bjp[new_just_bjp.columns[0:]].corr(
        )['chowkidar yes/no mean'][:-1].sort_values(ascending=False).to_frame()

# Defining a new table with all MPs not in the BJP, then getting the correlation.
not_bjp = total_info.loc[total_info['partyname'] != 'BJP']
new_not_bjp = not_bjp.copy().dropna(axis=1).drop(['year'], axis=1)
corr_not_bjp = new_not_bjp[new_not_bjp.columns[0:]].corr(
        )['chowkidar yes/no mean'][:-1].sort_values(ascending=False).to_frame()

# Save to csv to keep the data.
corr_all.to_csv(r'../Results/corr_all.csv')
corr_bjp.to_csv(r'../Results/corr_bjp.csv')
corr_not_bjp.to_csv(r'../Results/corr_not_bjp.csv')














