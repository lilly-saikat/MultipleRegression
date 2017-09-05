# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:21:16 2017

@author: Saikat
"""

# =============================================================================
#  Importing Modules
# =============================================================================
import numpy as np
import os

# Cross Validation and Hyperparameter Tuning
from statsmodels.sandbox.tools import cross_val
import seaborn as sns
import matplotlib.pyplot as plt

# For Linear model
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
# =============================================================================
#  Define Location
# =============================================================================
path = os.getcwd()


# =============================================================================
# Create training and testing data
# =============================================================================

def bootstrap_it(data, split):
    data['is_train'] = np.random.uniform(low=0, high=1, size=len(data)) < split
    data_train = data.loc[data['is_train'] == True]
    data_test = data.loc[data['is_train'] == False]
    del data_train['is_train']
    del data_test['is_train']
    print(data_train.shape)
    print(data_test.shape)
    return data_train, data_test


# K-Fold CrossValidation
def k_fold_ols(data, k, dep_var, drop_var):
    kf = cross_val.KFold(data.shape[0], k=k)
    model = {}
    count = 0
    mape = {}
    for train_ind, test_ind in kf:
        X_train, X_test, y_train, y_test = cross_val.split(
                train_ind, test_ind, data.drop([drop_var],
                                               axis=1,
                                               inplace=False), data[dep_var])
        model[count] = sm.OLS(y_train, X_train).fit()
        print(model[count].summary())
        mape[count] = np.mean(np.abs(y_test -
                              model[count].predict(X_test))/y_test)
        count += 1
    plt.figure(figsize=[20, 10])
    sns.distplot(list(mape.values()))
    plt.savefig(path+"/reports/figures/k_fold_valid.jpg")
    print('Bias:')
    print(np.mean(list(mape.values())))
    print()
    print('Variance:')
    print(np.std(list(mape.values())))


def hyperparameter_linear(data, k, dep_var, drop_var):
    tuned_parameters = [{'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                   0.8, 0.9, 1.0]}]
    model = GridSearchCV(linear_model.Ridge(alpha=1.), tuned_parameters, cv=k)
    model.fit(data.drop([drop_var], axis=1, inplace=False), data[dep_var])
    sorted(model.cv_results_.keys())
    print(model.cv_results_['mean_test_score'])
    print(model.cv_results_['mean_train_score'])
