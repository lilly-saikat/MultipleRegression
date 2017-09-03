# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:21:16 2017

@author: Saikat
"""

# =============================================================================
#  Importing Modules
# =============================================================================
import os
# For Linear model
import statsmodels.api as sm
import pickle
from pyearth import Earth
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================
#  Define Location
# =============================================================================
path = os.getcwd()

# =============================================================================
# Create training and testing data
# =============================================================================


def ols_model(train_data, dep_var, drop_vars):
    # Run the OLS model
    model = sm.OLS(train_data.loc[:, dep_var], train_data.drop(drop_vars,
                   axis=1, inplace=False)).fit()
    print(model.summary())
    model_filename = path+'/models/models.p'
    pickle.dump(model, open(model_filename, 'wb'))
    return model


def regularized_model(data, dep_var, drop_vars):
    regularized_model = sm.OLS(data.loc[:, dep_var], data.drop(drop_vars,
                               axis=1,
                               inplace=False)).fit_regularized(alpha=0.1)
    print(regularized_model.params)
    return regularized_model


def mars_comp_model(data, penalty, vibrose, dep_var, ind_var):
    model = Earth(verbose=vibrose, feature_importance_type='nb_subsets',
                  penalty=penalty)
    model.fit(data[ind_var], data[dep_var])
    print(model.summary())
    data['Predicted'] = model.predict(data[dep_var])
    fig = plt.figure(figsize=[20,10])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.scatter(data[ind_var], data['Predicted'])
    ax2.scatter(data[ind_var], data[dep_var])
    plt.savefig(path+"/models/.jpg")
    mape = np.mean(np.abs(data[dep_var]-data['Predicted'])/data[dep_var])
    print(mape)



def mars_full_model(data, penalty, vibrose, drop_vars, dep_var):
    model = Earth(verbose=vibrose, feature_importance_type='nb_subsets',
                  penalty=penalty)
    model.fit(data.drop(drop_vars, axis=1, inplace=False), data[dep_var])
