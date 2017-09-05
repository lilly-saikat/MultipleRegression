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
# from pyearth import Earth
# import matplotlib.pyplot as plt
# import numpy as np
# =============================================================================
#  Define Location;
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
