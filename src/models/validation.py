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
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  Define Location
# =============================================================================
path = os.getcwd()

# =============================================================================
# Create training and testing data
# =============================================================================


def mape(model, test_data, dep_var, drop_vars):
    test_data['Predicted'] = model.predict(test_data.drop(drop_vars,
                                           axis=1, inplace=False))
    mape = np.mean(np.abs(test_data[dep_var] -
                          test_data['Predicted'])/test_data[dep_var])
    print(mape)
    plt.figure(); test_data[dep_var].plot(); test_data['Predicted'].plot(); plt.legend(loc='best')
    plt.savefig(path+"/models/pred_vs_actual[model].jpg")
    return mape
