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
from sklearn import tree
# For saving model
import pickle

# =============================================================================
#  Define Location
# =============================================================================
path = os.getcwd()

# =============================================================================
# Create training and testing data
# =============================================================================


def DTree(train_data, dep_var, drop_vars, depth):
    # Run the Decision Tree model
    regressor = tree.DecisionTreeRegressor(max_depth=depth)
    model = regressor.fit(train_data.drop(drop_vars, axis=1, inplace=False),
                          train_data[dep_var])
    model_filename = path+'/models/models.p'
    pickle.dump(model, open(model_filename, 'wb'))
    return model
