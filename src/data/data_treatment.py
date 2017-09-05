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
import pandas as po
# For Standardization
from scipy.stats.mstats import zscore


# =============================================================================
#  Define Location
# =============================================================================
path = os.getcwd()


# =============================================================================
#  Observe missing data
# =============================================================================

def missing_treat(data):
    data.fillna(data.mean())
    return data

# =============================================================================
#  Observe outliers data
# =============================================================================


def outlier_tret(data, in_var, low_margin, high_margin):
    p = data[in_var].quantile([low_margin, high_margin])
    data['temp'] = np.where(data[in_var] > p[high_margin],
                            p[high_margin], data[in_var])
    data['temp_new'] = np.where(data['temp'] < p[low_margin],
                                p[low_margin], data['temp'])
    new_data = data.drop(['temp', in_var], axis=1)
    new_data.rename(columns={'temp_new': in_var}, inplace=True)
    return new_data


# =============================================================================
#  Create Dummy Variable
# =============================================================================
def dummy(data, var):
    temp_data = po.concat([data, po.get_dummies(data[var],
                                                prefix=var)], axis=1)
    new_data = temp_data.drop(var, axis=1)
    return new_data


# =============================================================================
# Create Standardization data
# =============================================================================
def stadardize(data, drop_vars, drop):
    if drop == 1:
        temp_data = data.drop(drop_vars, axis=1, inplace = False)
        po.DataFram`e(zscore(temp_data), columns=temp_data.columns)
    else:
        po.DataFrame(zscore(data), columns=data.columns)
    return data
