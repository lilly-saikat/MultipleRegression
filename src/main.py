# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:26:21 2017

@author: Saikat
"""

# =============================================================================
# Importing modules
# =============================================================================
# import data extract module
from data.extract_data import pull_data_function
# imoprt data exploration module
from data.explore_data import which_variables, describe_data
# Import data distribution modules
from data.explore_data import pairplot, jointplot
# Import data treatment modules
from data.data_treatment import missing_treat, outlier_tret, dummy
# Import boosting module
from models.boosting import bootstrap_it, k_fold_ols, hyperparameter_linear
# Import basic models module
from models.basic_models import ols_model, regularized_model
# Import Tree Models
from models.tree_models import DTree
# Import validation modues
from models.validation import mape
# =============================================================================
# Step 1: Import data
# =============================================================================
input_raw = pull_data_function()

# =============================================================================
# Step 2: Explore Data
# =============================================================================

datatype = which_variables(input_raw)

datadesc = describe_data(input_raw)

# pairplot(input_raw)

jointplot(input_raw, 'rm', 'medv', 'scatter')

# =============================================================================
# Step 3: Data Treatment
# =============================================================================

# missing value treatment : complete data
data_treat_miss = missing_treat(input_raw)

# outlier treatment : One variable at a time
outlier_data = outlier_tret(data_treat_miss, 'medv', 0.01, 0.95)

# create dummy variable : One variable at a time
dummy_data = dummy(outlier_data, 'chas')

# Standardize data

# =============================================================================
# Step 4: Data Boosting
# =============================================================================

# train_data, test_data = bootstrap_it(dummy_data, 0.8)

k_fold_ols(dummy_data, 10, 'medv', 'medv')

hyperparameter_linear(dummy_data, 10, 'medv', 'medv')
# =============================================================================
# Step 5: Data Modelling
# =============================================================================

# train_data, test_data = bootstrap_it(dummy_data, 0.8)

# ols_model = ols_model(train_data, 'medv',
#                        ['medv', 'crim', 'chas_0', 'chas_1'])

# reg_model = regularized_model(train_data, 'medv', 'medv')

# detree_model = DTree(train_data, 'medv', ['medv'], 2)

# =============================================================================
# Step 6: Validation
# =============================================================================
# mape = mape(detree_model, test_data, 'medv', ['medv'])
