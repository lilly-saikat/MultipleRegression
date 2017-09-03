# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:21:16 2017

@author: Saikat
"""

# =============================================================================
#  Importing Modules
# =============================================================================
import seaborn as sns
import matplotlib.pyplot as plt
import os


# =============================================================================
#  Define Location
# =============================================================================
path = os.getcwd()


# =============================================================================
#  Describe data
# =============================================================================

# Which variables
def which_variables(data):
    return data.info()


def describe_data(data):
    data.describe().to_csv(path+"/data/interim/data_describe.csv", header=True)
    return data.describe()


def pairplot(data):
    sns.pairplot(data)
    plt.savefig(path+"/reports/figures/pairplot.jpg")


def jointplot(data, var1, var2, plot):
    sns.jointplot(data[var1], data[var2], kind=plot)
    plt.savefig(path+"/reports/figures/jointplot.jpg")
