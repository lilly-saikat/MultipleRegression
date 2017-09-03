# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:21:16 2017

@author: Saikat
"""

# =============================================================================
#  Importing Modules
# =============================================================================
import statsmodels.datasets as dt
import os


# =============================================================================
#  Define Location
# =============================================================================
path = os.getcwd()


# =============================================================================
#  Import data
# =============================================================================


def pull_data_function():
    print("Initiating data pull...")
    data = dt.get_rdataset("Boston", "MASS").data
    print("Data pull complete...")
    print()
    print("Saving...")
    data.to_csv(path+"/data/raw/input_raw.csv", header=True)
    print("Save complete!")
    return data
