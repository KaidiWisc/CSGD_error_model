# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:11:19 2022

@author: Kaidi Peng

CSGD error model
Main function
Description:
     "imerg" is the precipitation time-series to be corrected
     "gtruth" is the ground truth precipitation time-series to correct imerg
     "Covs" is one Covariate time-series, WAR used here.

Reference:

    Wright, D. B., Kirschbaum, D. B., & Yatheendradas, S. (2017). 
    Satellite Precipitation Characterization, Error Modeling, 
    and Error Correction Using Censored Shifted Gamma Distributions. 
    Journal of Hydrometeorology, 18(10), 2801-2815. 
    https://doi.org/10.1175/JHM-D-17-0060.1

"""

import os
import numpy as np
from netCDF4 import Dataset,date2num,num2date
import matplotlib.pyplot as plt
from datetime import date,datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

# additional python functions
from csgd_fitting_codes import fitcsgd_clim,fit_regression
from csgd_draw_prob import draw_pdf,draw_cdf,draw_scatter


#%% model Training =========================================
    
imerg=np.array(pd.read_csv("imerg.hourly.csv"))
gtruth=np.array(pd.read_csv("gauge.hourly.csv"))
covs=np.array(pd.read_csv("cov.hourly.csv"))
       
imerg_ = imerg[(np.isnan(imerg)==False) & (np.isnan(gtruth)==False) &  (np.isnan(covs)==False) ]
gtruth_  = gtruth[(np.isnan(imerg)==False) & (np.isnan(gtruth)==False) & (np.isnan(covs)==False) ]
covs_  = covs[(np.isnan(imerg)==False) & (np.isnan(gtruth)==False) & (np.isnan(covs)==False) ]

# define pars to store parameters
pars=np.zeros(10)

# First Step: train climatological CSGD using ground truth data
clim_params = fitcsgd_clim(gtruth_,constrained=True)[0].x

# store in parameter array
pars[0] = clim_params[0]  # clim_mu
pars[1] = clim_params[1]  # clim_sigma
pars[2] = clim_params[2] # clim_delta

# Second Step: train regression CSGD use both ground truth, imerg and one covariate
reg_covar = fit_regression(imerg_, gtruth_, clim_params, linear = False,
                           include_covariates=True,  covars=covs_ )

# store in parameter array
pars[3] = reg_covar[0].x[0]  # alpha1
pars[4] = reg_covar[0].x[1]  # alpha2
pars[5] = reg_covar[0].x[2]  # alpha3
pars[6] = reg_covar[0].x[3]  # alpha4
pars[7] = reg_covar[0].x[4]  # alpha5
 
# store imerg mean & covariate mean
pars[8] = reg_covar[1]
pars[9] = reg_covar[2]

print(pars)
# Congrats! you get all need in the CSGD error model

#%% model evaluatiom =========================================

# draw pdf of the CSGD model, given imerg and covs
imergobs=5 # mm/hr
covobs=0.5

draw_pdf(pars,imergobs,include_covariates=True,covobs=covobs,linear=False)

draw_cdf(pars,imergobs,include_covariates=True,covobs=covobs,linear=False)

draw_scatter(pars,imerg_ ,gtruth_ ,covariate=covs_ ,linear=False,include_covariates=True)    
  