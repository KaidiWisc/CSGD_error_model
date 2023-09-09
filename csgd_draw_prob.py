# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:10:46 2022

@author: Kaidi Peng
"""

import os
from netCDF4 import Dataset
from datetime import date, timedelta
import numpy as np
from csgd_fitting_codes import fitcsgd_clim,fit_regression,GenerateSpecQuant,CondDistr,qcsgd,dcsgd,pcsgd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc,cm
import sys
import scipy as sp


def draw_pdf(all_para,imergobs,include_covariates=False,covobs=False,linear=False):
    
    # all_para: 'clim1','clim2','clim3','par1','par2','par3','par4','par5','mean','covmean'

    #------calculate parameters------------------------ 
    CondParams = CondDistr(imergobs,all_para[8],all_para[0:3],all_para[3:8],linear,
                           include_covariates,covobs,covmean = all_para[9])
    # mu,sigma,delta
    median=qcsgd(0.5,CondParams)
    
    # plot PDF of conditional CSGD, also indicate IMERG estimate 
    qs = np.arange(0.01,1.0,0.001)
    # this function generates precipitation values at 
    prcp_ = GenerateSpecQuant(CondParams, qs)  
       
    # this ensures that we have enough values left of the x-axis represented in our plot
    negprcp = None
    if prcp_[0]==0.:
        negprcp = np.arange(-0.5,0.,0.5/len(prcp_[prcp_==0.]))
        prcp_[:len(prcp_[prcp_==0.])] = negprcp[:len(prcp_[prcp_==0.])]
    
    # now we retrieve the probability density function of this conditional CSGD for every value of precipitation
    pdf_ = np.zeros(len(qs))
    n=0
    for p in prcp_:
        pdf_[n] = dcsgd(p,CondParams)
        n+=1
        
    #--draw figure-------------------------------------------
    
    
    fig = plt.subplots(figsize=(10,6))
    ax1 = plt.subplot(1, 1, 1)
    
    
    ax1.plot(prcp_,pdf_,c='r')
    
    ax1.plot((0.,0.),(0.,np.max(pdf_)),linestyle="dashed",c='k')  # ?
    
    ax1.fill_between(prcp_[prcp_<=0.],np.zeros(len(prcp_[prcp_<=0.])),pdf_[prcp_<=0.],color='yellow',alpha=0.4,label="P(R=0) = %.03f"%pcsgd(0.,CondParams))
    
    ax1.plot((imergobs,imergobs),(0.,np.max(pdf_)),linestyle="dashed",c='green',label='IMERG observation')
    
    ax1.plot((CondParams[0],CondParams[0]),(0.,np.max(pdf_)),linestyle="dashed",c='m',label='mean')
    
    ax1.plot((median,median),(0.,np.max(pdf_)),linestyle="dashed",c='blue',label='median')
    
    ax1.set_xlim((-0.5,np.max((1.0,1.5*imergobs))))
    
    if include_covariates ==False:
         ax1.set_title('CSGD with precip. = %0.2f mm/h'%(imergobs),fontsize=18)  # WAR = %.02f  ,WAR
    else:
         ax1.set_title('CSGD with precip. = %0.2f mm/h & cov = %0.2f'%(imergobs, covobs),fontsize=18)  # WAR = %.02f  ,WAR

    ax1.xaxis.set_tick_params(labelsize=15);ax1.yaxis.set_tick_params(labelsize=15)
    ax1.set_xlabel('Precip [mm/hr]',fontsize=15)
    ax1.set_ylabel('PDF',fontsize=15)
    ax1.legend(fontsize=17)
    
    # ax1.set_xlim(-0.2,12)
    # ax1.set_ylim(-0.02,2.5)
    plt.savefig("pdf.jpg",dpi=200)
    # plt.show()


def draw_cdf(all_para,imergobs,include_covariates=False,covobs=False,linear=False):
  
    # all_para: 'clim1','clim2','clim3','par1','par2','par3','par4','par5','mean', 'covmean'
    
    CondParams = CondDistr(imergobs,all_para[8],all_para[0:3],all_para[3:8],linear=False,
                               include_covariates=True,covobs=covobs,covmean = all_para[9])
     # mu,sigma,delta
    
    prcp_cdf=pcsgd(np.arange(0.,15,0.1),CondParams)
        
    
    fig = plt.subplots(figsize=(10,6))
    ax1 = plt.subplot(1, 1, 1)
    
    ax1.plot(np.arange(0.,15,0.1),prcp_cdf,c='r',label="CSGD")
    # ax1.plot(np.arange(0.1,15,0.1),prcp_stage,c='b',label='NLCSGD trained by Stage IV')
    
    ax1.plot((imergobs,imergobs),(0.,1),linestyle="dashed",c='k',label='IMERG observation')

    if include_covariates ==False:
         ax1.set_title('CSGD with precip. = %0.2f mm/h'%(imergobs),fontsize=18)  # WAR = %.02f  ,WAR
    else:
         ax1.set_title('CSGD with precip. = %0.2f mm/h & cov = %0.2f'%(imergobs, covobs),fontsize=18)  

    ax1.xaxis.set_tick_params(labelsize=15);ax1.yaxis.set_tick_params(labelsize=15)

    ax1.set_xlabel('Precip [mm/hr]',fontsize=15)
    ax1.set_ylabel('CDF',fontsize=15)
    ax1.legend(fontsize=10)
    # ax1.set_ylim(0.97,1.001)
    plt.grid("True")
    plt.savefig("cdf.jpg",dpi=200)
    # plt.show()
#%%===============================================================================

def draw_scatter (all_para,imergset,gtset,covariate=False, linear=False,
                  include_covariates=False)  :
        
        
    trainIMERG = imergset[(np.isnan(imergset)==False)& (np.isnan(gtset)==False)]
    trainGT = gtset[(np.isnan(imergset)==False)& (np.isnan(gtset)==False)]
    
    
    if include_covariates == True:
        trainCOV = covariate[(np.isnan(imergset)==False) & (np.isnan(gtset)==False)]
        covmean=all_para[9]


    
    climPars =all_para[0:3]
    regPars = all_para[3:8]
    
    rainmax     =40.   # maximum rain rate to plot Cond. CSGD to [mm/hr]
    step        = 0.1    # step to use between rain rate values when plotting Cond. CSGD
    xylimit     = 40       # x, y axis limit for scatterplots for Cond. CSGD
    
    # arrays to hold confidence intervals and means of trained CSGD error models
    lens=len(np.arange(0,rainmax,step))
    dis_mid   = np.zeros((5,lens)) # for median of each Cond. CSGD
    dis_low  = np.zeros((5,lens)) 
    dis_high = np.zeros((5,lens)) 
    dis_mean = np.zeros((5,lens))  # for mean of each Cond. CSGD
    dis_sigma = np.zeros((5,lens))  # for sigma parameter of each Cond. CSGD
    
    
    # Now iterate through each rain rate value to generate a Cond. CSGD and grab confidence interval, mean, etc.
    k=0
    for covobs in [0.0,0.3,0.6,0.9]:
        
        ensctr=0
        for j in np.arange(0.,rainmax,step):
           
            # Nonlinear Model
            CondParams_c1nl = CondDistr(j, all_para[8], climPars, regPars, linear,include_covariates,covobs,covmean)
            dis_low[k,ensctr], dis_mid[k,ensctr], dis_high[k,ensctr] = GenerateSpecQuant(CondParams_c1nl, [0.025,0.5,0.975])
                
            dis_mean[k,ensctr] = CondParams_c1nl[0]
            dis_sigma[k,ensctr] = CondParams_c1nl[1]
                
            ensctr+=1
        k+=1
          
    #==================================================
    fig = plt.figure(figsize=(9, 8),dpi=200)
    ax1 = plt.subplot(111)
    
    cmap = plt.get_cmap('viridis')
    ps_train = ax1.scatter(trainIMERG, trainGT, s=50, marker='o', c=trainCOV ,cmap=cmap,alpha=0.6)
    plt.colorbar(ps_train)

    k=0
    
    for covobs in [0.0,0.3,0.6,0.9]:
        
        lnl = ax1.plot(np.arange(0,rainmax,step),dis_low[k,:],color=cmap(covobs), ls='--',linewidth=1.5)
        hnl = ax1.plot(np.arange(0,rainmax,step),dis_high[k,:],color=cmap(covobs), ls='--',linewidth=1.5,label="q95 WAR =%0.2f "%(covobs))
        mdnl = ax1.plot(np.arange(0,rainmax,step),dis_mid[k,:],color=cmap(covobs),linewidth=1.5,label="median WAR =%0.2f "%(covobs))
        k+=1
        
    ax1.set_xlim(0.1,xylimit)
    ax1.set_ylim(0.1,xylimit)
    ax1.set_autoscale_on(False)
    ax1.plot([0, 100],[0, 100],c='black', linewidth=1, alpha=0.8)
    ax1.set_xlabel("$IMERG$ $Obs.$ $[mm]$", fontsize=17, labelpad=3)
    ax1.xaxis.set_label_coords(0.5, -0.04, transform=ax1.transAxes)
    ax1.set_ylabel("GT $Obs.$ $[mm]$", fontsize=17, labelpad=18)
    ax1.yaxis.set_label_coords(-0.04, 0.5, transform=ax1.transAxes)
    plt.legend()
    # plt.show()
    plt.savefig("scatter.jpg",dpi=200)