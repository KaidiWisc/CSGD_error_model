# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:11:19 2022

@author: Kaidi Peng



"""

import os
import numpy as np
from netCDF4 import Dataset,date2num,num2date
import matplotlib.pyplot as plt
from datetime import date,datetime, timedelta
from csgd_fitting_codes import fitcsgd_clim,fit_regression
from csgd_draw_prob import draw_pdf,draw_cdf,draw_scatter

# Precip reference from GPM-2BCMB
BCMBds = Dataset("/slow/kpeng43/Gauge/CSGD_CONUS/MRMSCMB.DPRV072017.1.halfhourly.nc")
# gridded precip from IMERG V07-Early
IMERGds = Dataset("/slow/kpeng43/Gauge/CSGD_CONUS/MRMS.IMERGE72017.halfhourly.nc")


lats=IMERGds["latitude"][:]
lons=IMERGds["longitude"][:]
ysize=len(lats)
xsize=len(lons)
ts=len(IMERGds["time"][:])

# choose to linear or non-linear model. 
# It is recommanded to try both (lin=False and lin=True), and then select the better one 
lin=False

# window size
w = 10 

#==========================================================================
# to speed up, here is a parallel training for each 1deg latitude
def train_CSGD(latid):
    
    
    paras=np.zeros((w,xsize,12))

    # set one covariate, if you do not have covariates [False, False, False]
    covarsID=[False, False, False]
    
    
    print("start training")
    

    y=latid*w  

    for x in np.arange(0,xsize,w):


        y1 = y # starting y index of subset window
        y2 = min(y+w,ysize) # ending y index of subset window
        x1 = x # starting x index of subset window
        x2 = min(x+w,xsize) # ending x index of subset window
        
        #------------------------------------------------------------------------
        imerg=IMERGds.variables['prcp'][y1:y2,x1:x2,:].astype(np.float32)
        # set a rain-or-not threshold, 0.1 is usually used for IMERG
        imerg[imerg<0]=np.nan
        imerg[imerg<0.1]=0
        imerg=np.round(imerg,2)  
        
        pcmb=BCMBds.variables['2BCMB'][y1:y2,x1:x2,:].astype(np.float32)
        pcmb[pcmb<0]=np.nan
        pcmb[pcmb<0.1] = 0.
        pcmb=np.round(pcmb,2) 
      
        
        liqprcp=IMERGds.variables['prabliquidprcp'][y1:y2,x1:x2,:].astype(np.float32)     

        pcmbsubset = pcmb.flatten()
        imergsubset = imerg.flatten()
        liqprcpsubset=liqprcp.flatten()
        
        # select good samples
        imerg_ = imergsubset[(np.isnan(imergsubset)==False) & (np.isnan(pcmbsubset)==False) & (pcmbsubset<300.)  & (liqprcpsubset==100)]
        pcmb_  = pcmbsubset[(np.isnan(imergsubset)==False) & (np.isnan(pcmbsubset)==False)&(pcmbsubset<300.) & (liqprcpsubset==100)]

        # store # of samples and hit ratio.
        paras[:,x1:x2,10]=len(pcmb_)
        paras[:,x1:x2,11]=np.sum((imerg_>0) & (pcmb_>0))/len(imerg_)
        
        
        # train climatological CSGD using reference only
        clim_params = fitcsgd_clim(pcmb_,constrained=True)[0].x
        
        # store in parameter array
        paras[:,x1:x2,0] = clim_params[0]
        paras[:,x1:x2,1] = clim_params[1]
        paras[:,x1:x2,2] = clim_params[2]

        reg_covar = fit_regression(imerg_, pcmb_, clim_params,linear = lin,
                                   include_covariates=False,  covars=False, whichcovars=covarsID)
        
        
        # store in parameter array
        paras[:,x1:x2,3] = reg_covar.x[0]
        paras[:,x1:x2,4] = reg_covar.x[1]
        paras[:,x1:x2,5] = reg_covar.x[2]
        paras[:,x1:x2,6] = reg_covar.x[3]
        paras[:,x1:x2,7] = reg_covar.x[4]
        
        
        paras[:,x1:x2,8] = np.nanmean(imerg_)  
        # paras[:,x1:x2,9] = np.nanmean(WAR_) # this is to store covariate but we do nothave here.
    
        # Visilization of error models    
        # darw the pdf/cdf of precipitation uncertainty giving satellite observation of 5 mm/hr
        draw_pdf(paras[0,x1,:],imergobs=5,include_covariates=False,covobs=False,linear=False)

        draw_cdf(paras[0,x1,:],imergobs=5,include_covariates=False,covobs=False,linear=False)

        draw_scatter(paras[0,x1,:],imerg_ ,pcmb_  ,covariate=False ,linear=False,include_covariates=False)    
        
    # temporaly save the par files,
    np.save("pars%d"%(latid),paras)

#%%

import multiprocessing as mp
import os

if __name__ =='__main__':
    
    pool = mp.Pool(20)  
    operation= pool.map(train_CSGD, range(35))  # total 35 degs in latitude
    pool.close()
    pool.join()

    if lin==True:
        model="L"
    else:
        model="NLWAR"

    fname = "csgd%s_DPR.nc"%(model)


    print('Writing %s \n'%fname)

    new_cdf = Dataset(fname, 'w', format = "NETCDF4", clobber=True)

    # create dimensions
    new_cdf.createDimension('latitude', ysize)
    new_cdf.createDimension('longitude', xsize)


    # add lat, and lon variables
    latitude = new_cdf.createVariable('latitude', 'f4', ('latitude'), zlib=True)
    latitude.units = 'degrees_north'
    latitude.long_name = 'latitude'
    latitude[:] = lats

    longitude = new_cdf.createVariable('longitude', 'f4', ('longitude'), zlib=True)
    longitude.units = 'degrees_east'
    longitude.long_name = 'longitude'
    longitude[:] = lons


    climpar1 = new_cdf.createVariable('clim1', 'f4', ('latitude','longitude'), zlib=True)
    climpar1.units = '--'
    climpar1.long_name = 'Climatological mu'

    climpar2 = new_cdf.createVariable('clim2', 'f4', ('latitude','longitude'), zlib=True)
    climpar2.units = '--'
    climpar2.long_name = 'Climatological sigma'
    # climpar2[:,:] = pars[1,:,:]

    climpar3 = new_cdf.createVariable('clim3', 'f4', ('latitude','longitude'), zlib=True)
    climpar3.units = '--'
    climpar3.long_name = 'Climatological delta'
    # climpar3[:,:] = pars[2,:,:]

    par1 = new_cdf.createVariable('par1', 'f4', ('latitude','longitude'), zlib=True)
    par1.units = '--'
    par1.long_name = 'Regression par 1'
    # par1[:,:] = pars[3,:,:]

    par2 = new_cdf.createVariable('par2', 'f4', ('latitude','longitude'), zlib=True)
    par2.units = '--'
    par2.long_name = 'Regression par 2'
    # par2[:,:] = pars[4,:,:]

    par3 = new_cdf.createVariable('par3', 'f4', ('latitude','longitude'), zlib=True)
    par3.units = '--'
    par3.long_name = 'Regression par 3'
    # par3[:,:] = pars[5,:,:]

    par4 = new_cdf.createVariable('par4', 'f4', ('latitude','longitude'), zlib=True)
    par4.units = '--'
    par4.long_name = 'Regression par 4'
    # par4[:,:] = pars[6,:,:]

    par5 = new_cdf.createVariable('par5', 'f4', ('latitude','longitude'), zlib=True)
    par5.units = '--'
    par5.long_name = 'Covariate WAR regression par 1'
    # par5[:,:] = pars[7,:,:]

    imean = new_cdf.createVariable('mean', 'f4', ('latitude','longitude'), zlib=True)
    imean.units = 'mm/hr'
    imean.long_name = 'IMERG mean'
    # imean[:,:] = pars[8,:,:]

    WARmean = new_cdf.createVariable('WARmean', 'f4', ('latitude','longitude'), zlib=True)
    WARmean.units = '--'
    WARmean.long_name = 'WAR mean'
    # WARmean[:,:] = pars[9,:,:]

    size = new_cdf.createVariable('Dsize', 'f4', ('latitude','longitude'), zlib=True)
    size.units = '--'
    size.long_name = 'coincident data number'
    # size[:,:]

    hitf = new_cdf.createVariable('hitfrac', 'f4', ('latitude','longitude'), zlib=True)
    hitf.units = '%'
    hitf.long_name = 'hit fraction'
    
    for ii in range(35):
        pp=np.load("pars%d.npy"%(ii))
        climpar1[ii*10:(ii+1)*10,:] = pp[:,:,0]
        climpar2[ii*10:(ii+1)*10,:] = pp[:,:,1]
        climpar3[ii*10:(ii+1)*10,:] = pp[:,:,2]
        par1[ii*10:(ii+1)*10,:]=pp[:,:,3]
        par2[ii*10:(ii+1)*10,:]=pp[:,:,4]
        par3[ii*10:(ii+1)*10,:]=pp[:,:,5]
        par4[ii*10:(ii+1)*10,:]=pp[:,:,6]
        par5[ii*10:(ii+1)*10,:]=pp[:,:,7]
        imean[ii*10:(ii+1)*10,:]=pp[:,:,8]
        WARmean[ii*10:(ii+1)*10,:]=pp[:,:,9]
        size[ii*10:(ii+1)*10,:]=pp[:,:,10]
        hitf[ii*10:(ii+1)*10,:]=pp[:,:,11]
        
        os.remove("pars%d.npy"%(ii))
        
    new_cdf.close()

