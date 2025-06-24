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

# window size, one CSGD model in trained over w*w
w = 10 

    
paras=np.zeros((ysize,xsize,12))

# set one covariate, if you do not have covariates [False, False, False]
covarsID=[False, False, False]


print("start training")


for x in np.arange(0,xsize,w):
    for y in np.arange(0,ysize,w):

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
        paras[y1:y2,x1:x2,10]=len(pcmb_)
        paras[y1:y2,x1:x2,11]=np.sum((imerg_>0) & (pcmb_>0))/len(imerg_)
        
        
        # train climatological CSGD using reference only
        clim_params = fitcsgd_clim(pcmb_,constrained=True)[0].x
        
        # store in parameter array
        paras[y1:y2,x1:x2,0] = clim_params[0]
        paras[y1:y2,x1:x2,1] = clim_params[1]
        paras[y1:y2,x1:x2,2] = clim_params[2]
    
        reg_covar = fit_regression(imerg_, pcmb_, clim_params,linear = lin,
                                   include_covariates=False,  covars=False, whichcovars=covarsID)
        
        
        # store in parameter array
        paras[y1:y2,x1:x2,3] = reg_covar.x[0]
        paras[y1:y2,x1:x2,4] = reg_covar.x[1]
        paras[y1:y2,x1:x2,5] = reg_covar.x[2]
        paras[y1:y2,x1:x2,6] = reg_covar.x[3]
        paras[y1:y2,x1:x2,7] = reg_covar.x[4]
        
    
        paras[y1:y2,x1:x2,8] = np.nanmean(imerg_)  
        # paras[:,x1:x2,9] = np.nanmean(WAR_) # this is to store covariate but we do nothave here.
    
        # Visilization of error models    
        # darw the pdf/cdf of precipitation uncertainty giving satellite observation of 5 mm/hr
        draw_pdf(paras[y1,x1,:],imergobs=5,include_covariates=False,covobs=False,linear=False)
    
        draw_cdf(paras[y1,x1,:],imergobs=5,include_covariates=False,covobs=False,linear=False)
    
        draw_scatter(paras[y1,x1,:],imerg_ ,pcmb_  ,covariate=False ,linear=False,include_covariates=False)    
    

if lin==True:
    model="L"
else:
    model="NL"

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
climpar1[:,:] = paras[:,:,0]

climpar2 = new_cdf.createVariable('clim2', 'f4', ('latitude','longitude'), zlib=True)
climpar2.units = '--'
climpar2.long_name = 'Climatological sigma'
climpar2[:,:] = paras[:,:,1]

climpar3 = new_cdf.createVariable('clim3', 'f4', ('latitude','longitude'), zlib=True)
climpar3.units = '--'
climpar3.long_name = 'Climatological delta'
climpar3[:,:] = paras[:,:,2]

par1 = new_cdf.createVariable('par1', 'f4', ('latitude','longitude'), zlib=True)
par1.units = '--'
par1.long_name = 'Regression par 1'
par1[:,:] = paras[:,:,3]

par2 = new_cdf.createVariable('par2', 'f4', ('latitude','longitude'), zlib=True)
par2.units = '--'
par2.long_name = 'Regression par 2'
par2[:,:] = paras[:,:,4]

par3 = new_cdf.createVariable('par3', 'f4', ('latitude','longitude'), zlib=True)
par3.units = '--'
par3.long_name = 'Regression par 3'
par3[:,:] = paras[:,:,5]

par4 = new_cdf.createVariable('par4', 'f4', ('latitude','longitude'), zlib=True)
par4.units = '--'
par4.long_name = 'Regression par 4'
par4[:,:] = paras[:,:,6]

par5 = new_cdf.createVariable('par5', 'f4', ('latitude','longitude'), zlib=True)
par5.units = '--'
par5.long_name = 'Covariate WAR regression par 1'
par5[:,:] = paras[:,:,7]

imean = new_cdf.createVariable('mean', 'f4', ('latitude','longitude'), zlib=True)
imean.units = 'mm/hr'
imean.long_name = 'IMERG mean'
imean[:,:] = paras[:,:,8]

# one more to store covariates' mean
# WARmean = new_cdf.createVariable('WARmean', 'f4', ('latitude','longitude'), zlib=True)
# WARmean.units = '--'
# WARmean.long_name = 'WAR mean'
# WARmean[:,:] = pars[:,:,9]

size = new_cdf.createVariable('Dsize', 'f4', ('latitude','longitude'), zlib=True)
size.units = '--'
size.long_name = 'coincident data number'
size[:,:]=paras[:,:,10]

hitf = new_cdf.createVariable('hitfrac', 'f4', ('latitude','longitude'), zlib=True)
hitf.units = '%'
hitf.long_name = 'hit fraction'
hitf[:,:]=paras[:,:,11]
    
new_cdf.close()

