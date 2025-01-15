# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:33:25 2022

@author: Kaidi Peng

Write 30-min DPR V07 for specific time and region
Attention: 
1. DPR data has missing files, repleced by NAN
2. Clip based on latitude and longitude: Due to the different rule that assign values to each grid, you may need look at lines  59-62
3. I follow the last version dataset. '/KuGMI/estimSurfPrecipTotRate'  is used for 2BCMB, 
3.  '/FS/SLV/precipRateESurface' is used for 2ADPR
"""

import os
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta, date
import numpy as np
import h5py
import cv2

#==============================================================================

def get_dates(start,end,res):
    # start date, end date, and resolution of time step in minutes
    
    startdate = datetime(start.year,start.month,start.day,0,0)
    dates = [startdate]
    
    n=0
    while (dates[n] - end).total_seconds() < 0:
        
        n+=1
        nextday = startdate + timedelta(seconds=n*res*60)
        
        dates.append(nextday)
    
    return(dates)

#==============================================================================

# function to write daily IMERG data to a netcdf over Rio study area for a give date range
def write_BCMB_Hhour(tRange,latstart,latend,lonstart,lonend):
    
    print("Creating Hhourly DPRV07 file for period %s to %s."%(str(tRange[0]),str(tRange[1])))
        

    fileName = "/slow/DPRV07/data/DPR-CMB-7A.GLOB.20200401-S010000-E012959.HDF5"
    f = h5py.File(fileName, 'r')
    g = f['Comb']
    lat = g['lat'][:]
    lon = g['lon'][:]

    # identify indices that mark all corners of the study area
    # note that in CMB files, lat lon are lower corner of the grid
    lati0 = np.where(np.isclose(lat, latstart))[1][0]+1  
    lati1 = np.where(np.isclose(lat, latend))[1][0]+1  #  
    loni0 = np.where(np.isclose(lon, lonstart))[1][0]  
    loni1 = np.where(np.isclose(lon, lonend))[1][0]   #

    
    ysize = abs(lati0-lati1)
    xsize = abs(loni0-loni1)
    
    dates = get_dates(tRange[0],tRange[1],30)  #half hourly data, output date every half hour
    
    print(len(dates))
    
    DPR_CMB_agg= np.zeros((ysize,xsize,len(dates)))
    DPR_DPR_agg = np.zeros((ysize,xsize,len(dates)))
    
    # initialize counter for this loop
    n=0
    # cycle through all dates in date range and retrieve IMERG half hourly data
    for d in dates:
        
        
        
        if d.minute == 0:
            fileName = "/slow/DPRV07/data/DPR-CMB-7A.GLOB.%s-S%.02d%.02d00-E%.02d2959.HDF5"%(d.strftime('%Y%m%d'),
                                                                         d.hour,d.minute,d.hour)
        else:
            fileName = "/slow/DPRV07/data/DPR-CMB-7A.GLOB.%s-S%.02d%.02d00-E%.02d5959.HDF5"%(d.strftime('%Y%m%d'),
                                                                         d.hour,d.minute,d.hour)
        try:
            f = h5py.File(fileName, 'r')

        except:           
            DPR_DPR_agg[:,:,n] = np.nan
            DPR_CMB_agg[:,:,n] = np.nan
            print(d) # print missing files!!!

        else:
            g = f['Comb']
            DPR_DPR = g['2ADPR'][loni0:loni1,lati0:lati1]
            DPR_CMB = g['2BCMB'][loni0:loni1,lati0:lati1]
            
            # transpose array so that lats line up with rows and lons line up with columns
            DPR_DPR = np.transpose(DPR_DPR)
            DPR_CMB = np.transpose(DPR_CMB)

            # flip array vertically so that latitudes decrease from top to bottom
            DPR_DPR_agg[:,:,n] = DPR_DPR
            DPR_CMB_agg[:,:,n] = DPR_CMB
            
            
        finally:
            n+=1
            
    
    
    # ========================================================================
    # ========================= SAVE TO NETCDF FILE ==========================
 
    new_file ="/slow/kpeng43/Gauge/CSGD_CONUS/%s.DPRV07%d.%d.halfhourly.nc"%(ident,tRange[0].year,tRange[0].month)

    
    # create array of time stamps
    time_days = [datetime(tRange[0].year,tRange[0].month,tRange[0].day,0,0)+n*timedelta(seconds=30*60) for n in range(len(dates))]
    units = 'hours since 1970-01-01 00:00:00 UTC'
        
    new_cdf = Dataset(new_file, 'w', format = "NETCDF4", clobber=True)
    
    # create dimensions
    new_cdf.createDimension('latitude', ysize)
    new_cdf.createDimension('longitude', xsize)
    new_cdf.createDimension('time', len(time_days))
    
    # write time stamps to variable
    time = new_cdf.createVariable('time','d', ('time'))
    time.units = units
    time[:] = date2num(time_days,units,calendar="gregorian")
    
    # add lat, and lon variables
    latitude = new_cdf.createVariable('latitude', 'f4', ('latitude'), zlib=True,least_significant_digit=3)
    latitude.units = 'degrees_north'
    latitude.long_name = 'latitude'
    latitude[:] = np.arange(latstart-0.05,latend,-0.1)

    
    longitude = new_cdf.createVariable('longitude', 'f4', ('longitude'), zlib=True,least_significant_digit=3)
    longitude.units = 'degrees_east'
    longitude.long_name = 'longitude'
    longitude[:] = np.arange(lonstart+0.05,lonend,0.1)

    
    ADPR = new_cdf.createVariable('2ADPR', 'f4', ('latitude','longitude','time'), zlib=True,least_significant_digit=3)
    ADPR.units = 'mm/hr'
    ADPR.long_name = 'DPR-2ADPR' 
    ADPR[:,:,:] = DPR_DPR_agg
    
    

    BCMB = new_cdf.createVariable('2BCMB', 'f4', ('latitude','longitude','time'), zlib=True,least_significant_digit=3)
    BCMB.units = 'mm/hr'
    BCMB.long_name = 'DPR-2BCMB'
    BCMB[:,:,:] = DPR_CMB_agg
    
    new_cdf.close()

#==============================================================================


if __name__ == "__main__":
    
    timeRange = (datetime(2017,1,1),datetime(2021,12,31,23,30))
    
    write_BCMB_Hhour(timeRange,55,20,-130,-60)
    