# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:33:25 2022

@author: Kaidi Peng

Write 30-min IMERG-E (StageIV 0.1) for 2016-2020 to netcdf over 
study area (-100. - -85., 35. - 45.)

"""

import os
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta, date
import numpy as np
import h5py
import sys

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
def write_to_file(ident,tRange,latstart,latend,lonstart,lonend):
    
    print("Creating hourly IMERG-early file for period %s to %s."%(str(tRange[0]),str(tRange[1])))
    
    fileName = "/home/group/IMERG_Early_V07B/GPM_3IMERGHHE.07/2023/001/3B-HHR-E.MS.MRG.3IMERG.20230101-S000000-E002959.0000.V07B.HDF5"
    f = h5py.File(fileName, 'r')
    g = f['Grid']
    lat = g['lat'][:]
    lon = g['lon'][:]

    # identify indices that mark all corners of the study area
    lati0 = np.where(np.isclose(lat, latstart-0.05))[0][0]
    lati1 = np.where(np.isclose(lat, latend+0.05))[0][0]
    loni0 = np.where(np.isclose(lon, lonstart+0.05))[0][0]
    loni1 = np.where(np.isclose(lon, lonend-0.05))[0][0]

    
    ysize = abs(lati0-lati1)+1
    xsize = abs(loni0-loni1)+1
    
    dates = get_dates(tRange[0],tRange[1],30)  #hourly data  ,output date every hour
    
    print(len(dates))
    
    imerg_agg = np.zeros((ysize,xsize,len(dates)), dtype=np.float32)
    imerg_PLP = np.zeros((ysize,xsize,len(dates)), dtype=np.float32)


    n=0
    # cycle through all dates in date range and retrieve IMERG and Stage IV hourly data
    for d in dates:
        # /home/group/IMERG_Early_V07B/GPM_3IMERGHHE.07/2001/002/3B-HHR-E.MS.MRG.3IMERG.20010102-S000000-E002959.0000.V07B.HDF5          
        # open first 30-min IMERG file
        if d.minute == 0:
            fileName = "/home/group/IMERG_Early_V07B/GPM_3IMERGHHE.07/%d/%.03d/3B-HHR-E.MS.MRG.3IMERG.%s-S%.02d0000-E%.02d2959.%.04d.V07B.HDF5"%(d.year,
              d.timetuple().tm_yday, d.strftime('%Y%m%d'), d.hour,d.hour,60*d.hour+d.minute)
   
        else:
            fileName = "/home/group/IMERG_Early_V07B/GPM_3IMERGHHE.07/%d/%.03d/3B-HHR-E.MS.MRG.3IMERG.%s-S%.02d3000-E%.02d5959.%.04d.V07B.HDF5"%(d.year,
              d.timetuple().tm_yday,d.strftime('%Y%m%d'), d.hour,d.hour,60*d.hour+d.minute)
   
        f = h5py.File(fileName, 'r')
            
        data = f['Grid']['precipitation'][0,loni0:loni1+1,lati1:lati0+1]
        PLPdata = f['Grid']['probabilityLiquidPrecipitation'][0,loni0:loni1+1,lati1:lati0+1]
        
        data = np.flip(np.transpose(data),axis=0)
        PLPdata = (np.flip(np.transpose(PLPdata),axis=0)).astype(float)
        
        imerg_agg[:,:,n] = data
        imerg_PLP[:,:,n] = PLPdata
        
        print(d)
        sys.stdout.flush()
   
        n+=1
  
  
    # ========================================================================
    # ========================= SAVE TO NETCDF FILE =========================
        
            
    new_file ="IMERGE7%d.halfhourly.nc"%(tRange[0].year)
        
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
    latitude = new_cdf.createVariable('latitude', 'f4', ('latitude'), zlib=True)
    latitude.units = 'degrees_north'
    latitude.long_name = 'latitude'
    latitude[:] = np.arange(latstart-0.05,latend,-0.1)
    
    longitude = new_cdf.createVariable('longitude', 'f4', ('longitude'), zlib=True)
    longitude.units = 'degrees_east'
    longitude.long_name = 'longitude'
    longitude[:] = np.arange(lonstart+0.05,lonend,0.1)
    
    prcp = new_cdf.createVariable('prcp', 'f4', ('latitude','longitude','time'), zlib=True)
    prcp.units = 'mm/hr'
    prcp.long_name = 'precipitationCal'
    prcp[:,:,:] = imerg_agg
    

    liqprcp = new_cdf.createVariable('prabliquidprcp', 'f4', ('latitude','longitude','time'), zlib=True)
    liqprcp.units = 'mm/hr'
    liqprcp.long_name = 'probabilityLiquidPrecipitation'
    liqprcp[:,:,:] = imerg_PLP
    
    new_cdf.close()

#==============================================================================


if __name__ == "__main__":
    
    timeRange = (datetime(2017,1,1),datetime(2021,12,31,23,30))  # endday included
    write_to_file(timeRange,55,20,-130,-60)  #latstart,latend,lonstart,lonend

    