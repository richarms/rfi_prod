#!/usr/bin/env python
# coding: utf-8


import katdal
import h5py
import numpy as np
import matplotlib as plt
import xarray as xr
import pandas as pd
import pylab as plt
import time as tme
from dask import array as da



def readfile(pathfullvis):
    '''
    Reading in the full visibility file
    
    Arg : path2full visibity file
    '''

    visfull = katdal.open(pathfullvis)
    
    return visfull


def remove_bad_ants(fullvis):
    '''
    Input: Take a fullvis rdb file, called h5 here, which is open using:
        MyFile = "1543190535_sdp_l0.rdb"
        MyLiteVis = katdal.open(MyFile)
        h5 = MyLiteVis.select(scans='track')
    Output: List of good antennas
    '''
    # This pull all the antenna used for observation
    AntList = []
    for ant in fullvis.ants:
        AntList.append(ant.name)
    
    # This will give the antenna activity list
    AntsActivity = []
    for AntName in AntList: 
        AntsActivity.append((AntName, fullvis.sensor[AntName+'_activity']))

    for i in range(len(AntsActivity)):
        if 'stop' in AntsActivity[i][1]:
            AntList.remove(AntsActivity[i][0])
        else:
            pass
    
    return AntList




def select_and_apply_with_good_ants(fullvis,pol_to_use, corrprod,compscan,scan,clean_ants):
    '''
    This function is going to select correlation products.
        
    Arg: full visibility, pol to use, corrproducts and scan and good antennas.
    
    Output : The flags array
    '''
    
    fullvis.select(corrprods = corrprod, pol = pol_to_use,compscans=compscan ,scans = scan,ants = clean_ants)
    
    
    flag = fullvis.flags[:, :, :]
    return flag



def get_az_and_el(fullvis):
    '''
    Getting the full the elevation and azimuth of the file.
    
    Arg: fullvis file
    
    Return: List of avaraged elevation and azimuth of all antennas per time stamp
    '''
    
    # Getting the azmuth and elevation
    az = fullvis.az
    el = fullvis.el
    
    azmean = (np.array([np.mean(az[:][i]) for i in range(az.shape[0])])).astype(int)
    elmean = (np.array([np.mean(el[:][i]) for i in range(el.shape[0])])).astype(int)
    
    return elmean,azmean


def get_time_idx(fullvis):
    import datetime
    '''
    This function is going to convert unix time to hour of a day
    
    Input : h5 obeject
    
    Output : list with time dumps converted to hour of a day
    '''
    unix  = fullvis.timestamps

    local_time = []
    for i in range(len(unix)):
        local_time.append(datetime.datetime.fromtimestamp((unix[i])).strftime('%H:%M:%S'))

    # Converting time to hour of a day
    hour = []
    for i in range(len(local_time)):
        hour.append(int(round(int(local_time[i][:2]) + int(local_time[i][3:5])/60 + float(local_time[i][-2:])/3600)))
    return np.array(hour)[None,:]



def get_az_idx(az,bins):
    '''
    This function is going get the index of the azimuth 
    
    Input : List of Azimuthal angle and azimuthal bins
    
    Output : Azimuthal index
    '''
    az_idx = []
    for az in az:
        for j in range(len(bins)-1):
            if bins[j] <= az < bins[j+1]:
                az_idx.append(j)
    
    return np.array(az_idx)[None,:]



def get_el_idx(el,bins):
    '''
    This function is going get the index of the elevation
    
    Input : List of elevation angle and bins
    
    Output : Elevation index
    
    '''
    el_idx = []
    for el in el:
        for j in range(len(bins)-1):
            if bins[j] <= el < bins[j+1]:
                el_idx.append(j)
    
    return np.array(el_idx)[None,:] 


def get_corrprods(fullvis):
    '''
    This function is getting the corr products
    
    Input : Visibility file
    
    Output : Correlation products
    '''
    bl = fullvis.corr_products
    bl_idx = []
    for i in range(len(bl)):
        bl_idx.append((bl[i][0][0:-1]+bl[i][1][0:-1]))
            
    return np.array(bl_idx)


def get_bl_idx(corr_prods,nant):
    '''
    This function is getting the index of the correlation products
    
    Input  : Correlation products, number of antennas
    
    Output : Baseline index
    '''
    nant = nant
    A1, A2 = np.empty(nant*(nant-1)/2, dtype=np.int32), np.empty(nant*(nant-1)/2, dtype=np.int32)
    k = 0
    for i in range(nant):
        for j in range(i+1,nant):
            A1[k] = i
            A2[k] = j
            k += 1

    # Baseline antenna cobinations
    corr_products = np.array(['m{:03d}m{:03d}'.format(A1[i], A2[i]) for i in range(len(A1))])
    
    # Number of baselines
    nbl = (nant**2 - nant)/2


    df = pd.DataFrame(data=np.arange(nbl), index=corr_products).T
    
    bl_idx = df[corr_prods].values
    return (bl_idx[0])[:,None],corr_products


if __name__ == "__main__":
    # Getting the files
    # Getting the files
    import os, fnmatch

    listOfFiles = os.listdir('/scratch2/Data_for_isaac_RFI/')  
    pattern = "*.rdb"  

    datafiles = []
    for entry in listOfFiles:  
        if fnmatch.fnmatch(entry, pattern):
            datafiles.append(entry)
    #Initializing the master array and the weghting
    master = np.zeros((24,4096,2016,10,24),dtype=np.uint16)
    counter = np.zeros((24,4096,2016,10,24),dtype=np.uint16)
    badfiles =[]
    for i in range(len(datafiles)):
        try:
            print 'adding file number',i
            start = tme.time()
            # Path to the files
            pathfullvis = '/scratch2/Data_for_isaac_RFI/'+datafiles[i]
            # Reading in the files
            fullvis= readfile(pathfullvis)
            # List of clean atennas
            clean_ants = remove_bad_ants(fullvis)
            # Good flags
            good_flags = select_and_apply_with_good_ants(fullvis,pol_to_use = 'HH',corrprod='cross',compscan='track',
                                                         scan='track',clean_ants = clean_ants)
            # Elevation and Azimuth 
            el,az = get_az_and_el(fullvis)
            # time index
            time_idx = get_time_idx(fullvis)
            # Getting azimuth and elevation
            az_idx = get_az_idx(az,np.arange(0,370,15))
            # Elevation index
            el_idx = get_el_idx(el,np.arange(0,100,10))
            # Corr_produts
            corr_prods = get_corrprods(fullvis)
            # Baseline index
            bl_idx,corr_products = get_bl_idx(corr_prods,nant=64)
            # Updating the master array
            master[time_idx,:,bl_idx,el_idx,az_idx] += np.transpose(good_flags, axes=[2,0,1])
            counter[time_idx,:,bl_idx,el_idx,az_idx] += 1

            print 'It took ',tme.time() - start,' seconds to add file number',i

            np.save('/scratch2/Data_for_isaac_RFI/master.npy',master)
            np.save('/scratch2/Data_for_isaac_RFI/counter.npy',counter)
            print 'It has saved file number',i,' into a disk'


        except:
            print datafiles[i],'file has a problem'
            badfiles.append(datafiles[i])
            np.save('/scratch2/Data_for_isaac_RFI/badfiles.npy',badfiles)
            pass



    



