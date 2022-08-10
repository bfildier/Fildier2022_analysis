#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1

Created on Thu Aug  4 07:28:36 2022

@author: bfildier
"""

##-- modules

# general
import scipy.io
import sys, os, glob
import numpy as np
import xarray as xr
from datetime import datetime as dt
from datetime import timedelta, timezone
import pytz
import pickle
import argparse

# stats
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from scipy import optimize

# images
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from PIL import Image
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

##-- directories

workdir = os.path.dirname(os.path.realpath(__file__))
# workdir = '/Users/bfildier/Code/analyses/EUREC4A/Fildier2022_analysis/scripts'
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures','paper')
inputdir = os.path.join(repodir,"input")
radinputdir = os.path.join(repodir,"input")
imagedir = os.path.join(repodir,'figures','snapshots','with_HALO_circle')


# Load own module
projectname = 'Fildier2022_analysis'
thismodule = sys.modules[__name__]

## Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
from radiativescaling import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *
from thermoConstants import *
from thermoFunctions import *

mo = MatrixOperators()

##--- local functions

def defineSimDirectories():
    """Create specific subdirectories"""
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)
    
    
if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Draw paper figures from all precomputed data")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)

    # output directory
    defineSimDirectories()
    
    ##-- Load all data
    
    exec(open(os.path.join(workdir,"load_data.py")).read())


    
#%% Figure 1


    def getProfileCoords(rad_features, data_day):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak.data)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude.data[:,50]
        lat_day = data_day.latitude.data[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        
        # combined
        k = np.logical_and(keep_large,keep_box)
        
        # longitude
        lon = lon_day[k]
        # latitude
        lat = lat_day[k]
        # time
        time_day = np.array([dt.strptime(str(launch_time)[:16],'%Y-%m-%dT%H:%M').replace(tzinfo=pytz.UTC) for launch_time in data_day.launch_time.data])
        time = time_day[k]
        
        return lon,lat,time
    
    def getProfiles(rad_features, data_day, z_min, z_max):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude[:,50]
        lat_day = data_day.latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        # z range
        keep_between_z =  np.logical_and(rad_features.z_net_peak <= z_max, # m
                                         rad_features.z_net_peak > z_min)
        # combined
        k = np.logical_and(np.logical_and(keep_large,keep_box),keep_between_z)
        
        # relative humidity    
        rh = data_day.relative_humidity.values[k,:]*100
        # lw cooling
        qradlw = rad_features.qrad_lw_smooth[k,:]
        # longitude
        lon = lon_day[k]
        # latitude
        lat = lat_day[k]
        # original indices
        ind = np.where(k)[0]
        
        return lon,lat,rh,qradlw,ind
    
    def getProfilesPWrange(rad_features, data_day, pw_min, pw_max):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude[:,50]
        lat_day = data_day.latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        # high-level peak
        keep_between_pw =  np.logical_and(rad_features.pw <= pw_max, # m
                                          rad_features.pw > pw_min)
        # combined
        k = np.logical_and(np.logical_and(keep_large,keep_box),keep_between_pw)
        
        # relative humidity    
        rh = data_day.relative_humidity.values[k,:]*100
        # lw cooling
        qradlw = rad_features.qrad_lw_smooth[k,:]
        # longitude
        lon = lon_day[k]
        # latitude
        lat = lat_day[k]
        # original indices
        ind = np.where(k)[0]
        
        return lon,lat,rh,qradlw,ind
    
    # Figure layout
    fig = plt.figure(figsize=(10,8))
    
    gs = GridSpec(2, 4, width_ratios=[1,1,1,1], height_ratios=[1, 1.5], hspace=0.34)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[0,2:4],projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[1,:2])
    ax5 = fig.add_subplot(gs[1,2:4])
    
    cmap = plt.cm.BrBG_r
    vmax = 8
    vmin = -vmax
    
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    rad_features = rad_features_all[day]
    
    #-- background (c)
    ax = ax3
    
    # # image at specific time
    i_h = 15 # here, index = time UTC
    image = images_vis[i_h]
    
    ax.coastlines(resolution='50m')
    ax.set_extent([*lon_box,*lat_box])
    ax.imshow(image,extent=[*lon_box_goes,*lat_box_goes],origin='upper')
    gl = ax.gridlines(color='Grey',draw_labels=True)
    
    
    #-- (a), (b) & (c)
    
    # separate peaks above and below 2.5 km altitude
    cols = 'r','b'
    z_min_all = 100, 2200
    z_max_all = 2200, 100000
    pw_min_all = 10, 30
    pw_max_all = 30, 50
    alphas = 0.15, 0.2
    labs = r'peak below %1.1f km'%(z_min_all[1]/1e3), r'peak above %1.1f km'%(z_min_all[1]/1e3)
    # labs = r'PW $\le$ 30mm', r'PW $\gt$ 30mm'
    
    for col, z_min, z_max, lab, alpha in zip(cols,z_min_all,z_max_all,labs,alphas):
        
        lon, lat, rh, qradlw, ind = getProfiles(rad_features, data_day, z_min, z_max)
        rh_mean = np.nanmean(rh,axis=0)
        qradlw_mean = np.nanmean(qradlw,axis=0)
        
        for i_s in range(rh.shape[0]):
            
            # rh
            ax1.plot(rh[i_s],z,c=col,linewidth=0.3,alpha=alpha)
            # qradlw
            ax2.plot(qradlw[i_s],z,c=col,linewidth=0.3,alpha=alpha)
            # location
            time_init = pytz.utc.localize(dt(2020,1,26))
            time_image = time_init+timedelta(hours=i_h)
            time_current = dt.strptime(str(data_day.launch_time.data[ind[i_s]])[:16],'%Y-%m-%dT%H:%M')
            time_current = time_current.replace(tzinfo=pytz.UTC)
            delta_time = time_current-time_image
            delta_time_hour = (delta_time.days*86400 + delta_time.seconds)/3600
            # print(ind[i_s],time_current,delta_time_hour)
            
            # ax3.scatter(lon[i_s],lat[i_s],marker='o',color=col,alpha=exp(-np.abs(delta_time_hour)/2),s=20)
            ax3.scatter(lon[i_s],lat[i_s],marker='o',color=col,alpha=0.3,s=20)
            
        # rh
        ax1.plot(rh_mean,z,c=col,linewidth=1,alpha=1)
        # qradlw
        ax2.plot(qradlw_mean,z,c=col,linewidth=1,alpha=1,label=lab)
        
    # RH labels & range
    ax = ax1
    ax.set_xlabel(r'Relative humidity (%)')
    ax.set_ylabel(r'z (km)')
    ax.set_xlim(-8,108)
    ax.set_ylim(-0.2,6.2)
    # QradLW labels & range
    ax = ax2
    ax.set_xlabel(r'Longwave cooling (K/day)')
    # ax.set_ylabel(r'z (km)')
    ax.set_xlim(-14,1)
    ax.set_ylim(-0.2,6.2)
    ax.legend(loc='upper left', fontsize=6)
    
    
    #---- bottom panels, PW composites and circulation
    
    xlim = (8,59)
    ylim = (0,10)
    var_lab = r'Longwave cooling (K/day)'
    
    #-- (d)
    ax = ax4
    
    cond_varid = 'QRADLW'
    ref_varid = 'PW'
    
    array = cond_dist_all[cond_varid][day].cond_mean
    pw_perc = cond_dist_all[cond_varid][day].on.percentiles
    X,Y = np.meshgrid(pw_perc,z)
    ax.contourf(X,Y,array,cmap=cmap,vmin=vmin,vmax=vmax,levels=20)
    
    ax.set_xlabel(r'Precipitable water $W(z=0)$ (mm)')
    ax.set_ylabel('z(km)')
    ax.set_title(r'EUREC$^4$A observations, 2020-01-26')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    xlim = (8,59)
    ylim = (0,10)
    
    #- colorbar 
    # colors
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cols = cmap(norm(array),bytes=True)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
    cb.set_label(var_lab)

    #-- (e)
    ax = ax5

    # qrad
    X,Y = np.meshgrid(c_muller['pws'][:-1,0],c_muller['z1d'][:,0]/1e3)
    var = c_muller['Qrs_lw']
    ax.contourf(X,Y,var.T,cmap=cmap,vmin=vmin,vmax=vmax,levels=20)
    # cloud water
    ax.contour(X,Y,c_muller['qns'].T,colors='w',linewidths=1,origin='lower')
    # circulation
    ax.contour(X,Y,c_muller['psis'].T,colors='k',levels=4,linewidths=1,origin='lower')
    
    ax.set_xlabel(r'Precipitable water $W(z=0)$ (mm)')
    # ax.set_ylabel('z(km)')
    ax.set_title('Deep convective aggregation,\nexample of simulation')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    #- colorbar 
    # colors
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cols = cmap(norm(var),bytes=True)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
    cb.set_label(var_lab)
    
    #--- Add panel labeling
    axs = ax1,ax2,ax3,ax4,ax5
    pan_labs = '(a)','(b)','(c)','(d)','(e)'
    x_locs = 0.04,0.04,0.03,0.03,0.03
    y_locs = 0.04,0.04,0.04,0.93,0.93
    t_cols = 'k','k','w','k','k'
    for ax,pan_lab,x_loc,y_loc,t_col in zip(axs,pan_labs,x_locs,y_locs,t_cols):
        ax.text(x_loc,y_loc,pan_lab,transform=ax.transAxes,fontsize=14,color=t_col)
        
    #--- Save
    # plt.savefig(os.path.join(figdir,'Figure1.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'Figure1.png'),dpi=200,bbox_inches='tight')
