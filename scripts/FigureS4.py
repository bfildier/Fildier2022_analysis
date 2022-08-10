#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S4

Created on Thu Jun 16 16:05:54 2022

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
    
#%% Draw FigureS4

    i_fig = 4

    label_jump = '^\dagger'
    m_to_cm = 1e2
    day_to_seconds = 86400
    hPa_to_Pa = 1e2
    
    days2show = days

    def computeBeta(pres,pres_jump,rh_min,rh_max,alpha,i_surf=-1):
        """beta exponent
        
        Arguments:
        - pres: reference pressure array (hPa)
        - pres_jump: level of RH jump (hPa)
        - rh_max: lower-tropospheric RH
        - rh_min: upper-tropospheric RH
        - alpha: power exponent
        """
        
        hPa_to_Pa = 100 
    
        # init
        beta = np.full(pres.shape,np.nan)
        # lower troposphere
        lowert = pres >= pres_jump
        beta[lowert] = (alpha+1)/(1 - (1-rh_min/rh_max)*(pres_jump/pres[lowert])**(alpha+1))
        # upper troposphere
        uppert = pres < pres_jump
        beta[uppert] = alpha+1
        
        return beta

    def scatterDensity(ax,x,y,s,alpha):
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        
        return ax.scatter(x,y,c=z,s=s,alpha=0.4)
        
    def createMask(day):
        
        # remove surface peaks
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        remove_sfc = z_peak > 0.5 # km
        
        # keep large peaks only
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        keep_large = np.logical_and(remove_sfc,qrad_peak > 2.5) # K/day
        
        # keep soundings in domain of interest
        lon_day = data_all.sel(launch_time=day).longitude.data[:,50]
        lat_day = data_all.sel(launch_time=day).latitude.data[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        
        # merge all
        keep = np.logical_and(keep_large,keep_box)
        
        return keep

    def createMaskIntrusions(day):
        
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        
        if day in days_high_peaks:
                
            # subset without intrusions
            i_d = np.where(np.array(days_high_peaks) == day)[0][0]
            k_high = np.logical_and(z_peak <= z_max_all[i_d],
                                    z_peak > z_min_all[i_d])
            k_low = np.logical_not(k_high)
        
            return k_low
        
        else:
            
            return np.full(z_peak.shape,True)
        

    fig,axs = plt.subplots(ncols=1,nrows=3,figsize=(4.5,13.5))
    plt.subplots_adjust(hspace=0.3,wspace=0.3)



#-- (a) peak magnitude, using the approximation for the full profile, showing all profiles
    ax = axs[0]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    mask_all = {}
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            # beta
            beta = f.beta_peak[i_s]
            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak] + rs.spectral_integral_vr[i_s][i_peak]
            # constants
            C = -gg/c_pd
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * beta * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
    
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
    
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
        ## mask
        # geometry
        keep = createMask(day)
        # intrusions
        remove_instrusions = createMaskIntrusions(day)
        # merge
        mask_all[day] = np.logical_and(keep,remove_instrusions)
    
    # plot
    m = np.hstack([mask_all[day] for day in days2show])
    m_inv = np.logical_not(m)
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    
    ax.scatter(x[m_inv],y[m_inv],s[m_inv],'k',alpha=0.1)
    # scatterDensity(ax,x,y,s,alpha=0.5)
    h = scatterDensity(ax,x[m],y[m],s[m],alpha=0.5)

    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('Measured cooling peak magnitude (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Peak magnitude as eq. (8)')
    # square figure limits
    xlim = (-25.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    xmin,xmax = np.min(x), np.max(x) 
    xrange = xmax-xmin
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x[m], y[m],p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x[m],y[m])
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')

    # # 1:1 line
    # ax.plot([-21,0],[-21,0],'k')

    # write numbers
    t = ax.text(0.05,0.05,'slope = %1.2f \n r = %1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="2%",  # width = 70% of parent_bbox width
                    height="50%",  # height : 5%
                    loc='center left')
    cb = fig.colorbar(h, cax=axins1, orientation="vertical")
    axins1.yaxis.set_ticks_position("right")
    axins1.tick_params(axis='y', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=5)
    
    
    
    alpha_qvstar = 2.3
    piB_star = 0.0054
    delta_nu = 160 # cm-1
    
#-- (b) peak magnitude, using the intermediate approximation (beta and 1 wavenumber), showing all profiles
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    mask_all = {}
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):

            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            # beta
            beta = f.beta_peak[i_s]
            # approximation of spectral integral
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            C = -gg/c_pd
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * beta * spec_int_approx * day_to_seconds
        
        H_peak_all[day] = H_peak
    
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
    
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
        ## mask
        # geometry
        keep = createMask(day)
        # intrusions
        remove_instrusions = createMaskIntrusions(day)
        # merge
        mask_all[day] = np.logical_and(keep,remove_instrusions)
    
    # plot
    m = np.hstack([mask_all[day] for day in days2show])
    m_inv = np.logical_not(m)
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    
    ax.scatter(x[m_inv],y[m_inv],s[m_inv],'k',alpha=0.1)
    # scatterDensity(ax,x,y,s,alpha=0.5)
    h = scatterDensity(ax,x[m],y[m],s[m],alpha=0.5)

    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('Measured cooling peak magnitude (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Peak magnitude as eq. (9)')
    # square figure limits
    xlim = (-25.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    xmin,xmax = np.min(x), np.max(x) 
    xrange = xmax-xmin
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x[m], y[m],p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x[m],y[m])
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')

    # # 1:1 line
    # ax.plot([-21,0],[-21,0],'k')

    # write numbers
    t = ax.text(0.05,0.05,'slope = %1.2f \n r = %1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="2%",  # width = 70% of parent_bbox width
                    height="50%",  # height : 5%
                    loc='center left')
    cb = fig.colorbar(h, cax=axins1, orientation="vertical")
    axins1.yaxis.set_ticks_position("right")
    axins1.tick_params(axis='y', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=5)
    
    
    
#-- (c) peak magnitude, using the simplified scaling (RH step function and 1 wavenumber), showing all profiles
    ax = axs[2]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    mask_all = {}
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # approximation of spectral integral
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            alpha = alpha_qvstar
            C = -gg/c_pd * (1+alpha)
            
            # compute
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int_approx * day_to_seconds
 
            # # Try including inversion factor
            # deltaT_inv = 5
            # inv_factor = exp(-0.065*deltaT_inv)
            # # compute
            # H_peak[i_s] = C/(p_peak*hPa_to_Pa) * inv_factor * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int_approx * day_to_seconds
        
        H_peak_all[day] = H_peak
    
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak

        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))

        ## mask
        # geometry
        keep = createMask(day)
        # intrusions
        remove_instrusions = createMaskIntrusions(day)
        # merge
        mask_all[day] = np.logical_and(keep,remove_instrusions)
    
    # plot
    m = np.hstack([mask_all[day] for day in days2show])
    m_inv = np.logical_not(m)
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    
    ax.scatter(x[m_inv],y[m_inv],s[m_inv],'k',alpha=0.1)
    h = scatterDensity(ax,x[m],y[m],s[m],alpha=0.5)

    
    ax.set_xlabel('Measured cooling peak magnitude (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Peak magnitude as eq. (10)')
    # square figure limits
    xlim = (-25.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    xmin,xmax = np.min(x), np.max(x) 
    xrange = xmax-xmin
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x[m], y[m],p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x[m],y[m])
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')

    # # 1:1 line
    # ax.plot([-21,0],[-21,0],'k')

    # write numbers
    t = ax.text(0.05,0.05,'slope = %1.2f \n r = %1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="2%",  # width = 70% of parent_bbox width
                    height="50%",  # height : 5%
                    loc='center left')
    cb = fig.colorbar(h, cax=axins1, orientation="vertical")
    axins1.yaxis.set_ticks_position("right")
    axins1.tick_params(axis='y', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=5)
        
 
    
    #--- Add panel labeling
    pan_labs = '(a)','(b)','(c)'
    for ax,pan_lab in zip(axs,pan_labs):
        t = ax.text(0.03,0.92,pan_lab,transform=ax.transAxes,fontsize=14)
        t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    #--- save
    # plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')
    
