#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S6

Created on Wed Aug  3 10:38:12 2022

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


#%% Figure S6 -- Moist intrusion, spectral figure

i_fig = 6

fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(8.5,4))

W = 2.39 # mm
H1 = 3 # km
H2 = 5.95 # km
i_0 = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_uniform_RH'%(W)))[0][0]
i_1 = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_H_%1.2fkm'%(W,H1)))[0][0]
i_2 = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_H_%1.2fkm'%(W,H2)))[0][0]

i_zint_0 = i_zint_1 = np.where(z >= H1)[0][0]
i_zint_2 = np.where(z >= H2)[0][0]

# inds = (1,5,i_0,i_1,i_2)
# N_prof = len(inds)
# radprf_s = [radprf_MI_20200213lower]*N_prof
# cols = 'grey','k','b','b','b'
# linestyles = '-','-','--','-',':'

inds = (1,i_1,i_2)
inds_int = (i_zint_0,i_zint_1,i_zint_2)
N_prof = len(inds)
radprf_s = [radprf_MI_20200213lower]*N_prof
cols = 'grey','b','b','b'
linestyles = '-','-',':'


z = np.array(radprf_MI_20200213.zlay[0]/1e3) # km

##-- (a) RH
ax = axs[0]

for i_prof,radprf,col,linestyle in zip(inds,radprf_s,cols,linestyles):

    rh_prof = radprf['rh'].data[i_prof]
    ax.plot(rh_prof,z,c=col,linestyle=linestyle)
    
ax.set_ylabel('z(km)')
ax.set_xlabel(r'Relative humidity')
ax.set_title('Varying intrusion height ($W=%1.2f$mm)'%W)
ax.set_xlim((-0.03,1.03))
ax.set_ylim((-0.15,8.15))

##-- (b) nu_star
ax = axs[1]

def computeWPaboveZ(qv,pres,p_top):
    """Calculates the integrated water path above each level.

    Arguments:
        - qv: specific humidity in kg/kg, Nz-vector
        - pres: pressure coordinate in hPa, Nz vector
        - p_top: pressure of upper integration level

    returns:
        - wp_z: water path above each level, Nz-vector"""

    Np = qv.shape[0]
    wp_z = np.full(Np,np.nan)

    p_increasing = np.diff(pres)[0] > 0
    
    if p_increasing:
        
        i_p_top = np.where(pres >= p_top)[0][0]
        
        for i_p in range(i_p_top,Np):
        # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)

            arr = qv
            p = pres
            p0 = p_top
            p1 = p[i_p]
            i_w = i_p
            
            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    else:
        
        i_p_top = np.where(pres >= p_top)[0][-1]

        for i_p in range(i_p_top):
            
            arr = np.flip(qv)
            p = np.flip(pres)
            p0 = p_top
            p1 = pres[i_p]
            i_w = i_p

            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    return wp_z

def computeNuProfile(i_prof,radprf,band='rot'):
    
    qv_prof = radprf['h2o'].data[i_prof]
    pres_prof = radprf['play'].data/1e2
    W_prof = computeWPaboveZ(qv_prof,pres_prof,0)
    kappa_prof = 1/W_prof
    nu_prof = rad_scaling_all['20200213'].nu(kappa_prof,band=band)/1e2 # in cm-1
    
    return nu_prof,W_prof
    
#- reference without intrusion (from piecewise linear fit)
#- reference with intrusion (from piecewise linear fit)
#- homogenized rh, same water path at peak level
# radprf_s = radprf_MI_20200213, radprf_MI_20200213,radprf_RMI_20200213

band = 'rot'

z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
i_jump = np.where(z >= z_jump)[0][0]

nu_peak_all = []
nu_int_all = []

for i_prof,i_int,radprf,col,linestyle in zip(inds,inds_int,radprf_s,cols,linestyles):

    nu_prof,W_prof = computeNuProfile(i_prof,radprf,band=band)
    print('W_peak =',W_prof[i_jump])
    nu_peak_all.append(nu_prof[i_jump])
    nu_int_all.append(nu_prof[i_int])
    ax.plot(nu_prof,z,c=col,linestyle=linestyle)

# add points at peak height
ax.scatter(nu_peak_all[1:2],[z_jump],facecolor='k')#cols[1:2])
inv = ax.transData.inverted()
# ax.plot([nu_peak_all[1]]*2,[-1,z_jump],c='k',linestyle='-',linewidth=0.5)
# add points at intrusion center of mass
ax.scatter(nu_int_all[1:],[z[i_zint_1],z[i_zint_2]],facecolor='none',edgecolor='k')#cols[1:])
# ax.plot([nu_int_all[1]]*2,[-1,z[i_zint_1]],c='k',linestyle='-',linewidth=0.5)
# ax.plot([nu_int_all[2]]*2,[-1,z[i_zint_2]],c='k',linestyle='-',linewidth=0.5)
ax.grid()

ax.set_ylabel(' z(km)')
ax.set_xlabel(r'$\tilde{\nu}^+$ (cm$^{-1}$), obeying $\kappa(\tilde{\nu}^+)W(z) = 1$')
ax.set_title(r'Most emitting/absorbing $\nu$ ($\tau = 1$)')
ax.set_xlim((350,800))
ax.set_ylim((-0.15,8.15))

#--- Add panel labeling
pan_labs = '(a)','(b)'
for ax,pan_lab in zip(axs,pan_labs):
    t = ax.text(0.03,0.92,pan_lab,transform=ax.transAxes,fontsize=14)
    t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))

#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
# plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')
