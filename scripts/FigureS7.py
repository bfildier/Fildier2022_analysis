#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S7

Created on Fri Jul 29 12:22:04 2022

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
    
#%%  Figure S7 -- intrusion height and water path, with fixed and varying kappa

i_fig = 7

#-- data

N_sample = Nsample = 20
inds_uniformRH = slice(6,26)

# coordinates
W_all = [float(str(radprf_MI_20200213lower.name[inds_uniformRH][i].data)[2:6]) for i in range(N_sample)]
# W_all = np.array(W_all)-W_ref
H_all = [float(str(radprf_MI_20200213lower.name[inds_uniformRH.stop:inds_uniformRH.stop+N_sample][i].data)[11:15]) for i in range(N_sample)]


i_ref = 1
radprf2show = radprf_MI_20200213lower

# peak height
# z_jump = 1.66883978
z_jump = 1.81
z = radprf2show.zlay[0]/1e3 # km
k_jump = i_jump = np.where(z>=z_jump)[0][0]
pres_jump = radprf2show.play[k_jump]/1e2 # hPa

qradlw_peak = np.full((Nsample,Nsample),np.nan)
qradlw_peak_fix_k = np.full((Nsample,Nsample),np.nan)
qradlw_peak_ref = np.full((Nsample,Nsample),np.nan)

delta_qradlw_peak = np.full((Nsample,Nsample),np.nan)
delta_qradlw_peak_fix_k = np.full((Nsample,Nsample),np.nan)

ratio_qradlw_peak = np.full((Nsample,Nsample),np.nan)
ratio_qradlw_peak_fix_k = np.full((Nsample,Nsample),np.nan)

for i_W in range(Nsample):
    for i_H in range(Nsample):
        
        W = W_all[i_W]
        H = H_all[i_H]
        name = 'W_%2.2fmm_H_%1.2fkm'%(W,H)
        i_prof = np.where(np.isin(radprf2show.name.data,name))[0][0]
        
        qradlw_peak[i_W,i_H] = radprf_MI_20200213lower.q_rad_lw[i_prof,k_jump].data
        qradlw_peak_fix_k[i_W,i_H] = radprf_MI_20200213lower_fix_k.q_rad_lw[i_prof,k_jump].data
        qradlw_peak_ref[i_W,i_H] = radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        
        delta_qradlw_peak[i_W,i_H] = radprf_MI_20200213lower.q_rad_lw[i_prof,k_jump].data - radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        delta_qradlw_peak_fix_k[i_W,i_H] = radprf_MI_20200213lower_fix_k.q_rad_lw[i_prof,k_jump].data - radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        
        ratio_qradlw_peak[i_W,i_H] = radprf_MI_20200213lower.q_rad_lw[i_prof,k_jump].data / radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        ratio_qradlw_peak_fix_k[i_W,i_H] = radprf_MI_20200213lower_fix_k.q_rad_lw[i_prof,k_jump].data / radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        

#-- plot

fig,axs = plt.subplots(figsize=(4.5,8),nrows=2)
fig.tight_layout()
fig.subplots_adjust(hspace=0.25)

# cmap = plt.cm.nipy_spectral_r
# cmap = plt.cm.Spectral_r
cmap = plt.cm.RdYlBu_r
vmin = 0.
vmax = 1.
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

#--  delta Qrad
ax = axs[0]

# colors
ax.contourf(W_all,H_all,ratio_qradlw_peak.T,levels=30,cmap=cmap,vmin=vmin,vmax=vmax)

# lines
cont = ax.contour(W_all,H_all,ratio_qradlw_peak.T,levels=np.linspace(0.1,0.9,9),colors=('grey',),
                  linestyles=('-',),linewidths=(0.8,),vmin=vmin,vmax=vmax)
plt.clabel(cont, fmt = '%1.1f', colors = 'k', fontsize=10) #contour line labels

# labels
ax.set_xlabel('Intrusion water path (mm)')
ax.set_ylabel('Intrusion height (km)')

# colorbar
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
# cb.set_label(r'LW $Q_{rad}$ peak (K/day)')
cb.set_label(r'Ratio to reference peak')

ax.set_title(r'Normalized lower peak (at 1.81 km)')


#-- with fixed kappa(nu)
ax = axs[1]

# colors
ax.contourf(W_all,H_all,ratio_qradlw_peak_fix_k.T,levels=30,cmap=cmap,vmin=vmin,vmax=vmax)

# lines
cont = ax.contour(W_all,H_all,ratio_qradlw_peak_fix_k.T,levels=np.linspace(0.1,0.9,9),colors=('grey',),
                  linestyles=('-',),linewidths=(0.8,),vmin=vmin,vmax=vmax)
plt.clabel(cont, fmt = '%1.1f', colors = 'k', fontsize=10) #contour line labels

# labels
ax.set_xlabel('Intrusion water path (mm)')
ax.set_ylabel('Intrusion height (km)')

# colorbar
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
# cb.set_label(r'LW $Q_{rad}$ peak (K/day)')
cb.set_label(r'Ratio to reference peak')

ax.set_title(r'when fixing $\kappa_\nu$(T=290K,p=800hPa)')

#--- Add panel labeling
pan_labs = '(a)','(b)'
pan_cols = 'k','k'
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    t = ax.text(0.02,0.98,pan_lab,c=pan_col,ha='left',va='top',
            transform=ax.transAxes,fontsize=16)
    t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))

#--- save
# plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')
