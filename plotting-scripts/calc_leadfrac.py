"""
Created on Tue Dec 21 2021
@authors: Jonathan Rheinlænder
"""

from pynextsim.projection_info import ProjectionInfo
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

def calc_leadAreaFrac_MODIS(arc, bbox=None):
    
    '''
    Parameters:
    -----------
    arc : xarray.dataset
        Arcleads dataset
    bbox : tuple or ndarray
        bounding box of x and y points (x0,x1,y0,y1) or boolean array (mask)   
    '''
    
    # mask out leads
    leads = xr.where(arc.leadmap==4, 1, 0)

    # apply land mask
    leads_masked = leads.where(arc.leadmap!=0)

    if bbox:
        if type(bbox) == np.ndarray: #if bbox is a mask ndarray
            leads_masked = leads_masked.where(bbox)
        else: # bbox is a tuple
            (x0,x1,y0,y1) = bbox
            leads_masked = leads_masked.sel(x=slice(x0,x1), y=slice(y0,y1))
  
    lead_area_frac = leads_masked.mean(dim=("x", "y"))

    return lead_area_frac


def get_leadfraction_nextsim(ds):
    
    owfraction = 1 - ds['sic']
    leadfraction = owfraction + ds['sic_thin']

    return leadfraction

def leadmap_nextsim(ds, cutoff=0.05):
    
    leadfraction = get_leadfraction_nextsim(ds)
    leadmap = xr.where(leadfraction>cutoff, 1, 0)
    
    # apply land mask
    leadmap_masked = leadmap.where(~xr.ufuncs.isnan(leadfraction[0]))
     
    return leadmap_masked

def calc_leadAreaFrac_nextsim(ds, cutoff=0.05, bbox=None):
    
    owfraction = 1 - ds['sic']
    leadfraction = owfraction + ds['sic_thin']

    leadmap = xr.where(leadfraction>cutoff, 1, 0)
    
    # apply land mask
    leads_masked = leadmap.where(~xr.ufuncs.isnan(owfraction[0]))
    
    if bbox:
        if type(bbox) == np.ndarray: #if bbox is a mask ndarray
            leads_masked = leads_masked.where(bbox)
        else: # bbox is a tuple
            (x0,x1,y0,y1) = bbox
            leads_masked = leads_masked.sel(x=slice(x0,x1), y=slice(y0,y1))
    
    # find total number of gridcells identified as leads
    nlead = np.count_nonzero(leads_masked)
    
    # Calculate the mean lead area fraction as a fraction of the Beaufort Sea area
    lead_area_frac = leads_masked.mean(dim=("x", "y"))
      
    return lead_area_frac, nlead

