"""
Created on Mon Feb 28 2022
@authors: Jonathan Rheinlænder
"""

from pynextsim.projection_info import ProjectionInfo
from netCDF4 import Dataset, num2date, date2num
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes


def get_vol_fluxes(ds, var, bbox, res=5e3):
    """
    Parameters:
    -----------
    ds : xarray.Dataset
    var : xarray.DataArray
        'sit' or 'sit_thin'S
    bbox : tuple(int)
        (i0, i1, j0, j1), where
            i0 = bottom row index
            i1 = top row index
            j0 = left-most column index
            j1 = right-most column index
    res : float
        moorings resolution [m]

    Returns:
    --------
    flux : numpy.ndarray
        outward volume flux for each time entry [m^3/s]
    """

    flux = 0
    cell_area = res**2
    i0, i1, j0, j1 = bbox
    for vname, factor, i, j in [
        ('siv', 1,  i1, slice(j0, j1+1)), #top
        ('siv', -1, i0, slice(j0, j1+1)), #bottom
        ('siu', 1,  slice(i0, i1+1), j1),  #rhs
        ]:
        
        vel_out = factor*ds[vname][:,i,j].squeeze()
        Q = cell_area*vel_out*var[:,i,j].squeeze()
        flux += np.nansum(Q, axis=1)/res
    return flux

def flux_through_gate(ds, var, gate, res=5e3):
    """
    Parameters:
    -----------
    ds : xarray.Dataset
    var :xarray.DataArray
        e.g. 'sit' or 'sit_thin'
    gate : tuple(int) (vname, factor, y-slice, x-slice)
        
    res : float
        moorings resolution [m]

    Returns:
    --------
    flux_gate : numpy.ndarray
        outward volume flux for each time entry [m^3/s]
    """
    flux_gate = 0
    cell_area = res**2
       
    for vname, factor, i, j in [gate]:        
        vel_out = factor*ds[vname][:,i,j].squeeze()
        Q = cell_area*vel_out*var[:,i,j].squeeze()
        flux_gate += np.nansum(Q, axis=1)/res
    
    return flux_gate