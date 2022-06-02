"""
Created on Tue Dec 21 2021
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


def interp2grid(src_x, src_y, src_field, dst_x, dst_y, method='nearest'):
    '''
    Parameters:
    -----------
    src_x : 2D np.ndarray
        2D array (x,y) source points x-coords
    src_y : 2D np.ndarray
        2D array (x,y) source points y-coords    
    src_field : 2D np.ndarray
        Data values (x,y) on the source grid
    dst_x : 2D np.ndarray
        2D array (xi,yi) destination points x-coords
    dst_y : 2D np.ndarray
        2D array (xi,yi) destination points y-coords
    method : {‘linear’, ‘nearest’, ‘cubic’}
        interpolation method of scipy.interpolate.griddata (default is "nearest")
    '''

    # flatten 2D arrays
    points = np.array( (src_x.flatten(), src_y.flatten()) ).T

    vals = src_field.flatten()

    fld_interp = griddata(points, vals, (dst_x, dst_y), method=method)

    return fld_interp 


def arclead_to_nextsim_grid(arc, nlon, nlat):
    """regridding the Arcleads product to the Nextsim grid"""
    
    proj = ProjectionInfo() # default nextsim projection

    # get destination grid; x and y grid from nextsim grid
    dst_x,dst_y = proj.pyproj(nlon, nlat)
    
    # get source grid; Project lon and lat from Arcleads to x,y grid using nextsim projection
    src_x,src_y = proj.pyproj(arc['longitude'].values, arc['latitude'].values) 

    #  source field
    src_f = arc.leadMap

    # use the shape from dst grid
    nx,ny = dst_x.shape
    nt = src_f.shape[0] 
    dst_f = np.zeros((nt, nx, ny))
    print(dst_f.shape)

    # loop over time axis in source field
    for t,time in enumerate(src_f.time):
        vals = src_f.isel(time=[t]).squeeze().values
        dst_f[t,:,:] = interp2grid(src_x,
                                   src_y,
                                   vals, 
                                   dst_x, 
                                   dst_y,
                                   method='nearest')
    print('Interpolation done!')

    # convert interpolated leadmap product back to DataArray
    leadmap = dst_f
    time = arc.indexes['time'].to_datetimeindex()
    surface_classes = arc.attrs['surface classes']

    ds = xr.Dataset(   
        data_vars=dict(
            leadmap=(["time", "y", "x"], leadmap)    ),
        coords=dict(
            time=time   ),
        attrs=dict(
            description="ArcLeads product interpolated onto nextsim grid",
            surface_classes= surface_classes),
    )

    return ds

def make_cbar_arcleads(ax):
    
    # define colors
    col_dict={0:"darkgrey",
              1:"black",
              2:"ghostwhite",
              3:"moccasin",
              4:"mediumblue"}

    # We create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    
    # get colorbar tick labels 
    #attr = ds.attrs['surface_classes']

    labels={0:"land",
            1:"clouds",
            2:"sea ice",
            3:"artifact",
            4:"lead"}

    # prepare normalizer
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.
    norm_bins = np.insert(norm_bins, len_lab, np.max(norm_bins) + 1.0)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cax = inset_axes(ax,
                width="5%",  # width = 50% of parent_bbox width
                height="100%",  # height : 5%
                loc='lower left',
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0)

    cbar = plt.colorbar(im, cax=cax, format=fmt, ticks=tickz)
    
    return cbar

    
    
def plotArclead(ax, ds, dto, add_colorbar=True, **cbargs):
    # define colors
    col_dict={0:"darkgrey",
              1:"black",
              2:"ghostwhite",
              3:"moccasin",
              4:"mediumblue"}

    # We create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    
    # get colorbar tick labels 
    attr = ds.attrs['surface_classes']

    labels={0:"land",
            1:"clouds",
            2:"sea ice",
            3:"artifact",
            4:"lead"}

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.
    norm_bins = np.insert(norm_bins, 5, np.max(norm_bins) + 1.0)

    ## Make normalizer and formatter
    norm = mpl.colors.BoundaryNorm(norm_bins, len(labels), clip=True)
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
    
    # prepare plot
    lons = ds.longitude.transpose()
    lats = ds.latitude.transpose()
    leadMap = ds.leadmap.transpose()
    leadMap = leadMap.sel(time=dto).squeeze()
    
    ax.add_feature(cartopy.feature.LAND,zorder=1,alpha=1, facecolor="darkgrey")
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.set_aspect(1)
    ax.gridlines(zorder=2,linewidth=0.5, alpha=0.5,linestyle="--")
    ax.set_extent([-100, -50, 67, 87], ccrs.PlateCarree(central_longitude=300))

    im = ax.pcolormesh(lons, lats, leadMap, transform=ccrs.PlateCarree(), cmap=cm, norm=norm)

    # add title inside subplots
    ax.text(0.03,0.05,dto,
            horizontalalignment='left',
            transform=ax.transAxes, 
            bbox=dict(facecolor='white', edgecolor='None', alpha=0.75))

    if add_colorbar:
        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2
        cax = inset_axes(ax,
                    width="5%",  # width = 50% of parent_bbox width
                    height="100%",  # height : 5%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0., 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0)

        out = plt.colorbar(im, cax=cax, format=fmt, ticks=tickz)

    return fig,ax

def get_deformation(uv, res=5000, dt=24*60*60):
    '''
    Parameters:
    -----------
    uv : list
        uv = [u, v] with u, v x/y or lon/lat components of wind velocity
    res: resolution of moorings
    dt: time conversion from sec to day    
        
    '''
    
    # [1/s ==> 1/d]
    u_x, v_x = [dt*np.gradient(a, axis=1)/res for a in uv]
    u_y, v_y = [dt*np.gradient(a, axis=0)/res for a in uv]
    
    div = u_x + v_y
    shear = np.hypot(u_x - v_y, u_y + v_x)   
    deform = np.hypot(div, shear)   
    
    return shear, div, deform


def to_dataset(data_vars, coords, attrs, **kwargs):
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs)
    
    return ds