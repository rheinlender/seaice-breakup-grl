## Estimate ice growth from nextsim Moorings

"""
Created on Mon Dec 06 2021
@authors: Jonathan Rheinlænder
"""

import xarray as xr
import numpy as np
import datetime
import pandas as pd
import sys,os

#########################################################
class IceGrowth:
    
    def __init__(self, dataset, bbox=False):
        '''
        Parameters:
            dataset (xr.Dataset): Xarray dataset 
            bbox: list (x0, x1, y0, y1)
        '''
        self.dataset = dataset
        self.variables = ['newice', 'del_hi_thin', 'del_hi']
        
        if bbox:
            print('Applying bbox', bbox)
            x0, x1, y0, y1 = bbox
            self.dataset = self.dataset.sel(x=slice(x0,x1), y=slice(y0,y1))

        
        # Converts from slab thickness
        if self.variables == ['newice', 'del_hi_thin', 'del_hi']:
            thick_frac = self.dataset['sic'] - self.dataset['sic_thin']    # Concentration of old ice
            thin_frac = self.dataset['sic_thin']    # Concentration of thin ice

            newice = self.dataset['newice']
            del_vi_thin = self.dataset['del_hi_thin']*thin_frac 
            del_vi =self.dataset['del_hi']*thick_frac 
            
            # save to dataset
            self.dataset['newice'] = newice
            self.dataset['del_vi_thin'] =  del_vi_thin
            self.dataset['del_vi'] =  del_vi
            
            # update variable list names
            self.variables = ['newice', 'del_vi_thin', 'del_vi'] 
        
        return
        
    def fix_growthrate(self):

        '''
        Get the growth rate pr. unit output freq

        The units of the growth rate for (del_hi, del_hi_thin and newice) is in [m/day]. 
        Because the moorings are written at 3 hourly intervals we need to divide del_hi, del_hi_thin and
        newice with the output freq. This will give us the growth/melt during a 3 hr window.  
        '''

        # get freq from mooring
        dt_in_hours = self.dataset.time.dt.hour[1] - self.dataset.time.dt.hour[0]
        output_timestep = dt_in_hours.values/24 
        output_freq = 1/output_timestep

        #print("output frequency ", output_freq)

        # apply to growth rate variables
        for var in self.variables:
            self.dataset[var] = self.dataset[var]/output_freq
     
        return
    
    def calc_vol_growth(self, cell_area):
        '''
        Calculates the ice volume growth from nextsim moorings

            Parameters:
                ds (xr.Dataset): Xarray dataset 
                bbox: list (x0, x1, y0, y1)
                cell_area (int or array): grid cell area in m2

            Returns:
                delVsum (xr.Dataset): Total ice volume growth (m3/output_freq) 

        '''
        
        if self.variables != ['newice', 'del_vi_thin', 'del_vi']:
            print('wrong varlist! Should be newice, del_vi_thin or del_vi ') 
        
        
        delVsum=xr.Dataset() # make a new dataset
        for var in self.variables:
              
            # convert to volume growth (cell_area*thickness)
            delV = cell_area*self.dataset[var] # in m3/?hrs
                
            delVsum[var] = delV.sum(dim=("x", "y"), skipna=True) # m3/?hrs total volume growth
            
            # fix attributes 
            delVsum[var].attrs = {'units':'m2'}
            
        return  delVsum        
        

    def get_total_growth(self, dVol):
        '''
        Calculates the total amount of ice volume growth
            Parameters:
                dVol (xr.DataArray): ice growth 

            Returns:
                total_growth (dict):  Cumulative ice volume growth (m3) 
        '''

        # group by year
        yearly = dVol.groupby('time.year')

        total_growth = {}
        for var in self.variables:
            csum=[]
            for year, darr in yearly:
                # cumulative sum
                csum.append(darr[var].cumsum(axis=0).values[-1])

            total_growth[var] = np.array(np.float32(csum))    
        return total_growth
    
    
    def weighted_volume_growth(fraction_mask, var, mask):
            # OBS!! NEEDS FIXING
        # total growth in terms of volume (m3)
        dx=5*1000 # 5 km res
        dy=dx
        cellarea = dx*dy

        leadfraction_area = cellarea*fraction_mask # area of the cell which is lead or pack ice

        vol = leadfraction_area*var # ice growth in each cell multiplied by area (m3)

        # apply mask
        vol = vol.where(mask==1)
        vol_weighted = vol.sum(axis=(1,2)) # m3
        #vol_weighted = vol.sum(axis=(1,2))/leadfraction_area.sum(axis=(1,2)) # divide by total area of lead or pack ice

        return vol_weighted