import sys
import os
home = os.getenv("HOME")
sys.path.insert(0, '../')
sys.path.insert(0, '../setup/')
sys.path.insert(0, '../plotting/')

import numpy as np
import pickle
import matplotlib.pyplot as plt
from read_file import *
from plotmapfunction import *
from SALib.analyze import sobol
from X_labels import *
import cartopy.crs as ccrs
#ex = '2xCO2bound'
output = pickle.load(file=open('../../emulator_output/Sobolmap.file', "rb"))
Si1_map = output['Si1_map']
SiT_map = output['SiT_map']

# Get file
_, _, _, _, latitude, longitude = get_train_test()

print(Si1_map)
print(Si1_map[:,0,0])
print(Si1_map.shape)
level_max = [1., .5, .101, .051, .101, .101, .051, .101, .101]

level_min = [0.5] + [0.]*8
level_max = [1., .1, .051, 0.0051, 0.051, 0.051, 0.0051, 0.1, 0.051] 
X_save = ['CO2_Global', 'CH4_Global', 'SO2_Europe', 'SO2_North_America', 'SO2_East_Asia',
            'SO2_South_Asia', 'SO2_South_America', 'SO2_Africa', 'OCBC_Tropics']
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
plt.clf()
nrows=3
ncols=3
fig, axs = plt.subplots(nrows, ncols,figsize=(25, 10), gridspec_kw = {'wspace':0.13, 'hspace':0.2}, 
                        subplot_kw={'projection':ccrs.PlateCarree()} )
axs_flat = axs.flatten()
"""
print(axs_flat)
plt.clf()
fig = plt.figure(figsize=(16, 10))
axs_flat = [plt.subplot(311,projection=ccrs.PlateCarree()), plt.subplot(312,projection=ccrs.PlateCarree()), plt.subplot(313,projection=ccrs.PlateCarree()),
plt.subplot(321,projection=ccrs.PlateCarree()), plt.subplot(322,projection=ccrs.PlateCarree()), plt.subplot(323,projection=ccrs.PlateCarree()),
plt.subplot(331,projection=ccrs.PlateCarree()), plt.subplot(332,projection=ccrs.PlateCarree()), plt.subplot(333,projection=ccrs.PlateCarree()) ]
print(axs_flat)
"""
for n in range(9):
    print(X_labels[n])
    ax = axs_flat[n]
    print(ax)
    plt.sca(ax)

    plotmap(longitude,latitude,Si1_map[n],savefile=None, cmap="viridis_r",
            levels=np.arange(level_min[n], level_max[n], 0.0001),extend='both',
            variable_label='SI(1)',plottitle=X_labels[n],plotaxis=ax,colorbar=1,alpha=1)
    #plt.clf()
    #fig = plt.figure(figsize=(6, 4))       # Large plot
    #saveas = '../../sensitivity/SITmap_{}.png'.format(X_save[n].replace(' ','_'))
    """
    plotmap(longitude,latitude,SiT_map[n],savefile=None, cmap="Reds",levels=np.arange(level_min[n], level_max[n], 0.0001),extend='both',
            variable_label='SI(T)',plottitle='',plotaxis=None,colorbar=1,alpha=1, projection=None)
    plt.tight_layout()
    plt.savefig(saveas,bbox_to_inches='tight')
    plt.close()
    """
saveas = '../../sensitivity/SI1map_all.png'

plt.tight_layout()
plt.savefig(saveas, bbox_to_inches='tight')

