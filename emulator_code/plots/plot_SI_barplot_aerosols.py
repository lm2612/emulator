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
from RegionLatitudes import *
from DefineRegions import *
from AreaWeighting import *
from matplotlib.patches import Patch


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
level_max = [1., .1, .051, 0.01, 0.051, 0.051, 0.01, 0.051, 0.051] 
X_save = ['CO2_Global', 'CH4_Global', 'SO2_Europe', 'SO2_North_America', 'SO2_East_Asia',
            'SO2_South_Asia', 'SO2_South_America', 'SO2_Africa', 'OCBC_Tropics']
X_labels_condensed = ['CO$_2$', 'CH$_4$', 'SO$_2$\nEur', 'SO$_2$\nNAm', 'SO$_2$\nEAs',
                     'SO$_2$\nSAs', 'SO$_2$\nSAm', 'SO$_2$\nAfr', 'OC/BC\nTro']
weights = np.zeros((Si1_map.shape))
x = range(9)
regions = ['Global', 'Europe', 'North_America', 'East_Asia', 'South_Asia', 'Africa', 'South_America', 'Tropics']

area = Area(longitude, latitude)
my_cmap = plt.get_cmap("turbo")

# AEROSOLS ONLY
X_labels_condensed = X_labels_condensed[2:]
Si1_map = Si1_map[2:,:,:]
SiT_map = SiT_map[2:,:,:]
x = x[2:]
weights = np.zeros((Si1_map.shape))

plt.clf()
fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(14, 6), gridspec_kw = {'wspace':0.0, 'hspace':0.0})
axs_flat = axs.flatten()
for i in range(len(regions)):
    region = regions[i]
    grid = DefineRegion(region, longitude, latitude, regiondict=RegionLonsLats)
    weights[:, :, :] = area*grid

    plt.sca(axs_flat[i])
    SiT = np.average(SiT_map, axis=(1,2), weights=weights)
    Si1 = np.average(Si1_map, axis=(1,2), weights=weights)

    plt.bar(x, Si1, color=my_cmap((np.array(x)-2.)/(10)) )
    plt.xticks(x, X_labels_condensed, fontsize=12)
    plt.axis(ymax = 0.1, ymin = -0.01)
    plt.title(region.replace('_', ' '), y=0.87)

plt.tight_layout()
plt.savefig('../../sensitivity/Si1_bar_all_aero.png')


X_labels_full = ['CO$_2$', 'CH$_4$', 'SO$_2$ Europe', 'SO$_2$ North America', 'SO$_2$ East Asia',
                     'SO$_2$ South Asia', 'SO$_2$ South America', 'SO$_2$ Africa', 'BB OC/BC Tropics']
plt.clf()
fig, axs = plt.subplots(1,1, figsize=(14,2))
colors = my_cmap((np.array(x)-2.)/(10))

legend_elements = [Patch(facecolor=colors[i], label=X_labels_full[2:][i]) for i in range(7)]
plt.axis('off')
plt.legend(handles=legend_elements, bbox_to_anchor=(1., 1.), ncol=4, fontsize=14)
plt.savefig('../../sensitivity/Si1_labels_aero.png')
