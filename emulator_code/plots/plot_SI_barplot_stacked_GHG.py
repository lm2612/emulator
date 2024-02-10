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
print(np.min(np.sum(SiT_map[:], axis=0)))
print(SiT_map.shape)

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
X_labels_condensed = X_labels_condensed[:2]  + ['Aerosols']
#Si1_map = Si1_map[:3,:,:]
#Si1_map[2, :, :] = np.sum(Si1_map[2:, :, :], axis=0)
#SiT_map = SiT_map[:3,:,:]
#SiT_map[2, :, :] = np.sum(SiT_map[2:, :, :], axis=0)

x = x[:3]
weights = np.zeros((Si1_map.shape))

plt.clf()
plt.figure(figsize=(6,4))

x = range(len(regions))

for i in range(len(regions)):
    region = regions[i]
    xi = x[i]
    grid = DefineRegion(region, longitude, latitude, regiondict=RegionLonsLats)
    weights[:, :, :] = area*grid

    SiT = (np.average(SiT_map, axis=(1,2), weights=weights))
    Si1 = (np.average(Si1_map, axis=(1,2), weights=weights))
    Si2 = SiT - Si1
    plt.bar(xi, Si1[0],color='red'  )
    plt.bar(xi, Si1[1], color='orange', bottom=Si1[0])
    plt.bar(xi, np.sum(Si1[2:]), color='blue', bottom=Si1[1]+Si1[0])
    print(1.0 - np.sum(Si1))

plt.xticks(x, [region.replace('_','\n') for region in regions])
plt.axis(ymax = 1.05, ymin = -0.05)
legend_elements = [Patch(facecolor='red', label='CO$_2$'),
                   Patch(facecolor='orange', label='CH$_4$'),
                   Patch(facecolor='blue', label='Aerosols')]
# Create the figure
plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.))


plt.tight_layout()
plt.savefig('../../sensitivity/Si1_bar_all_ghg_stacked.png')



