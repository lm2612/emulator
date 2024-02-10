import sys
import os
home = os.getenv("HOME")
sys.path.insert(0, '../')
sys.path.insert(0, '../setup/')

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from read_file import *
from AreaWeighting import *
from RegionLatitudes import *
from DefineRegions import *

name_file = 'indepdims_GP_with_internalnoise'
name = 'indepdims_GP'

# Get file
_, _, _, _, latitude, longitude = get_train_test()

filename = '../../emulator_output/{}.file'.format(name_file)
output = pickle.load(file=open(filename,"rb"))
ypred = output['ypred']
ytest = output['ytest']
var = output['sd']
# Array sizes 
N = ypred.shape[0]
nlat = ypred.shape[1]
nlon = ypred.shape[2]

# Regional grid
# Take regional means to explore linearity for these 
area = Area(longitude, latitude)
areaN = np.tile(area,(N,1,1))

regions = ['Global', 'Europe', 'North_America', 'East_Asia', 'South_Asia', 'Africa', 'South_America', 'Tropics']
colors = ['grey', 'red', 'blue', 'green', 'darkgreen', 'orange', 'purple', 'darkblue', 'darkred', 'teal']*2
markers=['o', '*', '^', 's', 'd', 'h', '<', 'p', '>', 'X']

plt.clf()
fig, axs = plt.subplots(nrows = 2, ncols=4, sharex=True, sharey=True,figsize=(10, 6), gridspec_kw = {'wspace':0.0, 'hspace':0.0})
axs = axs.flatten()
for i in range(len(regions)):
    region = regions[i]
    grid = DefineRegion(region, longitude, latitude, regiondict=RegionLonsLats)
    gridN = np.tile(grid, (N,1,1))
    
    gridflat = grid.flatten()
    
    #ytest = np.zeros((Ytest.shape[0], sum(grid==1)))
    #ypred = np.zeros((Ytest.shape[0], sum(grid==1)))

    #ytest[:, :]  = Ytest_flat[:,grid==1]
    #ypred[:, :]  = Ypred_flat[:,grid==1]

    ax = axs[i]

    X = np.zeros(ytest.shape[0]-2)
    Y = np.zeros(ytest.shape[0]-2)
    s = 0
    plt.sca(ax)
    for j in range(ytest.shape[0]):
        if (j in [12,13]):
             continue
        ytest_j = np.average(ytest[j], weights=grid*area)
        ypred_j = np.average(ypred[j], weights=grid*area)
        X[s] = ytest_j
        Y[s] = ypred_j
        sd_j = np.sqrt(np.average(var[j], weights=grid*area))
        #sd_j = np.sqrt(np.average(sd[j]**2, weights=(grid*area)**2 )/np.sum(grid*area))
        plt.errorbar(ytest_j, ypred_j, c='red', yerr = sd_j, fmt='x', alpha=0.7)
        s+= 1
    #runnames = ['Test'+str(j) for j in range(1,11)]
    #legend_elements = [Line2D([0], [0], marker='x', color=colors[j], label=runnames[j]) for j in range(len(runnames))]
    #plt.legend(handles=legend_elements)
    plt.plot([-10.,10.],[-10.,10.],'k--', zorder=100)
    correlation_matrix = np.corrcoef(X,Y)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print("R2", r_squared)
    MAD = np.mean(np.abs(X-Y))

    plt.title(region.replace('_',' '), y = 0.9)
    plt.text(-2.5, 1.5, ' R$^2=${:.2f}\n MAD={:.2f}$\degree$C'.format(r_squared, MAD), fontsize=9)
    plt.axis(ymin=-2.5, xmin=-2.5, ymax=2.5, xmax=2.5)
    #if (i>=4):
    #    plt.xlabel('True Response ($\degree$C)')
    #if (i%4==0):
    #    plt.ylabel('Predicted Response ($\degree$C)')
    if i==0:
         plt.text(-3.5, -2.5, 'Predicted Response ($\degree$C)', rotation='vertical', va='center', ha='center', fontsize=14)
    if i==6:
         plt.text(-2.5, -3.5, 'True Response ($\degree$C)',va='center', ha='center', fontsize=14)
    

plt.tight_layout()


saveas = '../../GPplots/XY_{}_AllRegionalMeans.png'.format(name)
plt.savefig(saveas)
print("Plotted as {}".format(saveas))
plt.close()
    
