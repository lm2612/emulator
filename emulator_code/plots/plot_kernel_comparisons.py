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

name = 'indepdims_GP'

kernels = ['Linear','Poly','RBF','RBF_plus_linear','Matern32','Matern32_plus_linear','Matern52','Matern52_plus_linear']

# Get file
_, _, _, _, latitude, longitude = get_train_test()

regions = ['Global', 'Europe', 'North_America', 'East_Asia', 'South_Asia', 'Africa', 'South_America', 'Tropics', 'GridBox']
n_regions = len(regions)
n_kernels = len(kernels)
r_squareds = np.zeros((n_kernels, 18))
MADs = np.zeros((n_kernels, 18))
within_err = np.zeros((n_kernels, 18))

regions=['Global']
for k, kernel in enumerate(kernels):
    filename = '../../emulator_output/kernel_comparison/{}_{}.file'.format(name_file, kernel)
    output = pickle.load(file=open(filename,"rb"))
    ypred = output['ypred']
    ytest = output['ytest']
    var = output['sd']
    # Array sizes 
    print(ytest.shape)
    N = ypred.shape[0]
    nlat = ypred.shape[1]
    nlon = ypred.shape[2]

    # Regional grid
    # Take regional means to explore linearity for these 
    area = Area(longitude, latitude)
    areaN = np.tile(area,(N,1,1))

    #for grid box mean
    grid = DefineRegion("Global", longitude, latitude, regiondict=RegionLonsLats)
    i = n_regions - 1
    inds = list(range(0,12)) + list(range(14,20))
    MAD  = np.average(np.abs(ytest[inds] - ypred[inds]), weights=grid*area, axis=(1,2) )
    MADs[k, :] = MAD 
    inds = list(range(0,12)) + list(range(14,20))

    print((ytest.shape[0]-2)*ytest.shape[1]*ytest.shape[2])
    print(ytest.shape)
    print(144*192)
    print(np.sum(np.where(np.abs(ytest[inds] - ypred[inds]) < np.sqrt(var[inds]))))
    within_err[k, :] = np.sum((np.abs(ytest[inds,:,:] - ypred[inds,:,:]) < np.sqrt(var[inds,:,:])),axis=(1,2) ) / ( (ytest.shape[0]-2)*ytest.shape[1]*ytest.shape[2] )

print(kernels)
print(within_err)
print(MADs)
plt.clf()
# Plot probability of correct result
within_err_all = within_err[:, -1]
x = np.arange(1,19)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
plt.sca(ax)
plt.scatter(x, within_err_all[0,:], color="red")
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.scatter(x, MADs[0,:], color="blue",marker="o")
ax2.set_ylabel("mean abs difference",color="blue",fontsize=14)

saveas = '../../GPplots/kernel_comparison.png'
plt.savefig(saveas)
print("Plotted as {}".format(saveas))
plt.close()
