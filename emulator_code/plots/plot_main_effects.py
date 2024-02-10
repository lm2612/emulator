import sys
import os
home = os.getenv("HOME")
sys.path.insert(0, '../')
sys.path.insert(0, '../setup/')
sys.path.insert(0, '../plotting/')

import GPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
from sklearn.decomposition import PCA
import pickle

from plotmapfunction import *
from create_colormap import *
from read_file import *
from AreaWeighting import *
from RegionLatitudes import *
from DefineRegions import *
from conv_MMR_ppm import *
from X_labels import *
### DEFINE RANDOM SEED ####
random_seed = 1

### GET FILES ####
X, Y, Xtest, Ytest, latitude, longitude = get_train_test()

nlat, nlon = len(latitude), len(longitude)
scalerX = preprocessing.StandardScaler()
scalerX.fit(X)

# PICK INPUTS
X = X[:, :]
Xtest = Xtest[:, :]
print(X.shape)
print(Y.shape)
print(Xtest.shape)
print(Ytest.shape)

filename = '../../emulator_output/maineffects_avg.file'
maineffects = pickle.load(open(filename, 'rb'))
print(maineffects.keys())
y = maineffects['y_avg']
sd = (maineffects['y_sd'])
print(y.shape)
print(y)
# Regional grid
# Take regional means to explore linearity for these 
area = Area(longitude, latitude)
regions = ['Global', 'Europe', 'North_America', 'East_Asia', 'South_Asia', 'Africa', 'South_America', 'Tropics']
colours = ['grey', 'red', 'blue', 'green', 'darkgreen', 'orange', 'purple', 'darkblue', 'darkred', 'teal']
markers=['o', '*', '^', 's', 'd', 'h', '<', 'p', '>', 'X']

# Resolution N
max_vals = [CO2_ppm_MMR(834), CH4_ppb_MMR(3238), 10, 6, 4, 6, 6, 14, 4]
min_vals = [CO2_ppm_MMR(282) , CH4_ppb_MMR(248) ,  0, 0, 0, 0, 0, 0, 0]
base_vals = [CO2_ppm_MMR(410), CH4_ppb_MMR(1862),  1, 1, 1, 1, 1, 1, 1]

pollutants = ['CO2', 'CH4', 'SO2_Europe',  'SO2_NorthAmerica',  'SO2_EastAsia',  'SO2_SouthAsia',  'SO2_SouthAmerica',  'SO2_Africa',  'OCBC_Tropics']
N = 100

plt.clf()
fig = plt.figure(figsize=(12, 12))

regions = ['Global', 'Europe', 'North_America', 'East_Asia', 'South_Asia', 'South_America', 'Africa']


nrows = len(regions)
ncols = 7
n=1
fig, axs = plt.subplots(nrows, ncols, sharex='col', sharey='row',figsize=(12, 10), gridspec_kw = {'wspace':0.05, 'hspace':0.05})

grid_size = 10
N = 20
print(y.shape)

ypred_all = y
sd_all = sd
for j in range(9):
  min_i, base_i, max_i = min_vals[j], base_vals[j], max_vals[j]
  Xtest = np.linspace(min_i, max_i, N)
  Ypred = ypred_all[j, :, :]
  Sd = sd_all[j, :, :]

  weights = np.zeros(Ypred.shape)

  for region,k in zip(regions, range(len(regions))):
    print(k, region)    
    grid = DefineRegion(region, longitude, latitude, regiondict=RegionLonsLats)
    weights[:, :, :] = area*grid
    ypred = np.average(Ypred, axis=(1,2) , weights=weights)
    sd = np.average(Sd, axis=(1,2) , weights=weights)


    if j < 2:
        continue
    x = Xtest
    
    if j==0:
        x = CO2_MMR_ppm(x)
    elif j==1:
        x = CH4_MMR_ppb(x)
    
    pollutant = pollutants[j]

    ax = axs[k, j-2]
    plt.sca(ax)
    plt.axhline(y=0, linestyle='--', color='grey')

    plt.plot(x, ypred, color='black')
    plt.plot(x, ypred + sd, color='red')
    plt.plot(x, ypred - sd, color='red')
    if (k <=3):
        plt.axis(ymin=-.4, ymax=.10)
        yticks = np.arange(-.4, .1,0.2)
    else: 
        plt.axis(ymin = -.1, ymax=.05)
        yticks = np.arange(-.1, .04,.05)
    plt.axis(ymin=-1., ymax=1.)
    yticks = np.arange(-.5, .55, 0.5)
    #plt.ylabel('Predicted temp')
    if (j==2):
        plt.ylabel('Response in \n {} (K)'.format(region.replace('_',' ')))
        plt.yticks(yticks, yticks)
  
    if (k==len(regions)-1):
        plt.xlabel(X_labels[j])
    #plt.title(r'{} response to change in {}'.format(region.replace('_', ' '), X_labels[j]))
plt.savefig('../../GPplots/Main_effects_aerosols.pdf', bbox_inches='tight')



plt.clf()
nrows = len(regions)
ncols = 2
n=1
fig, axs = plt.subplots(nrows, ncols, sharex='col', sharey='row',figsize=(3, 10), gridspec_kw = {'wspace':0.05, 'hspace':0.05})

for j in range(9):
  min_i, base_i, max_i = min_vals[j], base_vals[j], max_vals[j]
  Xtest = np.linspace(min_i, max_i, N)
  Ypred = ypred_all[j, :, :]
  Sd = sd_all[j, :, :]

  weights = np.zeros(Ypred.shape)

  for region,k in zip(regions, range(len(regions))):
    print(k, region)
    grid = DefineRegion(region, longitude, latitude, regiondict=RegionLonsLats)
    weights[:, :, :] = area*grid
    ypred = np.average(Ypred, axis=(1,2) , weights=weights)
    sd = np.average(Sd, axis=(1,2) , weights=weights)


    if j >= 2:
        continue
    x = Xtest

    if j==0:
        x = CO2_MMR_ppm(x)
    elif j==1:
        x = CH4_MMR_ppb(x)

    pollutant = pollutants[j]

    ax = axs[k, j-2]
    plt.sca(ax)
    plt.axhline(y=0, linestyle='--', color='grey')

    plt.plot(x, ypred, color='black')
    plt.plot(x, ypred + sd, color='red')
    plt.plot(x, ypred - sd, color='red')
    if (k <=3):
        plt.axis(ymin=-.4, ymax=.10)
        yticks = np.arange(-.4, .1,0.2)
    else:
        plt.axis(ymin = -.1, ymax=.05)
        yticks = np.arange(-.1, .04,.05)
    plt.axis(ymin=-1.5, ymax=1.5)
    yticks = [-1., 0., 1.]
    #plt.ylabel('Predicted temp')
    if (j==0):
        plt.ylabel('Response in \n {} (K)'.format(region.replace('_',' ')))
        plt.yticks(yticks, yticks)

    if (k==len(regions)-1):
        plt.xlabel(X_labels[j])
    #plt.title(r'{} response to change in {}'.format(region.replace('_', ' '), X_labels[j]))
plt.savefig('../../GPplots/Main_effects_GHGs.pdf', bbox_inches='tight')

