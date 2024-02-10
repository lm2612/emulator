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
from AreaWeighting import *
name = 'indepdims_GP_with_internalnoise'

output = pickle.load(file=open('../../emulator_output/GP_SSPs_all_2014-2100.file',"rb"))
ypred = output['ypred']
ytest = output['ytest']
print(ypred)
sd = np.sqrt(output['sd'])
sd_GP = output['sd_GP']
print(sd)
print(sd_GP)
# Get file
_, _, _, _, latitude, longitude = get_train_test()
temps_all = (np.genfromtxt('../../ssp_scenarios/Temp_ssps.csv', delimiter=',',
                 skip_header=1, usecols=range(1,12)))
ssp_temps = temps_all[0:5, 2:] - temps_all[0:5, 2][:, np.newaxis]
print(ssp_temps[1,:])
magicc119 = ssp_temps[0]
magicc370 = ssp_temps[2]
magicc585 = ssp_temps[3] 

temps_all = (np.genfromtxt('../../ssp_scenarios/other_ssp_estimates.csv', delimiter=',',
                 skip_header=1, usecols=range(1,6)))
print(temps_all)
plt.clf()

MeanSd = np.mean(sd, axis=0)
print(np.mean(MeanSd))

plt.rcParams['font.size'] = '14'

N = ypred.shape[0]
print(N)

ssp_years = range(2020, 2101, 10)

years = range(2025, 2106, 10)
years = ssp_years
nyears = len(years)

ssps = ['ssp119', 'ssp245', 'ssp370', 'ssp434', 'ssp585']

ssp_ind = 0
yr = 0
temps = np.zeros((nyears))
errs = np.zeros((nyears))
area = Area(longitude, latitude)


for i in range(N): 
    ssp = ssps[ssp_ind]
    print(ssp)
    year = years[yr]
    print(year)
    y = ypred[i] 
    err = sd[i]
    temp = np.average(y, weights=area) 
    err2 = np.sqrt(np.average(err**2, weights=area))
    if year == 2050:    
        if ssp_ind == 0:
            ii=0
            ymax = 2.5
        elif ssp_ind == 2:
            ii=1
            ymax = 2.5
        elif ssp_ind == 4:
            ii=2
            ymax = 2.5
        else: 
            yr += 1
            continue
    
        plt.clf()
        plt.figure(figsize=(6,6))
        plt.axhline(y=0, linestyle='--', color='grey',alpha=0.5)
        #plt.plot([1], ssp_temps[ssp_ind,yr], color='black', marker = 'x', label='MAGICC', markersize=16)
        plt.errorbar([.8], temp, yerr=err2, color='red', capsize=12)
        plt.scatter([.8], temp, color='red', marker='x', label='GP emulator', s=100)
        #plt.plot([1.5],temps_all[ii,3] , marker='x', color='orange',label='FAIR', markersize=16)
        plt.scatter([1.2],temps_all[ii,4] , marker='x', color='black',label='FAIR', s=100)
        plt.errorbar([3],temps_all[ii,0], yerr=temps_all[ii,1]-temps_all[ii,0], color='blue', capsize=12 )
        plt.scatter([3],temps_all[ii,0], color='blue',label='CMIP6',   marker='x',s=100)

        saveas = '../../SSPplots/GlobMean_fair_{}_{}'.format(ssp, year)
        plt.legend()
        print(saveas)
        plt.ylabel('$\degree$C')
        plt.tight_layout()
        plt.axis(ymin=-.5, ymax=ymax, xmin = 0., xmax=4.)
        plt.xticks([1,3],['Fast adjustment','Transient response'])
        plt.savefig(saveas, pad_inches=0.1)
    yr += 1
    if year == 2100:
        ssp_ind += 1
        yr = 0
