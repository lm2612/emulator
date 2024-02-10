import sys
import os
home = os.getenv("HOME")
sys.path.insert(0, '../')
sys.path.insert(0, '../setup/')
sys.path.insert(0, '../plotting/')

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from read_file import *
from plotmapfunction import *
from conv_MMR_ppm import *
from X_labels import *
name = 'indepdims_GP_with_internalnoise'

output = pickle.load(file=open('../../emulator_output/indepdims_GP_with_internalnoise.file',"rb"))
ypred = output['ypred']
ytest = output['ytest']
sd = np.sqrt(output['sd'])
sd_GP = output['sd_GP']
print(sd)
print(sd_GP)
# Get file
_, _, X_test, _, latitude, longitude = get_train_test()

plt.clf()
MeanSd = np.mean(sd, axis=0)
print(np.mean(MeanSd))

N = ypred.shape[0]

max_vals = [CO2_ppm_MMR(834), CH4_ppb_MMR(3238), 10, 6, 4, 6, 6, 14, 4]

min_vals = [CO2_ppm_MMR(282) , CH4_ppb_MMR(248) ,  0, 0, 0, 0, 0, 0, 0]
base_vals = [CO2_ppm_MMR(410), CH4_ppb_MMR(1862),  1, 1, 1, 1, 1, 1, 1]
max_vals = [CO2_ppm_MMR(834), CH4_ppb_MMR(3238), 5, 3, 4, 3, 3, 7, 2]

min_vals = [282 , 248 ,  0, 0, 0, 0, 0, 0, 0]
base_vals = [410, 1862,  1, 1, 1, 1, 1, 1, 1]
max_vals = [834, 3238, 5, 3, 4, 3, 3, 7, 2]
X_test[:,0] = CO2_MMR_ppm(X_test[:,0])
X_test[:,1] = CH4_MMR_ppb(X_test[:,1])
X_labels[0] = 'CO$_{2}$ Global (ppm)'
X_labels[1] =  'CH$_4$ Global (ppb)'
plt.rcParams['axes.labelsize'] = '14'
plt.rcParams['font.size'] = '12'

# The two maps are: Test1 (roughly the median in terms of performance based on approx 78percent of grid cells being within 1 s.d.)
#                   Test11 (the worst performance, significantly worse than all others)
# Will plot Test1 in blue and Test11 in red
all_test_indices = [[0, 10], [1,2], [3,4], [5,6], [7,8], [9,11], [12,13], [14,15], [16,17]]
all_names = [ ['Example A', 'Example B'],
              ['Example C', 'Example D'],
              ['Example E', 'Example F'],
              ['Example G', 'Example H'],
              ['Example I', 'Example J'],
              ['Example K', 'Example L'],
              ['Example M', 'Example N'],
              ['Example O', 'Example P'],
              ['Example Q', 'Example R'],
              ['Example S', 'Example T'],
              ['Example U', 'Example V']]


pairs = range(0,8)

pair = 8


test_indices = all_test_indices[pair]
color_diamonds = ['blue', 'red']
name = all_names[pair]

plt.clf()
nrows = 3
ncols = 3
fig, axs = plt.subplots(nrows, ncols,figsize=(10, 2.5), gridspec_kw = {'wspace':0.15, 'hspace':0.1}, sharey=True)
axs_flat = axs.flatten()    

## First plot inputs on axis for all 9 diff parameters

for ii in range(len(test_indices)):
    i = test_indices[ii]
    for j in range(9):
        ax = axs_flat[j]
        plt.sca(ax)
        xmin = min_vals[j]
        xmax = max_vals[j]
        xmid = base_vals[j]
        # Set up axis
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_position('zero')        
        ax.get_xaxis().tick_bottom()
        ax.axes.get_yaxis().set_visible(False)
        # Add input values
        ax.scatter(X_test[i,j], [0.], color=color_diamonds[ii], marker="d")
        plt.axis(xmin=xmin, xmax=xmax)
        ax.set_xlabel(X_labels[j],labelpad = -44)
        # Add min, max and mid (baseline) for reference
        plt.xticks([xmin, xmid, xmax], [xmin, xmid, xmax])
        # Ticks
        ax.tick_params(direction='inout', which='major', width=2, length=10)
        if j == 0:
            ax.set_xticks(np.arange(300, 830, 100), minor=True)
        elif j == 1:
            ax.set_xticks(np.arange(400, 3200, 400), minor=True)

        else:
            ax.set_xticks(np.arange(0, xmax, 1), minor=True)
        ax.tick_params(direction='inout', which='minor', width=1, length=4)
        
  
saveas = '../../GPplots/Inputs_Pair{}'.format(pair)
plt.tight_layout()
plt.savefig(saveas, pad_inches=0.01, bbox_inches = 'tight')
plt.close()
print("Saved as ", saveas)

## Next, plot maps
for ii in range(len(test_indices)):
    plt.clf()
    fig = plt.figure(figsize=(6, 10))
    ax0 = plt.subplot(311,projection=ccrs.PlateCarree())
    ax1 = plt.subplot(312,projection=ccrs.PlateCarree())
    ax2 = plt.subplot(313,projection=ccrs.PlateCarree())

    i = test_indices[ii]

    levels = np.arange(-2., 2.1, 0.1)
    y = ytest[i]
    print(y.shape)
    plt.sca(ax0)
    plotmap(longitude,latitude,y,savefile=None,levels = levels,
            variable_label='',plottitle='$y_{test}$',plotaxis=ax0,colorbar=0,alpha=1)

    ticks = np.arange(-2., 2.1, 0.5)
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(levels[0], levels[-1]), cmap='RdBu_r')
    sm._A = []

    cbar_ax = fig.add_axes([1.0, 0.655, 0.02, 0.27 ])
    cbar = plt.colorbar(sm, cax=cbar_ax,orientation='vertical',ticks=ticks)
    cbar.set_label(r'$\degree$C')

    y = ypred[i]
    print(y.shape)
    #ax1 = plt.subplot2grid((2, 2), (0, 1))
    plt.sca(ax1)
    plotmap(longitude,latitude,y,savefile=None,levels = levels,
            variable_label='',plottitle='$y_{pred}$',plotaxis=ax1,colorbar=0,alpha=1)
    ticks = np.arange(-2., 2.1, 0.5)
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(levels[0], levels[-1]), cmap='RdBu_r')
    sm._A = []

    cbar_ax = fig.add_axes([1.0, 0.34, 0.02, 0.27 ])
    cbar = plt.colorbar(sm, cax=cbar_ax,orientation='vertical',ticks=ticks)
    cbar.set_label(r'$\degree$C')


    
    plt.sca(ax2)

    yt = ytest[i]
    yp = ypred[i]
    diff = np.abs(yt - yp)

    levels = np.arange(0., 1.25, 0.05)
    # Plot map of differences
    plotmap(longitude, latitude, diff, savefile=None, levels = levels, cmap='Reds',
            plotaxis=ax2, colorbar=0, alpha=1, plottitle='$|y_{pred}-y_{test}|$')

    # Add hatching when diff > 1 s.d.
    hatching = diff - sd[i]
    mpl.rcParams['hatch.linewidth'] = 0.2 
    add_hatching(longitude, latitude, hatching, ax2)

    # Option to add secondary hatching when diff > 2s.d.
    # hatching = diff - 2*sd[i]
    # add_hatching(longitude, latitude, hatching, ax2,'o.o.')

    # Add colorbar
    ticks = np.arange(0., 1.21, 0.2)
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(levels[0], levels[-1]), cmap='Reds')
    sm._A = []
    ###cbar_ax = fig.add_axes([0.98, 0.05, 0.01, 0.85 ])
    cbar_ax = fig.add_axes([1.0, 0.02, 0.02, 0.27])

    cbar = plt.colorbar(sm, cax=cbar_ax,orientation='vertical',ticks=ticks)
    cbar.set_label(r'$\degree$C')

    # Save
    saveas = '../../GPplots/AllMaps_pair{}_{}'.format(pair,i+1)
    plt.suptitle(name[ii], fontdict = {'color': color_diamonds[ii]}) 
    plt.tight_layout()
    plt.savefig(saveas, pad_inches=0.15, bbox_inches = 'tight')
    print("Saved as ", saveas)
    plt.close()
