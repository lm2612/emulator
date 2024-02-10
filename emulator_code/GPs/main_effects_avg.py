import sys
import os
home = os.getenv("HOME")
sys.path.insert(0, '../')
sys.path.insert(0, '../setup/')

### IMPORT PACKAGES ####
import GPy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
from sklearn.decomposition import PCA
import multiprocessing as mp

### MY PYTHON FILES ####
from read_file import *
from AreaWeighting import *
from RegionLatitudes import *
from DefineRegions import *
from conv_MMR_ppm import *

### DEFINE RANDOM SEED ####
random_seed = 1

output_filename = '../../emulator_files/AllTemps1-80.nc'
nyears = 5
filekeys, t, latitude, longitude, temp = open_file(output_filename, 'temps')
sorted_inds = np.argsort(filekeys)
print(filekeys)
sorted_keys = filekeys[sorted_inds]
sorted_temps = temp[sorted_inds, :nyears, :, :]

# Average over time dimension (dim=1)
average_temps = np.average(sorted_temps, axis=1)
temps = average_temps


# Internal variability
ctrl_keys = sorted_keys[0:6]
ctrl_temps = temps[0:6, :, :]
output_filename = '../../emulator_files/ControlTemps.nc'
filekeys, t, latitude, longitude, extra_ctrl_temps = open_file(output_filename, 'temps')

extra_ctrl_temps = np.average(extra_ctrl_temps[0:8], axis=1)
ctrl_temps_all = np.zeros((6+8, ctrl_temps.shape[1], ctrl_temps.shape[2]))
ctrl_temps_all[0:6,:,:] = ctrl_temps
ctrl_temps_all[6:,:,:] = extra_ctrl_temps

internal_var = np.std(ctrl_temps_all, axis=0)
internal_var = internal_var.flatten().reshape(1, -1)

### GET FILES ####
X, Y, Xtest, Ytest, latitude, longitude = get_train_test()

nlat, nlon = len(latitude), len(longitude)

### PREPROCESS X ####
scalerX = preprocessing.StandardScaler()
scalerX.fit(X)
X = scalerX.transform(X)

N = X.shape[0]
p = X.shape[1]

# Set up random array for main effects, must be centered at present day levels (base_vals)
# With Gaussian distribution with preset s.d. [not uniform distribution]
min_vals = [CO2_ppm_MMR(282) , CH4_ppb_MMR(248) ,  0, 0, 0, 0, 0, 0, 0]
base_vals = [CO2_ppm_MMR(410), CH4_ppb_MMR(1862),  1, 1, 1, 1, 1, 1, 1]
max_vals = [CO2_ppm_MMR(834), CH4_ppb_MMR(3238), 5, 3, 2, 3, 3, 7, 2]
sigma_vals = [CO2_ppm_MMR(50), CH4_ppb_MMR(100), .5, .5, .5, .5, .5, .5, .5]

NFuncs = 100
N = NFuncs
grid_size = 20
Xtest_all = np.zeros((9 , N * grid_size, 9))
for j in range(9):
  # Set up Xtest_all[j] with N randomly selected values across all grid points
  Xtest = np.zeros((N, grid_size, 9))
  for i in range(9):
    min_i, base_i, max_i = min_vals[i], base_vals[i], max_vals[i]
    if i==j:
        grid = np.linspace(min_i, max_i, grid_size)
        # Repeat for all N function replications
        for s in range(N):
            Xtest[s, :, i] = grid
    else: 
        # Random for all N function replications, centered on base_val, with sd = distance from min to max
        sigma_i = sigma_vals[i]
        Nreplications = np.random.randn(N)*sigma_i + base_i
        for g in range(grid_size):
            Xtest[:, g, i] = Nreplications 
        # Or uniform
        #Xtest[:, :, i] = np.random.rand(N, grid_size)*(max_i-min_i) + min_i

  # Condense into N * grid_size
  Xtest_scaled = scalerX.transform(Xtest.reshape(N*grid_size, 9))
  Xtest_all[j, :, :] = Xtest_scaled
Xtest = Xtest_all.reshape((9*N*grid_size, 9))

### SCALE Y ###
Yfull = Y
# reshape
Y = Yfull.reshape(Yfull.shape[0], Yfull.shape[1]*Yfull.shape[2])


# scale
scalerY = preprocessing.StandardScaler()
scalerY.fit(Y)
y = scalerY.transform(Y)
ctrl_temps = ctrl_temps.reshape(ctrl_temps.shape[0], Yfull.shape[1]*Yfull.shape[2])

ctrl_mean = np.mean(ctrl_temps, axis=0)
ctrl_temps = ctrl_temps - ctrl_mean
ctrl_scaled = scalerY.transform(ctrl_temps)
y_internal = np.std(ctrl_scaled, axis=0)


def predict_function(i, y, ypred, sd, sd_gp, X, Xtest):
    y_i = y[:, i]
    N = len(y_i)
    y_i = y_i.reshape((N, 1))

    ### BUILD MODEL ###
    kern1 = GPy.kern.RBF(p, ARD=True)
    kern1.variance = 1e-5
    kern1.lengthscale = [0.89, 0.83, 0.94, 1.11, 0.37, 1., 0.96, 0.93, 0.75]
    kern2 = GPy.kern.Linear(p,ARD=True)
    kern2.variances = [0.63, 0.03, 0.01, 0.001, 0.007, 0.006, 0.003, 0.037, 0.015]
    kern = kern1 + kern2
    m = GPy.models.GPRegression(X, y_i, kern)
    m.likelihood.variance.fix(y_internal[i])
    m.optimize('bfgs', max_iters=1000)

    ### PREDICT ####
    ypred_i, sd_full_i = m.predict(Xtest)
    _, sd_GP_i = m.predict_noiseless(Xtest)
    Ntest = len(ypred_i)
    ypred[(i*Ntest):(i+1)*Ntest] = ypred_i.flatten()
    sd[(i*Ntest):(i+1)*Ntest] = sd_full_i.flatten()
    sd_gp[(i*Ntest):(i+1)*Ntest] = sd_GP_i.flatten()


if __name__ == '__main__':
    # Initialise ypred and sd 
    ypred = np.zeros((Xtest.shape[0],  Yfull.shape[1]*Yfull.shape[2]))
    sd = np.zeros((ypred.shape))
    sd_gp = np.zeros((ypred.shape))

    N = Xtest.shape[0]
    k = ypred.shape[1]
    print(k)
    # set up arrays
    inds = mp.Array('i', range(k))
    ypred_arr = mp.sharedctypes.RawArray('d', N*k)
    sd_arr = mp.sharedctypes.RawArray('d', N*k)
    sd_gp_arr = mp.sharedctypes.RawArray('d', N*k)
    jobs = []
    for i in range(k):
        proc= mp.Process(target=predict_function, args=(i, y, ypred_arr, sd_arr, sd_gp_arr, X, Xtest))
        jobs.append(proc)
        proc.start()
    for j in jobs:
        j.join()
    print("Run all indices complete")
    print(ypred_arr)
    print(sd_arr)
    print(sd_gp_arr)
    Ntest= N

    for i in range(k):
        ypred[:, i] = ypred_arr[(i*Ntest):(i+1)*Ntest] #np.frombuffer(ypred_arr[i:(i+N)])
        sd[:, i] = sd_arr[(i*Ntest):(i+1)*Ntest] 
        sd_gp[:,i] = sd_gp_arr[(i*Ntest):(i+1)*Ntest]

    Ypred = scalerY.inverse_transform(ypred)
    Ypred_flat = Ypred
    Y_p_Sd = scalerY.inverse_transform(sd+ypred)
    Y_m_Sd = scalerY.inverse_transform(ypred-sd)
    Sd = Y_p_Sd - Ypred
    Sd_GP = scalerY.inverse_transform(sd_gp+ypred) - Ypred

    # un-shape
    ypred = Ypred.reshape(Ypred.shape[0], Yfull.shape[1], Yfull.shape[2])
    sd = Sd.reshape(sd.shape[0], Yfull.shape[1], Yfull.shape[2])
    sd_gp = Sd_GP.reshape(sd.shape[0], Yfull.shape[1], Yfull.shape[2])    
 
    # Avg over N replications
    N = NFuncs
    print(ypred.shape[0])
    # should be same as
    print(N * grid_size * 9)
    y_reshaped = ypred.reshape((9, N*grid_size, Yfull.shape[1], Yfull.shape[2]))
    y_reshaped2 = y_reshaped.reshape((9, N, grid_size, Yfull.shape[1], Yfull.shape[2]))
    y_avg = np.mean(y_reshaped2, axis=1)
    y_sd = np.std(y_reshaped2, axis=1)
    print("Saving")

    ### SAVE MODEL/PREDICTIONS ####
    save_file = '../../emulator_outputs/maineffects_avg.file'
    #save_objs = {'ypred':ypred, 'Xtest':Xtest, 'sd':sd, 'sd_GP':sd_gp}
    save_objs = {'y_avg':y_avg, 'y_sd': y_sd, 'y_full':y_reshaped[:,0:20]}
    pickle.dump(save_objs, file=open(save_file, "wb"))
    print("Saved objs in ", save_file)
    print("DONE")
    exit()
