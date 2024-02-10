### This file trains a Gaussian process that uses the internal variability to fix the GP error
# It trains an independent GP per grid cell.
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

# Calc internal variability
ctrl_keys = sorted_keys[0:6]
ctrl_temps = temps[0:6, :, :]
output_filename = '../../emulator_files/ControlTemps.nc'
filekeys, t, latitude, longitude, extra_ctrl_temps = open_file(output_filename, 'temps')
extra_ctrl_temps = np.average(extra_ctrl_temps[0:8], axis=1)

ctrl_temps_all = np.zeros((6+8, ctrl_temps.shape[1], ctrl_temps.shape[2]))
ctrl_temps_all[0:6,:,:] = ctrl_temps            # 6 independent control runs
ctrl_temps_all[6:,:,:] = extra_ctrl_temps       # 8 additional control runs from ControlTemps


### GET FILES ####
X, Y, Xtest, Ytest, latitude, longitude = get_train_test()

nlat, nlon = len(latitude), len(longitude)

### PREPROCESS X ####
scalerX = preprocessing.StandardScaler()
scalerX.fit(X)
X = scalerX.transform(X)
Xtest = scalerX.transform(Xtest)

N = X.shape[0]
p = X.shape[1]
print(p)

### SCALE Y ###
Yfull = Y
# reshape
Y = Yfull.reshape(Yfull.shape[0], Yfull.shape[1]*Yfull.shape[2])
ytest = Ytest.reshape(Ytest.shape[0], Ytest.shape[1]*Ytest.shape[2])


# scale
scalerY = preprocessing.StandardScaler()
scalerY.fit(Y)
y = scalerY.transform(Y)

# scale internal variability
ctrl_temps = ctrl_temps.reshape(ctrl_temps.shape[0], Yfull.shape[1]*Yfull.shape[2])
ctrl_mean = np.mean(ctrl_temps, axis=0)
ctrl_temps = ctrl_temps - ctrl_mean
ctrl_scaled = scalerY.transform(ctrl_temps)
y_internal = np.var(ctrl_scaled, axis=0)


def predict_function(i, y, ypred, sd, sd_gp, X, Xtest):
    y_i = y[:, i]
    N = len(y_i)
    y_i = y_i.reshape((N, 1))

    ### BUILD MODEL ###
    kern1 = GPy.kern.RBF(p, ARD=True)
    kern1.variance = 1e-5
    kern1.lengthscale = [0.89, 0.83, 0.94, 1.11, 0.37, 1., 0.96, 0.93, 0.75]         # start with lengthscales optimised from previous run
    kern2 = GPy.kern.Linear(p,ARD=True)
    kern2.variances = [0.63, 0.03, 0.01, 0.001, 0.007, 0.006, 0.003, 0.037, 0.015]
    kern = kern1 + kern2 
    m = GPy.models.GPRegression(X, y_i, kern)
    m.likelihood.variance.fix(y_internal[i])                 # fix internal variability as variance term
    m.optimize('bfgs', max_iters=1000)                       # optimise lengthscales/variances

    ### PREDICT ####
    ypred_i, sd_full_i = m.predict(Xtest)
    _, sd_GP_i = m.predict_noiseless(Xtest)
    Ntest = len(ypred_i)
    ypred[(i*Ntest):(i+1)*Ntest] = ypred_i.flatten()
    sd[(i*Ntest):(i+1)*Ntest] = sd_full_i.flatten()
    sd_gp[(i*Ntest):(i+1)*Ntest] = sd_GP_i.flatten()


if __name__ == '__main__':
    # Initialise ypred and sd 
    ytest = Ytest.reshape(Ytest.shape[0], Ytest.shape[1]*Ytest.shape[2])
    ypred = np.zeros((ytest.shape))
    sd = np.zeros((ytest.shape))
    sd_gp = np.zeros((ytest.shape))

    N = ytest.shape[0]
    k = ytest.shape[1]
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
        ypred[:, i] = ypred_arr[(i*Ntest):(i+1)*Ntest]
        sd[:, i] = sd_arr[(i*Ntest):(i+1)*Ntest] 
        sd_gp[:,i] = sd_gp_arr[(i*Ntest):(i+1)*Ntest]

    # un-scale
    Ypred = scalerY.inverse_transform(ypred)
    Ypred_flat = Ypred
    Y_p_Sd = scalerY.inverse_transform(sd+ypred)
    Y_m_Sd = scalerY.inverse_transform(ypred-sd)
    Sd = Y_p_Sd - Ypred
    Sd_GP = scalerY.inverse_transform(sd_gp+ypred) - Ypred

    # un-shape
    ypred = Ypred.reshape(Ypred.shape[0], Yfull.shape[1], Yfull.shape[2])
    ytest = Ytest
    sd = Sd.reshape(sd.shape[0], Yfull.shape[1], Yfull.shape[2])
    sd_gp = Sd_GP.reshape(sd.shape[0], Yfull.shape[1], Yfull.shape[2])    
    print("Saving")
    ### SAVE MODEL/PREDICTIONS ####
    save_file = '../../emulator_outputs/indepdims_GP_with_internalnoise.file'
    save_objs = {'ypred':ypred, 'ytest':ytest, 'sd':sd, 'sd_GP':sd_gp}
    pickle.dump(save_objs, file=open(save_file, "wb"))
    print("Saved objs in ", save_file)
    print("DONE")
    exit()
