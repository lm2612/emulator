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
from SALib.sample import saltelli
from SALib.analyze import sobol

### MY PYTHON FILES ####
from read_file import *
from AreaWeighting import *
from RegionLatitudes import *
from DefineRegions import *
from X_labels import *
from conv_MMR_ppm import *

### DEFINE RANDOM SEED ####
random_seed = 1

### GET FILES ####
X, Y, Xtest, Ytest, latitude, longitude = get_train_test()

nlat, nlon = len(latitude), len(longitude)

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
ctrl_keys = sorted_keys[0:6]
ctrl_temps = temps[0:6, :, :]


### GET X FOR SOBOL ####
# Define the model inputs
CO2_2x = 564 #ppm
CO2_mmr  =  CO2_ppm_MMR(CO2_2x)
CO2_mmr = 1.267E-3
print(CO2_mmr)
problem = {'num_vars': 9,
           'names': X_labels,
           'bounds': [[4.318E-4, CO2_mmr],
                      [1.37E-7, 1.799E-6],
                      [0., 5. ],
                      [0., 3. ],
                      [0., 2. ],
                      [0., 3. ],
                      [0., 3. ],
                      [0., 7. ],
                      [0., 2. ]]
}
# Generate samples
param_values = saltelli.sample(problem, 1000)
print(param_values)
print(param_values.shape)


### PREPROCESS X ####
scalerX = preprocessing.StandardScaler()
scalerX.fit(X)
X = scalerX.transform(X)
Xparam = scalerX.transform(param_values)

N = X.shape[0]
p = X.shape[1]
print(p)

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


def predict_function(i, y, ypred, sd, X, Xparam):
    y_i = y[:, i]
    N = len(y_i)
    y_i = y_i.reshape((N, 1))

    ### BUILD MODEL ###
    kern1 = GPy.kern.RBF(p, ARD=True)
    kern1.variance = 0.007
    kern1.lengthscale = [0.49, 0.29, 0.53, 0.82, 0.04, 0.65, 0.46, 0.44, 0.15]
    kern2 = GPy.kern.Linear(p,ARD=True)
    kern2.variances = [0.62, 0.02, 0.01, 0.001, 0.01, 0.005, 0.002, 0.03, 0.015]
    kern = kern1 + kern2
    m = GPy.models.GPRegression(X, y_i, kern)
    m.likelihood.variance.fix(y_internal[i])
    m.optimize('bfgs', max_iters=200)

    ### PREDICT ####
    ypred_i, sd_i = m.predict(Xparam)
    Ntest = len(ypred_i)
    ypred[(i*Ntest):(i+1)*Ntest] = ypred_i.flatten()
    sd[(i*Ntest):(i+1)*Ntest] = sd_i.flatten()


if __name__ == '__main__':
    # Initialise ypred and sd 
    ypred = np.zeros((Xparam.shape[0], Y.shape[1]))
    sd = np.zeros((Xparam.shape[0], Y.shape[1] ))
    
    N = Xparam.shape[0]
    k = Y.shape[1]
    print(k)
    # set up arrays
    inds = mp.Array('i', range(k))
    ypred_arr = mp.sharedctypes.RawArray('d', N*k)
    sd_arr = mp.sharedctypes.RawArray('d', N*k)
    jobs = []
    for i in range(k):
        proc= mp.Process(target=predict_function, args=(i, y, ypred_arr, sd_arr, X, Xparam))
        jobs.append(proc)
    #    proc.start()
    #    proc.join()
    for j in jobs:
        j.start()
        j.join()
    print("Run all indices complete")
    print(ypred_arr)
    print(sd_arr)
    Ntest= N

    for i in range(k):
        ypred[:, i] = ypred_arr[(i*Ntest):(i+1)*Ntest] #np.frombuffer(ypred_arr[i:(i+N)])
        sd[:, i] = sd_arr[(i*Ntest):(i+1)*Ntest] 

    # un-scale
    Ypred = scalerY.inverse_transform(ypred)
    Ypred_flat = Ypred
    Y_p_Sd = scalerY.inverse_transform(sd+ypred)
    Y_m_Sd = scalerY.inverse_transform(ypred-sd)
    Sd = Y_p_Sd - Ypred

    # un-shape
    ypred = Ypred.reshape(Ypred.shape[0], Yfull.shape[1], Yfull.shape[2])
    sd = Sd.reshape(sd.shape[0], Yfull.shape[1], Yfull.shape[2])
    

    Si1_map = np.zeros((9, ypred.shape[1], ypred.shape[2]))
    ST_map = np.zeros((9, ypred.shape[1], ypred.shape[2]))
    for i in range(ypred.shape[1]):
        for j in range(ypred.shape[2]):
            Si = sobol.analyze(problem, ypred[:,i,j], print_to_console=False)
            Si1_map[:, i,j] = Si['S1']
            ST_map[:, i,j] = Si['ST']

    pickle.dump({"Si1_map":Si1_map, "SiT_map":ST_map }, file=open('../../emulator_outputs/Sobolmap.file',"wb"))

    print("DONE")
    exit()
