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


### GET X FOR SOBOL ####
# Define the model inputs
CO2_current = 390 #ppm
CO2_mmr  =  CO2_ppm_MMR(CO2_current)
# Fix CO2 at this level
problem = {'num_vars': 9,
           'names': X_labels,
           'bounds': [[CO2_mmr, CO2_mmr],
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
param_values = saltelli.sample(problem, 100)
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


def predict_function(i, y, ypred, sd, X, Xparam):
    y_i = y[:, i]
    N = len(y_i)
    y_i = y_i.reshape((N, 1))

    ### BUILD MODEL ###
    kern1 = GPy.kern.RBF(p, ARD=True)
    kern1.variance = 0.000624
    kern1.lengthscale = [0.0048, 0.2836, 0.6138,  0.0164, 0.9702, 0.8029, 0.2043, 0.1746, 0.1804]
    kern2 = GPy.kern.Linear(p,ARD=True)
    kern2.variances = [0.6885, 0.0260, 0.0266,  0.0042, 0.0004, 0.0075, 0.0272, 0.0001, 0.0002]
    kern = kern1 + kern2
    m = GPy.models.GPRegression(X, y_i, kern)
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
        proc.start()
    for j in jobs:
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
    
    print("Saving")
    ### SAVE MODEL/PREDICTIONS ####
    save_file = '../../emulator_outputs/indepdims_GP_sobol_CO2_fixed.file'
    save_objs = {'X':Xparam, 'ypred':ypred, 'sd':sd, 'problem':problem}
    pickle.dump(save_objs, file=open(save_file, "wb"))
    print("Saved objs in ", save_file)
    print("DONE")
    exit()
