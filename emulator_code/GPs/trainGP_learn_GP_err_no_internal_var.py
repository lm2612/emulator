### This file trains a Gaussian process without using the internal variability to fix the 
# GP error. It trains an independent GP per grid cell. 
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


### GET FILES ####
X, Y, Xtest, Ytest, latitude, longitude = get_train_test()

nlat, nlon = len(latitude), len(longitude)

### PREPROCESS X  ####
# scale
scalerX = preprocessing.StandardScaler()
scalerX.fit(X)
X = scalerX.transform(X)
Xtest = scalerX.transform(Xtest)

N = X.shape[0]
p = X.shape[1]

### PREPROCESS Y ###
Yfull = Y
# reshape
Y = Yfull.reshape(Yfull.shape[0], Yfull.shape[1]*Yfull.shape[2])
ytest = Ytest.reshape(Ytest.shape[0], Ytest.shape[1]*Ytest.shape[2])
# scale
scalerY = preprocessing.StandardScaler()
scalerY.fit(Y)
y = scalerY.transform(Y)

def predict_function(i, y, ypred, sd, sd_gp, X, Xtest):
    """ Build GP emulator for grid point i and predict at new unseen Xtest
       Args: i (int) points to index we are building emulator for 
             y (np array) training data outputs.
             ypred (multiprocessing array) mp array to be filled with predictions from emulator for new unseen Xtest
             sd (multiprocessing array) mp array to be filled with 1 s.d. frmo emulator
             sd_gp (multiprocessing array) mp array to be filled with 1 s.d. from emulator (i.e. same as above, only different when we prespecify the internal variability).
             X (np array) training data inputs
             Xtest(np array) new unseen inputs we want to predict.
            """
             
    y_i = y[:, i]
    N = len(y_i)
    y_i = y_i.reshape((N, 1))

    ### BUILD MODEL ###
    kern1 = GPy.kern.RBF(p, ARD=True)
    kern1.variance = 1e-1
    kern1.lengthscale = [0.89, 0.83, 0.94, 1.11, 0.37, 1., 0.96, 0.93, 0.75]       # start optimisation from here
    kern2 = GPy.kern.Linear(p,ARD=True)
    kern2.variances = [0.63, 0.03, 0.01, 0.001, 0.007, 0.006, 0.003, 0.037, 0.015]
    kern = kern1 + kern2     # Kernel = Linear + RBF. Note I have set some initial variances and lengthscales based on past runs to start optimisation from a good initial point
    m = GPy.models.GPRegression(X, y_i, kern)     # Set up the GP model
    m.optimize('bfgs', max_iters=1000)            # Optimise GP model

    ### PREDICT ####
    ypred_i, sd_full_i = m.predict(Xtest)         # Predict and get the noise associated with prediction
    _, sd_GP_i = m.predict_noiseless(Xtest)       # Noiseless prediction (relevant when we preset a fixed noise due to internal variability)
    Ntest = len(ypred_i)
    # Save to grids we have assigned
    ypred[(i*Ntest):(i+1)*Ntest] = ypred_i.flatten()
    sd[(i*Ntest):(i+1)*Ntest] = sd_full_i.flatten()
    sd_gp[(i*Ntest):(i+1)*Ntest] = sd_GP_i.flatten()


if __name__ == '__main__':
    # Initialise ypred and sd 
    ytest = Ytest.reshape(Ytest.shape[0], Ytest.shape[1]*Ytest.shape[2])
    ypred = np.zeros((ytest.shape))
    sd = np.zeros((ytest.shape))
    sd_gp = np.zeros((ytest.shape))
    # ypred, sd, sd_gp are all arrays that we have initialised with zeros that will be filled with the predictions and standard deviations from the emulator. sd_gp is only relevant when we specify the internal variability as a fixed noise term (not done in this file).

    N = ytest.shape[0]
    k = ytest.shape[1]
    print(k)

    # set up arrays that share memory
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

    # Reshape arrays to fit original numpy array format specified
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
    save_file = '../../emulator_outputs/indepdims_GP_no_internalnoise.file'
    save_objs = {'ypred':ypred, 'ytest':ytest, 'sd':sd, 'sd_GP':sd_gp}
    pickle.dump(save_objs, file=open(save_file, "wb"))
    print("Saved objs in ", save_file)
    print("DONE")
    exit()
