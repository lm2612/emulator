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

def get_data_subset(X_orig, Y_orig, N):
    return( X_orig[:N,:], Y_orig[:N, :])

### GET FILES ####
X, Y, Xtest, Ytest, latitude, longitude = get_train_test()

X_orig = X
Y_orig = Y

Nsims = int(sys.argv[1])
print(Nsims)

X, Y = get_data_subset(X_orig, Y_orig, Nsims)
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


def predict_function(i, y, ypred, sd, X, Xtest):
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
    m.optimize('bfgs', max_iters=100)

    ### PREDICT ####
    ypred_i, sd_i = m.predict(Xtest)
    Ntest = len(ypred_i)
    ypred[(i*Ntest):(i+1)*Ntest] = ypred_i.flatten()
    sd[(i*Ntest):(i+1)*Ntest] = sd_i.flatten()


if __name__ == '__main__':
    # Initialise ypred and sd 
    ytest = Ytest.reshape(Ytest.shape[0], Ytest.shape[1]*Ytest.shape[2])
    ypred = np.zeros((ytest.shape))
    sd = np.zeros((ytest.shape))
    
    N = ytest.shape[0]
    k = ytest.shape[1]
    print(k)
    # set up arrays
    inds = mp.Array('i', range(k))
    ypred_arr = mp.sharedctypes.RawArray('d', N*k)
    sd_arr = mp.sharedctypes.RawArray('d', N*k)
    jobs = []
    for i in range(k):
        proc= mp.Process(target=predict_function, args=(i, y, ypred_arr, sd_arr, X, Xtest))
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
    ytest = Ytest
    sd = Sd.reshape(sd.shape[0], Yfull.shape[1], Yfull.shape[2])
    
    print("Saving")
    ### SAVE MODEL/PREDICTIONS ####
    save_file = '../../emulator_outputs/indepdims_GP_{}.file'.format(Nsims)
    save_objs = {'ypred':ypred, 'ytest':ytest, 'sd':sd}
    pickle.dump(save_objs, file=open(save_file, "wb"))
    print("Saved objs in ", save_file)
    print("DONE")
    exit()
