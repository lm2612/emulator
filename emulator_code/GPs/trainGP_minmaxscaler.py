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

### PREPROCESS X ####
scalerX = preprocessing.MinMaxScaler()
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
# scale
scalerY = preprocessing.MinMaxScaler()
scalerY.fit(Y)
y = scalerY.transform(Y)

### BUILD MODEL ###
kern = GPy.kern.RBF(p,ARD=True)+GPy.kern.Linear(p,ARD=True)

print("kernel used: {}".format(kern))
m = GPy.models.GPRegression(X, y, kern)
m.optimize('bfgs', max_iters=1000)
print("GP params optimised")
print(m)
print(m.kern)
print(m.kern.rbf.lengthscale)
print(m.kern.linear.variances)

### PREDICT ####
ypred, sd = m.predict(Xtest)

# un-scale
Ypred = scalerY.inverse_transform(ypred)
Ypred_flat = Ypred
Ytest_flat = Ytest.reshape(Ytest.shape[0], Ytest.shape[1]*Ytest.shape[2])

# un-shape
ypred = Ypred.reshape(Ypred.shape[0], Yfull.shape[1], Yfull.shape[2])
ytest = Ytest

### SAVE MODEL/PREDICTIONS ####
save_file = '../../emulator_outputs/orig_GP.file'
save_objs = {'m':m, 'kern':kern, 'ypred':ypred, 'ytest':ytest, 'sd':sd}
pickle.dump(save_objs, file=open(save_file, "wb"))
print("Saved objs in ", save_file)
print("DONE")
