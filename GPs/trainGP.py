import sys
import os
home = os.getenv("HOME")
sys.path.insert(0, '../')
sys.path.insert(0, '../setup/')

### IMPORT PACKAGES ####
import GPy
import numpy as np
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
scalerX = preprocessing.StandardScaler()
scalerX.fit(X)
X = scalerX.transform(X)
Xtest = scalerX.transform(Xtest)

N = X.shape[0]
p = X.shape[1]
print(p)


# Regional grid
# Take regional means to explore linearity for these 
area = Area(longitude, latitude)
regions = ['Global', 'Europe', 'North_America', 'East_Asia', 'South_Asia', 'Africa', 'South_America', 'Tropics']
colors = ['grey', 'red', 'blue', 'green', 'darkgreen', 'orange', 'purple', 'darkblue', 'darkred', 'teal']
markers=['o', '*', '^', 's', 'd', 'h', '<', 'p', '>', 'X']

i=0
j=0
### SCALE Y ###
Yfull = Y
# reshape
Y = Yfull.reshape(Yfull.shape[0], Yfull.shape[1]*Yfull.shape[2])
# scale
scalerY = preprocessing.StandardScaler()
scalerY.fit(Y)
y = scalerY.transform(Y)

### BUILD MODEL ###
kern = GPy.kern.RBF(p,ARD=True)+GPy.kern.Linear(p,ARD=True)

print("kernel used: {}".format(kern))
m = GPy.models.GPRegression(X, y, kern)
m.optimize()
print("GP params optimised")
print(m)

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

