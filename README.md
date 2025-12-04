# Emission Response Gaussian Process Emulator
Emulator built to predict short-term temperature response of the HadGEM3 global climate model to 9 different emission inputs. 
We ran HadGEM3 with a Latin Hybercube design across to emissions inputs to generate a training dataset of size 80 and a test dataset of size 18.

### Dependencies
This code uses `GPy`, developed and maintained by Sheffield Machine Learning group (https://github.com/SheffieldML/GPy). You will also need `scikit-learn` for preprocessing data, `SAlib` for sensitivity analysis code, `multiprocessing` for training GPs on multiple cores, and `matplotlib`, `cartopy` for plotting. See full list of packages required in the conda environment file `environment.yml`.

### Quickstart
Run the jupyter notebooks in `notebooks` to see a demo of how the code works.

## Inputs
Inputs to the emulator are emissions, both greenhouse gases and aerosol precursors.
1. Global CO2 concentration (parts per million), NOTE: Input files contain the concentration as mass mixing ratio (MMR) as these are the inputs to the GCM. These can be converted to parts per million, with function `CO2_MMR_ppm(MMR)` in `setup/conv_MMR_ppm.py`.
2. Global CH4 concentration (parts per billion), NOTE: Input files contain the concentration as mass mixing ratio (MMR) as these are the inputs to the GCM. These can be converted to parts per billion, with function `CH4_MMR_ppb(MMR)` in `setup/conv_MMR_ppm.py`.  
3. SO2 over Europe (scaling factor over entire region)
4. SO2 over Europe (scaling factor over entire region)
5. SO2 over Europe (scaling factor over entire region)
6. SO2 over Europe (scaling factor over entire region)
7. SO2 over Europe (scaling factor over entire region)
8. SO2 over Europe (scaling factor over entire region)
9. OC/BC from fires over Tropics (scaling factor over entire region)

Input files: 

Emissions values described above stored in csv files, each row is a one simulation, each column is one of the inputs in the order listed above. These can be found in in `design/`
* `design_matrix_1-86.csv` : The inputs for the training dataset of size 80, plus an additional 6 simulations for the control run that can be used for training or to estimate internal variability.
* `test_matrix_1-18.csv`   : The inputs for the test dataset of size 18

## Outputs
Outputs are global temperature response maps after five years of simulation. We have saved just these files containing the surface temperature in Kelvin. Size of arrays are 5 x 144 x 192 = time x latitude x longitude. The time dimension is over 5 years. In this study we average over the 5 year response to reduce internal variability, then predict each grid cell independently (i.e. 144 x 192 = 27648 independent emulators), but this could be adapted to predict global mean, regional means,  etc.

Output files:
* `AllTemps86.nc`   : The output from the 80 training simulations plus an additional 6 simulations for the control run.
* `TestTemps18.nc`  : The output from the 18 test simulations.
* `ControlTemps.nc` : The output from 8 additional control simulations.

The temperature maps are saved as netCDF files with dimesions:
* `time`: annual means over each year
* `latitude, longitude`: Define grid box centre. This is what we use in this code.
* `latitude_1, longitude_1`: Define grid box edges. These are not used in this code.
* `files`: labels each file for the training/test set. 

Variables:
* `temps`: surface temperature in Kelvin

## Directory Structure
You will need to download the training and test outputs from the zenodo repository (10.5281/zenodo.6209322).
* `design/` contains the training and test inputs as csv files.
* `emulator_files/` contains the training and test outputs as nc files (not in this repo, but download from zenodo).
* `emulator_output/` contains outputs from the emulator, generated when you run the code. 
* `emulator_code/` contains all the code necessary to reproduce results

Within `emulator_code/`:

* `setup/` contains utilities and code for setting up the data, including opening files, defining regions for regional analysis, converting greenhouse gas MMRs to parts per million, and so on.
* `plotting/` contains a simple function for plotting maps
* `notebooks/` contains a few simple notebooks for getting started, including creating an interactive widget
* `GPs/` contains GP training code. This is where most of the training goes on
* `plots/` contains plotting functions to replicate plots in paper.


## Useful functions and tips:

To open files: In `setup/read_files.py` you will find the files necessary to get the inputs and outputs to get started. If you use the same structure for storing inputs/outputs as above, you can open files automatically with `get_train_test()`. 

Global means: If you have a latitude-longitude array `(144 x 192)` and would like a global mean, you must take an area weighted mean as grid-cells are not all equal in size (by latitude). Using the function `Area` in `setup/AreaWeighting.py`, to get an area weighting, then use `np.average` and set the weights to this area weighting array:
```
area = Area(lon, lat)
global_mean = np.average(var, weights = area)
```
    
Regional means: Similarly for a regional mean you must take an area weighted mean. Many regions are already saved in setup/RegionLatitudes.py where you will find a dictionary of regions and the closest boxes to these (all rectangular, no unusual shapes). In this dictionary the key is the region name and the value is a `(lat_min, lat_max, lon_min, lon_max)` e.g. `{'Europe':(350., 40., 35., 70.)}`. You can then calculate a `144 x 192` grid containing 1s inside this box and 0s outside, with `AreaRegion` function in `setup/DefineRegions.py`. This grid can then be used to set the weights in `np.average`:

```
area = Area(lon, lat)
region = 'Europe'
region_grid =  DefineRegion(region, lon, lat)
regional_mean = np.average(var, weights = area*region_grid)
```

## Authors
All code was written by Laura Mansfield, during PhD at University of Reading with collaborators at Imperial College London. 
Please create an issue for any questions.

