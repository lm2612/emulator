import netCDF4 as nc
import numpy as np

def open_file(filename, varname='temps'):
    """ Opens an nc file and returns the selected variable, along with the dimensions 
    which are the keys/runnames, time, latitude, longitude
    Args: filename (str) the nc file you want to open
          varname (str) variable you want to open. Currently only option is 'temps' for temperature """
    print('Opening '+filename)
    ds = nc.Dataset(filename,'r')
    longitude = ds.variables['longitude'][:]
    latitude = ds.variables['latitude'][:]
    temp = ds.variables[varname][:]
    filekeys = ds.variables['filekeys'][:]
    t = ds.variables['t'][:]
    return(filekeys, t, latitude, longitude, temp)

def get_inputs_outputs(output_filename = '../../emulator_files/AllTemps1-86.nc', 
                       input_filename = '../../design/design_matrix_1-86.csv',
                       ctrl = None, return_ctrl = False, varname='temps'):
    """Get all your inputs and outputs and also return the control file if requested.
    Args: * output_filename (str): nc file to open containing the temperature output to be emulated. Files in here are size N x ntime x nlat x nlon.
          * input filename  (str): csv file containing the design grid.  Files are size N x 9 for 9 input parameters describing emissions. Number of rows (N) must be the same as output_filename, otherwise you will encounter errors later.
          * ctrl: (None or a numpy array control control run of size ntime x nlat x nlon, to be subtracted from all rows in output_filename. Leave as None if you are reading in the training data (AllTemps1-86.nc) and the control run will be calculated from this file. If you are reading the test data, you must provide the control array from the training data file (see next point).
          * turn_ctrl (bool): Set this to True if you want to return the control run from the training data file, which you will need to do to subtract the control run from the Test data files, then you will return (X, Y, latitude, longitude, ctrl). Otherwise leave as false and return only (X, Y, latitude, longitude)
          * varname (str) variable you want to open. Currently only option is 'temps' for temperature

    """
    print("Getting inputs and outputs")
    #### Outputs ####
    filekeys, t, latitude, longitude, temp = open_file(output_filename, varname)
    sorted_inds = np.argsort(filekeys)
    sorted_keys = filekeys[sorted_inds]
    sorted_temps = temp[sorted_inds, :, :, :]

    # Average over time dimension (dim=1)
    average_temps = np.average(sorted_temps, axis=1)
    temps = average_temps
    if (ctrl is None):
        ctrl_keys = sorted_keys[0:6]
        ctrl_temps = temps[0:6, :, :]

        em_keys = sorted_keys[6:]
        em_temps = temps[6:, :, :]

        ctrl = np.average(ctrl_temps, axis=(0))
    else:
        em_keys = sorted_keys
        em_temps = temps

    # Get outputs by minusing the ctrl run from all outputs
    Y =  temps - ctrl

    #### Inputs ####
    design_matrix = (np.genfromtxt(input_filename, delimiter=',',
                 skip_header=1, usecols=range(1,10)))       # We always have 9 inputs (GHGs/aerosol perturbations)
    
    X = design_matrix

    if return_ctrl:
        # This allows you to also return the control run you have calculated, which may be needed later
        return(X, Y, latitude, longitude, ctrl)
    else:
        return(X, Y, latitude, longitude)


def get_train_test(train_output_filename = '../../emulator_files/AllTemps1-86.nc',
                   train_input_filename = '../../design/design_matrix_1-86.csv',
                   test_output_filename = '../../emulator_files/TestTemps1-18.nc',
 	           test_input_filename = '../../design/test_matrix_1-18.csv'):
    """ This function automatically returns the training and test data in the form 
    (X, Y, X_test, Y_test, latitude, longitude)
    Arguments are all strings, locations of train output and input files. Output files should be .nc files
    and input files should be .csv files.  """

    # Get training data, return control run as well.
    X, Y, latitude, longitude, ctrl = get_inputs_outputs(output_filename = train_output_filename,
                                                   input_filename = train_input_filename,
                                                   return_ctrl = True)
    # Get test data. Here we must provide the control run (ctrl) as an extra input to subtract from
    # perturbation runs. However, do not need to return the control run.
    X_test, Y_test, _, _ = get_inputs_outputs(output_filename = test_output_filename,
                                              input_filename = test_input_filename, 
                                              ctrl = ctrl, return_ctrl = False)

    return(X, Y, X_test, Y_test, latitude, longitude)
