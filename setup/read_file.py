import netCDF4 as nc
import numpy as np

def open_file(filename, varname='temps'):
    print('Opening '+filename)
    ds = nc.Dataset(filename,'r')
    longitude = ds.variables['longitude'][:]
    latitude = ds.variables['latitude'][:]
    temp = ds.variables[varname][:]
    filekeys = ds.variables['filekeys'][:]
    t = ds.variables['t'][:]
    return(filekeys, t, latitude, longitude, temp)

def get_inputs_outputs(output_filename = '../../emulator_files/AllTemps.nc', 
                       input_filename = '../../design/design_matrix_1-40.csv',
                       ctrl = None, return_ctrl = False, varname='temps'):
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

    # Get outputs by minusing the ctrl run
    Y = em_temps - ctrl

    #### Inputs ####
    design_matrix = (np.genfromtxt(input_filename, delimiter=',',
                 skip_header=1, usecols=range(1,10)))
    
    X = design_matrix

    if return_ctrl:
        return(X, Y, latitude, longitude, ctrl)
    else:
        return(X, Y, latitude, longitude)


def get_train_test():
    X, Y, latitude, longitude, ctrl = get_inputs_outputs(output_filename = 
                                                   '../../emulator_files/AllTemps1-80.nc', 
                                                   input_filename = 
                                                   '../../design/design_matrix_1-80.csv', 
                                                   return_ctrl = True)

    X_test, Y_test, _, _ = get_inputs_outputs(output_filename =
                                              '../../emulator_files/TestTemps.nc',
                                              input_filename =  
                                              '../../design/test_matrix_1-10.csv',
                                              ctrl = ctrl)

    return(X, Y, X_test, Y_test, latitude, longitude)
