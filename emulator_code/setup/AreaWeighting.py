import numpy as np


def Area(longitude, latitude):
    """ Calculates area weighting given longitude and latitudes 
    returns the areas on same grid """
    nlon, nlat = len(longitude), len(latitude)

    areas = np.zeros((nlat, nlon))
    dlon = 2*np.pi/nlon     # Equal for all lats
    dlat = np.pi*(latitude[2]-latitude[1])/180.    # Equal for all lats
    # Contribution from cosine of latitude
    for j in range(nlat):
        dlatj = dlat*np.cos(np.pi*(latitude[j])/180.)
        areas[j, :] =dlon*dlatj

    return areas



